import argparse
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms

from model import resnet18


def setup_dist():
    """Initialize the distributed process group and return device mesh."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    dist.init_process_group(backend="gloo" if not torch.cuda.is_available() else "nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def build_device_mesh(use_gpu):
    """Build a 1-D device mesh for FSDP."""
    device_type = "cuda" if use_gpu else "cpu"
    mesh = init_device_mesh(
        device_type,
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=("fsdp",),
    )
    return mesh


def cleanup():
    """Destroy the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_dataloader(
    batch_size: int,
    rank: int,
    world_size: int,
    data_root: str = "./data",
    is_train: bool = True,
):
    """Build CIFAR-10 train or test loader with distributed sampler."""
    transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ] if is_train else []
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=is_train,
        download=True,
        transform=transforms.Compose(transform_list),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def wrap_model(model, use_gpu):
    """Wrap model with FSDP using device mesh."""
    device_mesh = build_device_mesh(use_gpu)
    device_type = "cuda" if use_gpu else "cpu"

    fsdp_model = FSDP(
        model,
        device_mesh=device_mesh,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.device(device_type),
        limit_all_gathers=True,
    )
    return fsdp_model


def save_checkpoint(model, optimizer, epoch, output_dir, rank):
    """Save model checkpoint (only from rank 0)."""
    state_dict = model.state_dict()
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, "checkpoint.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print(f"[Rank {rank}] Checkpoint saved to {ckpt_path}")


def load_checkpoint(model, optimizer, ckpt_path, rank):
    """Load model checkpoint. Returns the epoch to resume from, or 0 if no checkpoint.

    Handles checkpoints with only model_state_dict (no optimizer/epoch keys),
    which is the format used by federated learning rounds.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Rank {rank}] Loaded model weights from {ckpt_path}")



def train_one_epoch(model, loader, optimizer, criterion, rank, use_gpu, max_batches=None):
    """Run a single training epoch."""
    model.train()
    device = torch.device("cuda" if use_gpu else "cpu")
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    if rank == 0:
        print(f"  Loss: {avg_loss:.4f} | Acc@1: {accuracy:.2f}%")
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, rank, use_gpu, max_batches=None):
    """Run evaluation on the test set."""
    model.eval()
    device = torch.device("cuda" if use_gpu else "cpu")
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    if rank == 0:
        print(f"  Test Loss: {avg_loss:.4f} | Test Acc@1: {accuracy:.2f}%")
    return avg_loss, accuracy


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 FSDP Training")
    p.add_argument("--batch-size", type=int, default=128, help="Per-rank batch size")
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    p.add_argument("--data-root", type=str, default="./data", help="Path to CIFAR-10 data")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save checkpoints and logs",
    )
    p.add_argument("--save-freq", type=int, default=5, help="Checkpoint save frequency (epochs)")
    p.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="Limit batches per epoch (train and eval). None = full dataset.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size = setup_dist()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    backend = "nccl" if use_gpu else "gloo"

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Device: {device} (backend: {backend})")
        print(f"Batch size (per rank): {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Epochs: {args.epochs}, LR: {args.lr}")
        print(f"Max batches/epoch: {args.max_batches_per_epoch}")
        print(f"Output dir: {args.output_dir}")

    # Model
    model = resnet18()
    model = wrap_model(model, use_gpu)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # Data loaders
    train_loader = get_dataloader(args.batch_size, rank, world_size, args.data_root, is_train=True)
    test_loader = get_dataloader(args.batch_size, rank, world_size, args.data_root, is_train=False)

    # Resume from checkpoint
    if args.resume:
        load_checkpoint(model, optimizer, args.resume, rank)

    # Training loop
    for epoch in range(0, args.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, rank, use_gpu,
            max_batches=args.max_batches_per_epoch,
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, rank, use_gpu,
            max_batches=args.max_batches_per_epoch,
        )

        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            save_checkpoint(model, optimizer, epoch, args.output_dir, rank)

    if rank == 0:
        print("Training complete.")

    cleanup()


if __name__ == "__main__":
    main()

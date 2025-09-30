"""Standalone CIFAR-10 training script using the same model and preprocessing as in secaggexample.task.

This script trains a Resnet-18 on the full CIFAR-10 dataset
loaded via torchvision (no federated learning involved).
"""

import os
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

# =====================
# Hyperparameters
# =====================
DATA_DIR = os.environ.get("CIFAR10_DATA_DIR", "./data")
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = os.environ.get("MODEL_SAVE_PATH", "./cifar10_net.pth")


def get_resnet18(num_classes=10):
    """Create and return a ResNet18 model for CIFAR-10."""
    model = models.resnet18(weights=None)  # Weights=None means random initialization
    # Adjust the first conv layer for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove the first max pooling as CIFAR-10 images are small
    # Replace the final fully connected layer for 10 classes
    model.fc = nn.Linear(512, num_classes)
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_dataloaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test dataloaders using torchvision."""
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def train_one_epoch(
    model: torch.nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main() -> None:
    set_seed(SEED)

    # Create a unique directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/standalone_cifar10_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    print(f"Logging to: {log_dir}")

    train_loader, test_loader = get_dataloaders(BATCH_SIZE, NUM_WORKERS)

    model = get_resnet18().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # Using CosineAnnealingLR for smooth learning rate decay
    # eta_min is the minimum learning rate, which will be 1% of the initial learning rate
    eta_min = LEARNING_RATE * 0.01
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=eta_min)
    print(f"Using CosineAnnealingLR with T_max={NUM_EPOCHS}, eta_min={eta_min:.2e}")

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), SAVE_PATH)

        # Log metrics to TensorBoard
        writer.add_scalar("server/fit/weighted_accuracy", train_acc, epoch)
        writer.add_scalar("server/fit/weighted_loss", train_loss, epoch)
        writer.add_scalar("server/eval/accuracy", test_acc, epoch)
        writer.add_scalar("server/eval/loss", test_loss, epoch)
        writer.add_scalar("server/lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

    print(f"Training complete. Best test accuracy: {best_acc:.4f}. Model saved to: {SAVE_PATH}")
    writer.close()


if __name__ == "__main__":
    main()

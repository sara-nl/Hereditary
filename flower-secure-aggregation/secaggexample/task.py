"""secaggexample: A Flower with SecAgg+ app."""

import random
from collections import OrderedDict
from unittest.mock import Mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import datasets
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms


def get_resnet18(num_classes=10):
    """Create and return a ResNet18 model for CIFAR-10."""
    model = models.resnet18(weights=None)  # Weights=None means random initialization
    # Adjust the first conv layer for CIFAR-10's 32x32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove the first max pooling as CIFAR-10 images are small
    # Replace the final fully connected layer for the number of classes
    model.fc = nn.Linear(512, num_classes)
    return model


def make_net(seed=42):
    return get_resnet18()


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_data(partition_id: int, num_partitions: int, batch_size: int, is_demo: bool):
    """Load partitioned CIFAR-10 data using torchvision with a fixed seed.

    Clients use the FULL local partition for training. For metrics, we report
    training accuracy by returning the same DataLoader for both training and
    validation. Partitioning is deterministic across runs given the fixed seed.
    """
    print("partition_id: ", partition_id)
    if is_demo:
        trainloader = Mock(dataset=[0])
        return trainloader, trainloader

    seed = 42
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4)
        ]
    )

    # Load full CIFAR-10 training set
    full_train = datasets.CIFAR10(root="data", train=True, download=True, transform=pytorch_transforms)

    # Deterministic partitioning
    num_samples = len(full_train)
    all_indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)
    parts = np.array_split(all_indices, num_partitions)
    part_indices = parts[partition_id].tolist()

    subset = Subset(full_train, part_indices)
    trainloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    # Return the same loader for validation to compute training accuracy
    return trainloader, trainloader


def load_cifar10_test(batch_size: int, data_dir: str = "data") -> DataLoader:
    """Load the global CIFAR-10 test set for server-side evaluation.

    This uses torchvision's CIFAR10 test split. Assumes the data is already
    present under `data_dir` (no download performed).
    """
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    testset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=pytorch_transforms,
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return testloader


def train(net, trainloader, valloader, epochs, learning_rate, device, data_percentage: float):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4  # Add weight decay for better regularization
    )
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            if random.random() > data_percentage:
                continue
            if isinstance(batch, dict):
                images = batch["img"]
                labels = batch["label"]
            else:
                images, labels = batch
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()

    val_loss, val_acc = test(net, valloader, device)

    # Since valloader == trainloader, these represent training metrics
    results = {
        "loss": val_loss,
        "accuracy": val_acc,
    }
    return results


def test(net, testloader, device):
    """Evaluate the model on a dataset.

    Supports both dict-based batches (from `flwr_datasets`) with keys
    `"img"` and `"label"`, as well as standard tuple batches (images, labels)
    from torchvision datasets.
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            if isinstance(batch, dict):
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

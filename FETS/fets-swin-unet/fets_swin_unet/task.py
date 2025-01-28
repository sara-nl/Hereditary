"""FETS-swin-unet: A Flower / PyTorch app."""

import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai import data, transforms
from monai.data import decollate_batch
from torch.utils.data import DataLoader


def datafold_read_csv(partition, train_percentage=0.8, seed=42):
    """Read data partitioning information from a CSV file.

    Args:
        fold (int): Fold number to use for validation (matches Partition_ID)

    Returns:
        tuple: Lists of training and validation data dictionaries
    """
    # read basedir from environment variable
    basedir = os.getenv("FETS_DATA_DIR")

    # Read the CSV file
    csv_path = os.path.join(basedir, "partitioning_2.csv")
    df = pd.read_csv(csv_path)
    df = df[df["Partition_ID"] == partition]

    # Validate required columns exist
    required_cols = ["Subject_ID", "Partition_ID"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Create data dictionaries for each subject
    data_dicts = []
    for _, row in df.iterrows():
        subject_id = row["Subject_ID"]

        # Create subject directory path
        subject_dir = os.path.join(basedir, subject_id)

        # Create data paths for this subject with correct modalities
        # Note: MONAI expects a list of paths under the "image" key
        subject_dict = {
            "fold": partition,
            "Subject_ID": subject_id,
            "image": [  # List of modality paths under "image" key
                os.path.join(subject_dir, f"{subject_id}_flair.nii.gz"),
                os.path.join(subject_dir, f"{subject_id}_t1.nii.gz"),
                os.path.join(subject_dir, f"{subject_id}_t1ce.nii.gz"),
                os.path.join(subject_dir, f"{subject_id}_t2.nii.gz"),
            ],
            "label": os.path.join(subject_dir, f"{subject_id}_seg.nii.gz"),  # Single path for label
        }
        data_dicts.append(subject_dict)

    # Split into training and validation based on fold
    tr = []
    val = []
    np.random.seed(seed)
    train_idx = np.random.choice(len(data_dicts), int(len(data_dicts) * train_percentage), replace=False)
    for i, d in enumerate(data_dicts):
        if i in train_idx:
            tr.append(d)
        else:
            val.append(d)
    return tr, val


def load_data(batch_size, fold, roi):
    train_files, validation_files = datafold_read_csv(partition=fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
                allow_smaller=False,
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train(model, loader, optimizer, epochs, loss_func, batch_size, device):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for epoch in range(epochs):
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model(data)
            loss = loss_func(logits, target)
            loss.backward()
            optimizer.step()
            run_loss.update(loss.item(), n=batch_size)
            print(
                "Epoch {} {}/{}".format(epoch, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
    # Convert to float to ensure compatibility with Flower's metrics system
    return float(run_loss.avg)


def test(
    model,
    loader,
    acc_func,
    device,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{}".format(idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    # Return mean dice score across all classes
    return float(np.mean(run_acc.avg)), run_acc.avg.tolist()


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

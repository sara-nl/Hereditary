"""Data loading and partitioning utilities for federated clients."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from fed_kmeans_flower.data.clef_loading import get_data

logger = logging.getLogger(__name__)


@dataclass
class DataStatistics:
    """Statistics about a dataset."""

    num_samples: int
    num_features: int
    feature_means: np.ndarray
    feature_stds: np.ndarray
    data_range: Tuple[np.ndarray, np.ndarray]  # (min_values, max_values)


def validate_data_format(data: np.ndarray, labels: Optional[np.ndarray] = None) -> bool:
    """Validate that data is in the correct format for federated k-means.

    Args:
        data: Input data array, should be 2D (n_samples, n_features)
        labels: Optional ground truth labels for evaluation

    Returns:
        True if data format is valid, False otherwise
    """
    try:
        # Check if data is numpy array
        if not isinstance(data, np.ndarray):
            logger.error("Data must be a numpy array")
            return False

        # Check if data is 2D
        if len(data.shape) != 2:
            logger.error(f"Data must be 2D, got shape {data.shape}")
            return False

        # Check for minimum samples
        if data.shape[0] == 0:
            logger.error("Data must contain at least one sample")
            return False

        # Check for minimum features
        if data.shape[1] == 0:
            logger.error("Data must contain at least one feature")
            return False

        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.error("Data contains NaN or infinite values")
            return False

        # Validate labels if provided
        if labels is not None:
            if not isinstance(labels, np.ndarray):
                logger.error("Labels must be a numpy array")
                return False

            if len(labels.shape) != 1:
                logger.error(f"Labels must be 1D, got shape {labels.shape}")
                return False

            if len(labels) != data.shape[0]:
                logger.error(f"Labels length {len(labels)} doesn't match data samples {data.shape[0]}")
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating data format: {e}")
        return False


def get_data_statistics(data: np.ndarray) -> DataStatistics:
    """Compute statistics for a dataset.

    Args:
        data: Input data array, shape (n_samples, n_features)

    Returns:
        DataStatistics object containing dataset statistics
    """
    if not validate_data_format(data):
        raise ValueError("Invalid data format")

    num_samples, num_features = data.shape

    # Compute basic statistics
    feature_means = np.mean(data, axis=0)
    feature_stds = np.std(data, axis=0)
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)

    return DataStatistics(
        num_samples=num_samples,
        num_features=num_features,
        feature_means=feature_means,
        feature_stds=feature_stds,
        data_range=(min_values, max_values),
    )


def _load_test_data(partition_id: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load test data from .npz files in a directory."""
    data_path = os.getenv("TEST_DATA_PATH")
    if not data_path:
        raise ValueError("TEST_DATA_PATH environment variable must be set")

    if partition_id is None:
        raise ValueError("partition_id required for test_data")

    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    partition_file = data_dir / f"client_{partition_id}.npz"
    if not partition_file.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_file} (partition_id={partition_id})")

    loaded = np.load(partition_file)
    data = loaded["data"]
    labels = loaded.get("labels", None)
    data_info = {}
    return data, labels, data_info


def _load_als_data(partition_id: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    logger.info(f"ALS data loader called for partition {partition_id}")

    base_path = os.getenv("CLEF_DATA_PATH")
    if not base_path:
        raise ValueError("CLEF_DATA_PATH environment variable must be set")

    partition_mapping = {
        0: "T",
        1: "L",
        2: "U",
    }
    partition_path = os.path.join(base_path, partition_mapping[partition_id], "datasetC")
    X_train, y_train, X_test, y_test, continuous_indices = get_data(partition_path)
    data_info = {"continuous_indices": continuous_indices}
    return X_train, y_train, data_info
    

def load_client_data(
    client_id: str, data_format: str = "test_data", partition_id: Optional[int] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load data for a specific client based on the dataset name.

    Args:
        client_id: Unique identifier for the client
        data_format: Name of the dataset ("test_data", "ALS")
        partition_id: Partition ID for partitioned datasets

    Returns:
        Tuple of (data, labels) where labels may be None

    Raises:
        ValueError: If data format is invalid or data cannot be loaded
    """
    try:
        if data_format == "test_data":
            data, labels, data_info = _load_test_data(partition_id)
        elif data_format == "ALS":
            data, labels, data_info = _load_als_data(partition_id)
        else:
            raise ValueError(f"Unsupported dataset: {data_format}")

        # Validate the loaded data
        if not validate_data_format(data, labels):
            raise ValueError(f"Invalid data format for client {client_id}")

        logger.info(
            f"Loaded {data_format} for client {client_id}: "
            f"{data.shape[0]} samples, {data.shape[1]} features"
        )

        return data, labels, data_info

    except Exception as e:
        logger.error(f"Error loading {data_format} for client {client_id}: {e}")
        raise

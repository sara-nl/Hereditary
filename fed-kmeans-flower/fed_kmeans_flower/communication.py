"""Message serialization utilities for numpy arrays and cluster data."""

from typing import Any, Dict, Union

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord


def create_message_content(
    arrays: Dict[str, np.ndarray] = None,
    metrics: Dict[str, Union[float, int, bool]] = None,
    config: Dict[str, Any] = None,
) -> Dict[str, Union[ArrayRecord, MetricRecord, ConfigRecord]]:
    """Create message content for Flower messaging.

    Args:
        arrays: Dictionary of numpy arrays to include
        metrics: Dictionary of metrics to include (int, float, bool values)
        config: Dictionary of configuration values to include

    Returns:
        Dictionary with ArrayRecord, MetricRecord, and/or ConfigRecord
    """
    from flwr.app import Array

    content = {}

    if arrays is not None:
        # Convert numpy arrays to Array objects for ArrayRecord
        array_dict = {}
        for key, arr in arrays.items():
            if isinstance(arr, np.ndarray):
                array_dict[key] = Array(ndarray=arr)
            else:
                array_dict[key] = arr
        content["arrays"] = ArrayRecord(array_dict=array_dict)

    if metrics is not None:
        # Convert values to compatible types for MetricRecord
        converted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                converted_metrics[key] = int(value)
            elif isinstance(value, str):
                # Store strings in config instead of metrics
                if config is None:
                    config = {}
                config[key] = value
            else:
                converted_metrics[key] = value
        if converted_metrics:
            content["metrics"] = MetricRecord(converted_metrics)

    if config is not None:
        content["config"] = ConfigRecord(config)

    return content


def extract_message_content(content: Dict[str, Union[ArrayRecord, MetricRecord, ConfigRecord]]) -> Dict[str, Any]:
    """Extract content from Flower message.

    Args:
        content: Message content with Flower records

    Returns:
        Dictionary with extracted arrays, metrics, and config
    """
    extracted = {}
    
    if "arrays" in content:
        # Convert Array objects back to numpy arrays
        arrays_dict = {}
        for key, array_obj in content["arrays"].items():
            if hasattr(array_obj, "numpy"):
                arrays_dict[key] = array_obj.numpy()
            elif hasattr(array_obj, "ndarray"):
                arrays_dict[key] = array_obj.ndarray
            else:
                arrays_dict[key] = array_obj
        extracted["arrays"] = arrays_dict

    if "metrics" in content:
        extracted["metrics"] = dict(content["metrics"])

    if "config" in content:
        extracted["config"] = dict(content["config"])

    return extracted

"""Data handling utilities for federated k-means clustering."""

from .loader import (DataStatistics, get_data_statistics, load_client_data,
                     validate_data_format)

__all__ = [
    "load_client_data",
    "validate_data_format",
    "get_data_statistics",
    "DataStatistics",
]

"""
CLEF dataset-specific preprocessing for federated learning.

This module provides dataset-specific normalization that only normalizes continuous columns,
leaving one-hot encoded categorical columns as 0/1 values. This ensures the federated
approach produces identical PCA results to the centralized approach.
"""

from logging import INFO
from typing import Any, Dict, List, Tuple

import numpy as np
from flwr.common.logger import log


def calculate_clef_local_statistics(data: np.ndarray, continuous_indices: List[int]) -> Dict[str, Any]:
    """
    Calculate local statistics for CLEF dataset, only for continuous columns.
    
    Args:
        data: Local data array of shape (n_samples, n_features)
        continuous_indices: List of indices for continuous columns to normalize
        
    Returns:
        Dictionary containing:
            - n_samples: Number of samples
            - sum: Sum of continuous features only
            - sum_sq: Sum of squared values for continuous features only
            - continuous_indices: The indices used (for validation)
    """
    n_samples = data.shape[0]
    
    # Extract only continuous columns
    continuous_data = data[:, continuous_indices]
    
    feature_sum = np.sum(continuous_data, axis=0)
    feature_sum_sq = np.sum(continuous_data**2, axis=0)
    
    return {
        "n_samples": int(n_samples),
        "sum": feature_sum.tolist(),
        "sum_sq": feature_sum_sq.tolist(),
        "continuous_indices": continuous_indices,
        "n_continuous_features": len(continuous_indices)
    }


def aggregate_clef_statistics(local_stats_list: list) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Aggregate local statistics to compute global mean and variance for continuous columns only.
    
    Args:
        local_stats_list: List of dictionaries containing local statistics
        
    Returns:
        Tuple of (global_mean, global_std, continuous_indices) as numpy arrays and list
    """
    if not local_stats_list:
        raise ValueError("No local statistics provided")
    
    # Validate that all clients have the same continuous indices
    reference_indices = local_stats_list[0]["continuous_indices"]
    for i, stats in enumerate(local_stats_list):
        if stats["continuous_indices"] != reference_indices:
            raise ValueError(f"Client {i} has different continuous indices: {stats['continuous_indices']} vs {reference_indices}")
    
    total_samples = sum(stats["n_samples"] for stats in local_stats_list)
    n_continuous = local_stats_list[0]["n_continuous_features"]
    
    # Convert lists back to numpy arrays and aggregate
    global_sum = np.zeros(n_continuous)
    global_sum_sq = np.zeros(n_continuous)
    
    for stats in local_stats_list:
        global_sum += np.array(stats["sum"])
        global_sum_sq += np.array(stats["sum_sq"])
    
    # Calculate global mean and variance for continuous columns only
    global_mean = global_sum / total_samples
    global_variance = (global_sum_sq / total_samples) - (global_mean**2)
    
    # Avoid division by zero - use 1.0 for features with zero variance
    global_std = np.sqrt(np.maximum(global_variance, 1e-10))
    global_std = np.where(global_std < 1e-8, 1.0, global_std)
    
    log(INFO, f"CLEF global statistics computed for {n_continuous} continuous features")
    log(INFO, f"Global mean shape: {global_mean.shape}, std shape: {global_std.shape}")
    log(INFO, f"Total samples across all clients: {total_samples}")
    
    return global_mean, global_std, reference_indices


def normalize_clef_data(data: np.ndarray, global_mean: np.ndarray, global_std: np.ndarray, 
                       continuous_indices: List[int]) -> np.ndarray:
    """
    Normalize CLEF data using global statistics, only for continuous columns.
    
    One-hot encoded categorical columns remain as 0/1 values.
    
    Args:
        data: Data array to normalize of shape (n_samples, n_features)
        global_mean: Global mean for continuous features
        global_std: Global standard deviation for continuous features  
        continuous_indices: Indices of continuous columns to normalize
        
    Returns:
        Normalized data array with same shape as input
    """
    # Create a copy to avoid modifying original data
    normalized_data = data.copy()
    
    # Only normalize continuous columns
    continuous_data = data[:, continuous_indices]
    normalized_continuous = (continuous_data - global_mean) / global_std
    
    # Replace continuous columns with normalized values
    normalized_data[:, continuous_indices] = normalized_continuous
    
    log(INFO, f"Normalized {len(continuous_indices)} continuous columns, "
             f"kept {data.shape[1] - len(continuous_indices)} one-hot encoded columns as 0/1")
    return normalized_data

"""
Federated preprocessing utilities for data normalization.

This module provides functions to calculate global statistics (mean and variance)
across federated clients and normalize data using those statistics.
"""

from logging import INFO
from typing import Any, Dict, Tuple

import numpy as np
from flwr.common.logger import log


def calculate_local_statistics(data: np.ndarray) -> Dict[str, Any]:
    """
    Calculate local statistics needed for global mean and variance computation.

    Args:
        data: Local data array of shape (n_samples, n_features)

    Returns:
        Dictionary containing:
            - n_samples: Number of samples
            - sum: Sum of all features
            - sum_sq: Sum of squared values for all features
    """
    n_samples = data.shape[0]
    feature_sum = np.sum(data, axis=0)
    feature_sum_sq = np.sum(data**2, axis=0)

    return {"n_samples": int(n_samples), "sum": feature_sum.tolist(), "sum_sq": feature_sum_sq.tolist()}


def aggregate_statistics(local_stats_list: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate local statistics to compute global mean and variance.

    Args:
        local_stats_list: List of dictionaries containing local statistics

    Returns:
        Tuple of (global_mean, global_std) as numpy arrays
    """
    total_samples = sum(stats["n_samples"] for stats in local_stats_list)

    # Convert lists back to numpy arrays and aggregate
    global_sum = np.zeros_like(local_stats_list[0]["sum"])
    global_sum_sq = np.zeros_like(local_stats_list[0]["sum_sq"])

    for stats in local_stats_list:
        global_sum += np.array(stats["sum"])
        global_sum_sq += np.array(stats["sum_sq"])

    # Calculate global mean and variance
    global_mean = global_sum / total_samples
    global_variance = (global_sum_sq / total_samples) - (global_mean**2)

    # Avoid division by zero - use 1.0 for features with zero variance
    global_std = np.sqrt(np.maximum(global_variance, 1e-10))
    global_std = np.where(global_std < 1e-8, 1.0, global_std)

    log(INFO, f"Global statistics computed: mean shape={global_mean.shape}, std shape={global_std.shape}")
    log(INFO, f"Total samples across all clients: {total_samples}")

    return global_mean, global_std


def calculate_local_covariance_stats(data: np.ndarray) -> Dict[str, Any]:
    """
    Calculate local covariance statistics for federated PCA.

    This computes the local mean, sample count, and unnormalized covariance matrix
    needed for federated PCA computation without sharing raw data.

    Args:
        data: Local data array of shape (n_samples, n_features)

    Returns:
        Dictionary containing:
            - n_samples: Number of samples
            - mean: Local mean vector
            - cov: Unnormalized covariance matrix (X_centered.T @ X_centered)
    """
    n_samples = data.shape[0]
    local_mean = data.mean(axis=0)

    # Center the data
    data_centered = data - local_mean

    # Compute unnormalized covariance (will be normalized at server)
    local_cov = data_centered.T @ data_centered

    log(
        INFO,
        f"Local covariance stats: n_samples={n_samples}, mean shape={local_mean.shape}, cov shape={local_cov.shape}",
    )

    return {"n_samples": int(n_samples), "mean": local_mean.tolist(), "cov": local_cov.tolist()}


def aggregate_covariance_and_compute_pca(
    local_cov_stats_list: list, n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate local covariance statistics and compute federated PCA.

    This implements the federated PCA algorithm that produces identical results
    to centralized PCA without sharing raw data between clients.

    Algorithm:
    1. Compute global mean: μ = Σ(n_c * μ_c) / N
    2. Aggregate covariance with mean correction:
       Σ = (1/N) * Σ[cov_c + n_c * (μ_c - μ)(μ_c - μ)ᵀ]
    3. Eigen-decompose Σ to get principal components

    Args:
        local_cov_stats_list: List of dictionaries from calculate_local_covariance_stats
        n_components: Number of principal components to return (default: 2 for visualization)

    Returns:
        Tuple of:
            - global_mean: Global mean vector
            - principal_components: Top n_components eigenvectors (d x n_components)
            - explained_variance: Variance explained by each component
    """
    # Step 1: Aggregate sample counts and compute global mean
    total_samples = sum(stats["n_samples"] for stats in local_cov_stats_list)

    # Convert means to numpy arrays
    local_means = [np.array(stats["mean"]) for stats in local_cov_stats_list]
    local_ns = [stats["n_samples"] for stats in local_cov_stats_list]

    global_mean = sum(n * mean for n, mean in zip(local_ns, local_means)) / total_samples

    log(INFO, f"Federated PCA: total_samples={total_samples}, global_mean shape={global_mean.shape}")

    # Step 2: Aggregate covariance matrices with mean correction
    d = len(global_mean)
    global_cov = np.zeros((d, d))

    for stats, local_mean, n_c in zip(local_cov_stats_list, local_means, local_ns):
        local_cov = np.array(stats["cov"])

        # Mean difference correction term
        mean_diff = (local_mean - global_mean).reshape(-1, 1)
        correction = n_c * (mean_diff @ mean_diff.T)

        global_cov += local_cov + correction

    # Normalize by total samples to get covariance
    global_cov /= total_samples

    log(INFO, f"Global covariance matrix computed: shape={global_cov.shape}")

    # Step 3: Eigen-decomposition for PCA
    eigvals, eigvecs = np.linalg.eigh(global_cov)

    # Sort by eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Select top n_components
    principal_components = eigvecs[:, :n_components]
    explained_variance = eigvals[:n_components]

    # Calculate explained variance ratio
    total_variance = np.sum(eigvals)
    explained_variance_ratio = (
        explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)
    )

    log(INFO, f"PCA computed: {n_components} components")
    log(INFO, f"Explained variance ratio: {explained_variance_ratio}")

    return global_mean, principal_components, explained_variance


def project_to_pca(data: np.ndarray, global_mean: np.ndarray, principal_components: np.ndarray) -> np.ndarray:
    """
    Project data onto principal components.

    Args:
        data: Data array to project (n_samples, n_features)
        global_mean: Global mean vector from federated PCA
        principal_components: Principal component vectors (n_features, n_components)

    Returns:
        Projected data (n_samples, n_components)
    """
    # Center data using global mean
    data_centered = data - global_mean

    # Project onto principal components
    projected = data_centered @ principal_components

    return projected

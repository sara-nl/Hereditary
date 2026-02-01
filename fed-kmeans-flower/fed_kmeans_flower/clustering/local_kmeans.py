"""Local k-means clustering engine for federated learning."""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

logger = logging.getLogger(__name__)


def assign_to_global_clusters(local_data: np.ndarray, global_means: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Assign local data points to global cluster means using Euclidean distance.

    Args:
        local_data: Local data points, shape (n_samples, n_features)
        global_means: Global cluster means, shape (k_global, n_features)

    Returns:
        Tuple of (cluster_assignments, distances) where:
        - cluster_assignments: Array of cluster indices for each data point, shape (n_samples,)
        - distances: Array of distances to assigned clusters, shape (n_samples,)

    Raises:
        ValueError: If input arrays have incompatible shapes or contain invalid values
    """
    if local_data.size == 0:
        return np.array([]), np.array([])

    if global_means.size == 0:
        raise ValueError("Global means cannot be empty")

    if local_data.shape[1] != global_means.shape[1]:
        raise ValueError(
            f"Feature dimensions mismatch: local_data has {local_data.shape[1]} features, "
            f"global_means has {global_means.shape[1]} features"
        )

    # Check for NaN or infinite values
    if np.any(np.isnan(local_data)) or np.any(np.isinf(local_data)):
        raise ValueError("Local data contains NaN or infinite values")

    if np.any(np.isnan(global_means)) or np.any(np.isinf(global_means)):
        raise ValueError("Global means contain NaN or infinite values")

    # Use sklearn's optimized function for assignment
    cluster_assignments, distances = pairwise_distances_argmin_min(local_data, global_means, metric="euclidean")

    logger.debug(f"Assigned {len(local_data)} points to {len(global_means)} clusters")

    return cluster_assignments, distances


def remove_empty_clusters(global_means: np.ndarray, cluster_assignments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove empty clusters from global means and update assignments.

    Args:
        global_means: Global cluster means, shape (k_global, n_features)
        cluster_assignments: Cluster assignments for local data, shape (n_samples,)

    Returns:
        Tuple of (filtered_means, updated_assignments) where:
        - filtered_means: Global means with empty clusters removed
        - updated_assignments: Updated cluster assignments with consecutive indices

    Raises:
        ValueError: If inputs are invalid
    """
    if global_means.size == 0:
        return np.array([]).reshape(0, global_means.shape[1] if len(global_means.shape) > 1 else 0), np.array([])

    if cluster_assignments.size == 0:
        # No data points, all clusters are empty
        return np.array([]).reshape(0, global_means.shape[1]), np.array([])

    # Find unique clusters that have assigned points
    unique_clusters = np.unique(cluster_assignments)

    if len(unique_clusters) == 0:
        # No valid assignments
        return np.array([]).reshape(0, global_means.shape[1]), np.array([])

    # Filter global means to keep only non-empty clusters
    filtered_means = global_means[unique_clusters]

    # Create mapping from old cluster indices to new consecutive indices
    cluster_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_clusters)}

    # Update cluster assignments to use consecutive indices
    updated_assignments = np.array([cluster_mapping[old_idx] for old_idx in cluster_assignments])

    logger.debug(f"Removed {len(global_means) - len(filtered_means)} empty clusters")

    return filtered_means, updated_assignments


def perform_local_kmeans_iteration(
    local_data: np.ndarray, initial_means: np.ndarray, max_iterations: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform local k-means clustering iterations.

    Args:
        local_data: Local data points, shape (n_samples, n_features)
        initial_means: Initial cluster means, shape (k, n_features)
        max_iterations: Maximum number of k-means iterations to perform

    Returns:
        Tuple of (final_means, final_assignments, sample_counts) where:
        - final_means: Updated cluster means after iterations, shape (k, n_features)
        - final_assignments: Final cluster assignments, shape (n_samples,)
        - sample_counts: Number of samples per cluster, shape (k,)

    Raises:
        ValueError: If inputs are invalid
    """
    if local_data.size == 0:
        return (
            np.array([]).reshape(0, initial_means.shape[1] if len(initial_means.shape) > 1 else 0),
            np.array([]),
            np.array([]),
        )

    if initial_means.size == 0:
        raise ValueError("Initial means cannot be empty")

    if local_data.shape[1] != initial_means.shape[1]:
        raise ValueError(
            f"Feature dimensions mismatch: local_data has {local_data.shape[1]} features, "
            f"initial_means has {initial_means.shape[1]} features"
        )

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    current_means = initial_means.copy()
    k = len(current_means)

    for iteration in range(max_iterations):
        # Assign points to current means
        assignments, _ = assign_to_global_clusters(local_data, current_means)

        # Update means based on assignments
        new_means = np.zeros_like(current_means)
        sample_counts = np.zeros(k)

        for cluster_idx in range(k):
            cluster_mask = assignments == cluster_idx
            cluster_points = local_data[cluster_mask]

            if len(cluster_points) > 0:
                new_means[cluster_idx] = np.mean(cluster_points, axis=0)
                sample_counts[cluster_idx] = len(cluster_points)
            else:
                # Keep the original mean for empty clusters
                new_means[cluster_idx] = current_means[cluster_idx]
                sample_counts[cluster_idx] = 0

        # Check for convergence (means didn't change significantly)
        if np.allclose(current_means, new_means, rtol=1e-6):
            logger.debug(f"Local k-means converged after {iteration + 1} iterations")
            break

        current_means = new_means

    # Final assignment with updated means
    final_assignments, _ = assign_to_global_clusters(local_data, current_means)

    # Recompute sample counts for final assignments
    final_sample_counts = np.zeros(k)
    for cluster_idx in range(k):
        final_sample_counts[cluster_idx] = np.sum(final_assignments == cluster_idx)

    logger.debug(f"Local k-means completed {max_iterations} iterations")

    return current_means, final_assignments, final_sample_counts


def apply_privacy_filter(
    cluster_means: np.ndarray, sample_counts: np.ndarray, privacy_threshold: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply privacy filtering to remove clusters with insufficient samples.

    Args:
        cluster_means: Cluster means, shape (k, n_features)
        sample_counts: Number of samples per cluster, shape (k,)
        privacy_threshold: Minimum number of samples required per cluster

    Returns:
        Tuple of (filtered_means, filtered_counts) where:
        - filtered_means: Cluster means with small clusters removed
        - filtered_counts: Sample counts for remaining clusters

    Raises:
        ValueError: If inputs are invalid
    """
    if cluster_means.size == 0:
        return np.array([]).reshape(0, cluster_means.shape[1] if len(cluster_means.shape) > 1 else 0), np.array([])

    if len(cluster_means) != len(sample_counts):
        raise ValueError(
            f"Length mismatch: cluster_means has {len(cluster_means)} clusters, "
            f"sample_counts has {len(sample_counts)} counts"
        )
        
    # Find clusters that meet the privacy threshold
    valid_clusters = sample_counts >= privacy_threshold

    if not np.any(valid_clusters):
        # No clusters meet the threshold
        logger.warning(f"No clusters meet privacy threshold of {privacy_threshold}")
        return np.array([]).reshape(0, cluster_means.shape[1]), np.array([])

    # Filter means and counts
    filtered_means = cluster_means[valid_clusters]
    filtered_counts = sample_counts[valid_clusters]

    removed_count = len(cluster_means) - len(filtered_means)
    if removed_count > 0:
        logger.debug(f"Privacy filter removed {removed_count} clusters below threshold {privacy_threshold}")

    return filtered_means, filtered_counts


def compute_local_clustering_quality(
    local_data: np.ndarray, cluster_assignments: np.ndarray, cluster_means: np.ndarray
) -> Tuple[float, int]:
    """Compute local clustering quality metrics.

    Args:
        local_data: Local data points, shape (n_samples, n_features)
        cluster_assignments: Cluster assignments, shape (n_samples,)
        cluster_means: Cluster means, shape (k, n_features)

    Returns:
        Tuple of (inertia, num_clusters) where:
        - inertia: Within-cluster sum of squared distances
        - num_clusters: Number of active clusters

    Raises:
        ValueError: If inputs are invalid
    """
    if local_data.size == 0 or cluster_means.size == 0:
        return 0.0, 0

    if len(cluster_assignments) != len(local_data):
        raise ValueError(
            f"Length mismatch: local_data has {len(local_data)} samples, "
            f"cluster_assignments has {len(cluster_assignments)} assignments"
        )

    # Compute inertia (within-cluster sum of squared distances)
    inertia = 0.0
    unique_clusters = np.unique(cluster_assignments)

    for cluster_idx in unique_clusters:
        if cluster_idx >= len(cluster_means):
            continue  # Skip invalid cluster indices

        cluster_mask = cluster_assignments == cluster_idx
        cluster_points = local_data[cluster_mask]

        if len(cluster_points) > 0:
            cluster_center = cluster_means[cluster_idx]
            # Compute squared distances to cluster center
            squared_distances = np.sum((cluster_points - cluster_center) ** 2, axis=1)
            inertia += np.sum(squared_distances)

    num_clusters = len(unique_clusters)

    logger.debug(f"Local clustering quality: inertia={inertia:.4f}, clusters={num_clusters}")

    return inertia, num_clusters


def perform_complete_local_clustering(
    local_data: np.ndarray, global_means: np.ndarray, privacy_threshold: int, local_iterations: int = 1
) -> Tuple[np.ndarray, np.ndarray, bool, Optional[str]]:
    """Perform complete local clustering workflow as specified in the federated k-means algorithm.

    This function implements the complete local clustering process:
    1. Assign local data to global cluster means
    2. Remove empty clusters
    3. Perform local k-means iterations
    4. Apply privacy filtering

    Args:
        local_data: Local data points, shape (n_samples, n_features)
        global_means: Global cluster means from server, shape (k_global, n_features)
        privacy_threshold: Minimum samples per cluster for privacy
        local_iterations: Number of local k-means iterations to perform

    Returns:
        Tuple of (local_means, sample_counts, success, error_message) where:
        - local_means: Local cluster means after processing, shape (k_local, n_features)
        - sample_counts: Number of samples per local cluster, shape (k_local,)
        - success: True if clustering completed successfully
        - error_message: Error description if success is False, None otherwise
    """
    try:
        # Validate inputs
        if local_data.size == 0:
            return (
                np.array([]).reshape(0, global_means.shape[1] if len(global_means.shape) > 1 else 0),
                np.array([]),
                True,
                None,
            )
        # Step 1: Assign local data points to global cluster means
        cluster_assignments, _ = assign_to_global_clusters(local_data, global_means)

        # Step 2: Remove empty clusters from global means
        filtered_means, updated_assignments = remove_empty_clusters(global_means, cluster_assignments)

        if filtered_means.size == 0:
            return np.array([]).reshape(0, local_data.shape[1]), np.array([]), True, None

        # Step 3: Perform local k-means iterations
        local_means, final_assignments, sample_counts = perform_local_kmeans_iteration(
            local_data, filtered_means, local_iterations
        )

        # Step 4: Apply privacy filtering
        privacy_filtered_means, privacy_filtered_counts = apply_privacy_filter(
            local_means, sample_counts, privacy_threshold
        )

        logger.info(
            f"Local clustering completed: {len(privacy_filtered_means)} clusters, "
            f"{np.sum(privacy_filtered_counts)} total samples"
        )

        return privacy_filtered_means, privacy_filtered_counts, True, None

    except Exception as e:
        error_msg = f"Error in local clustering: {str(e)}"
        logger.error(error_msg)
        return (
            np.array([]).reshape(0, local_data.shape[1] if local_data.size > 0 else 0),
            np.array([]),
            False,
            error_msg,
        )

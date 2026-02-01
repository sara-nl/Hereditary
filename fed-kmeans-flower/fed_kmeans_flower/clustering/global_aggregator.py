"""Global aggregation engine for federated k-means clustering."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def validate_local_cluster_means(
    local_means_list: List[np.ndarray], sample_counts_list: List[np.ndarray]
) -> Tuple[bool, Optional[str]]:
    """Validate received local cluster means and sample counts.

    Args:
        local_means_list: List of local cluster means from clients
        sample_counts_list: List of sample counts corresponding to local means

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not local_means_list:
        return False, "No local cluster means received"

    if len(local_means_list) != len(sample_counts_list):
        return False, "Mismatch between number of local means and sample counts"

    # Check each client's data
    for i, (means, counts) in enumerate(zip(local_means_list, sample_counts_list)):
        if means.size == 0:
            continue  # Empty means are allowed (client had no valid clusters)

        # Validate shapes
        if len(means.shape) != 2:
            return False, f"Client {i}: cluster means must be 2D array, got shape {means.shape}"

        if len(means) != len(counts):
            return False, f"Client {i}: number of means ({len(means)}) doesn't match number of counts ({len(counts)})"

        # Check for invalid values
        if np.any(np.isnan(means)) or np.any(np.isinf(means)):
            return False, f"Client {i}: cluster means contain NaN or infinite values"

    # Check feature dimension consistency
    non_empty_means = [means for means in local_means_list if means.size > 0]
    if non_empty_means:
        n_features = non_empty_means[0].shape[1]
        for i, means in enumerate(non_empty_means[1:], 1):
            if means.shape[1] != n_features:
                return False, f"Feature dimension mismatch: expected {n_features}, got {means.shape[1]} from client {i}"

    return True, None


def concatenate_local_means(
    local_means_list: List[np.ndarray], sample_counts_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate local cluster means from all clients.

    Args:
        local_means_list: List of local cluster means from clients
        sample_counts_list: List of sample counts corresponding to local means

    Returns:
        Tuple of (concatenated_means, concatenated_counts) where:
        - concatenated_means: All local means concatenated, shape (total_clusters, n_features)
        - concatenated_counts: All sample counts concatenated, shape (total_clusters,)

    Raises:
        ValueError: If input validation fails
    """
    # Validate inputs
    is_valid, error_msg = validate_local_cluster_means(local_means_list, sample_counts_list)
    if not is_valid:
        raise ValueError(f"Invalid local cluster means: {error_msg}")

    # Filter out empty means
    valid_means = []
    valid_counts = []

    for means, counts in zip(local_means_list, sample_counts_list):
        if means.size > 0:
            valid_means.append(means)
            valid_counts.append(counts)

    if not valid_means:
        # No valid cluster means from any client
        logger.warning("No valid cluster means received from any client")
        return np.array([]).reshape(0, 0), np.array([])

    # Concatenate all valid means and counts
    concatenated_means = np.vstack(valid_means)
    concatenated_counts = np.concatenate(valid_counts)

    logger.debug(f"Concatenated {len(concatenated_means)} cluster means from {len(valid_means)} clients")

    return concatenated_means, concatenated_counts


def perform_weighted_kmeans_aggregation(
    concatenated_means: np.ndarray,
    sample_counts: np.ndarray,
    k_global: int,
    weighted: bool = True,
    max_iterations: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, bool, Optional[str]]:
    """Perform k-means clustering on concatenated local cluster means.

    This implements the global aggregation step of the federated k-means algorithm.
    It can perform either weighted or unweighted k-means clustering on the local
    cluster centroids.

    Args:
        concatenated_means: Concatenated local cluster means, shape (n_local_clusters, n_features)
        sample_counts: Sample counts for each local cluster, shape (n_local_clusters,)
        k_global: Target number of global clusters
        weighted: Whether to weight local centroids by their sample counts
        max_iterations: Maximum iterations for k-means
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (global_means, success, error_message) where:
        - global_means: Aggregated global cluster means, shape (k_global, n_features)
        - success: True if aggregation completed successfully
        - error_message: Error description if success is False, None otherwise
    """
    if concatenated_means.size == 0:
        error_msg = "No cluster means to aggregate"
        logger.warning(error_msg)
        return np.array([]).reshape(0, 0), False, error_msg

    if len(concatenated_means) != len(sample_counts):
        error_msg = f"Length mismatch: {len(concatenated_means)} means vs {len(sample_counts)} counts"
        logger.error(error_msg)
        return np.array([]).reshape(0, concatenated_means.shape[1]), False, error_msg

    if k_global <= 0:
        error_msg = f"Invalid k_global: {k_global}"
        logger.error(error_msg)
        return np.array([]).reshape(0, concatenated_means.shape[1]), False, error_msg

    try:
        n_local_clusters, n_features = concatenated_means.shape

        # If we have fewer local clusters than k_global, return all local means
        if n_local_clusters <= k_global:
            logger.info(
                f"Number of local clusters ({n_local_clusters}) <= k_global ({k_global}), returning all local means"
            )
            return concatenated_means.copy(), True, None

        # Perform k-means clustering on local cluster means
        # If weighted=True, we use sample_weight to respect data distribution
        kmeans = KMeans(
            n_clusters=min(k_global, n_local_clusters),
            max_iter=max_iterations,
            random_state=random_state,
            n_init=10,
        )

        if weighted:
            logger.debug(f"Performing weighted aggregation with {n_local_clusters} local clusters")
            kmeans.fit(concatenated_means, sample_weight=sample_counts)
        else:
            logger.debug(f"Performing unweighted aggregation with {n_local_clusters} local clusters")
            kmeans.fit(concatenated_means)

        global_means = kmeans.cluster_centers_

        # If we got fewer clusters than requested, pad with random initialization
        if len(global_means) < k_global:
            logger.warning(f"K-means returned {len(global_means)} clusters, padding to {k_global}")

            # Calculate data range for random initialization
            data_min = np.min(concatenated_means, axis=0)
            data_max = np.max(concatenated_means, axis=0)

            # Generate additional random cluster centers
            additional_clusters = k_global - len(global_means)
            if random_state is not None:
                np.random.seed(random_state)

            random_centers = np.random.uniform(data_min, data_max, size=(additional_clusters, n_features))

            global_means = np.vstack([global_means, random_centers])

        logger.info(
            f"Global aggregation completed: {len(global_means)} clusters from {n_local_clusters} local clusters"
        )

        return global_means, True, None

    except Exception as e:
        error_msg = f"Error in weighted k-means aggregation: {str(e)}"
        logger.error(error_msg)
        return (
            np.array([]).reshape(0, concatenated_means.shape[1] if concatenated_means.size > 0 else 0),
            False,
            error_msg,
        )


def match_clusters_between_rounds(previous_means: np.ndarray, current_means: np.ndarray) -> np.ndarray:
    """Match clusters between rounds to handle permutation invariance.

    Uses the Hungarian algorithm to find the optimal assignment that minimizes
    the total distance between matched clusters.

    Args:
        previous_means: Previous global cluster means, shape (k, n_features)
        current_means: Current global cluster means, shape (k, n_features)

    Returns:
        Reordered current_means that best matches previous_means order
    """
    from scipy.optimize import linear_sum_assignment

    k = len(previous_means)

    # Compute pairwise distances between all clusters
    # distance_matrix[i, j] = distance from previous cluster i to current cluster j
    distance_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            distance_matrix[i, j] = np.linalg.norm(previous_means[i] - current_means[j])

    # Find optimal assignment using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Reorder current_means according to the optimal assignment
    reordered_means = np.zeros_like(current_means)
    for i, j in zip(row_indices, col_indices):
        reordered_means[i] = current_means[j]

    logger.debug(
        f"Cluster matching: assignment={col_indices}, total_distance={distance_matrix[row_indices, col_indices].sum():.6f}"
    )

    return reordered_means


def compute_convergence_change(
    previous_means: Optional[np.ndarray], current_means: np.ndarray, convergence_tolerance: float
) -> Tuple[float, bool]:
    """Compute convergence change between consecutive global cluster means.

    This function handles the permutation invariance problem by matching clusters
    between rounds before computing the change magnitude.

    Args:
        previous_means: Previous global cluster means, shape (k, n_features)
        current_means: Current global cluster means, shape (k, n_features)
        convergence_tolerance: Tolerance threshold for convergence

    Returns:
        Tuple of (change_magnitude, is_converged) where:
        - change_magnitude: Magnitude of change between means
        - is_converged: True if change is below tolerance
    """
    if previous_means is None:
        # First iteration, no convergence yet
        return float("inf"), False

    if previous_means.shape != current_means.shape:
        logger.warning(f"Shape mismatch in convergence check: {previous_means.shape} vs {current_means.shape}")
        return float("inf"), False

    if current_means.size == 0:
        return 0.0, True  # Empty means are considered converged

    try:
        # Match clusters between rounds to handle permutation invariance
        matched_current_means = match_clusters_between_rounds(previous_means, current_means)

        # Compute Frobenius norm of the difference after matching
        change_magnitude = np.linalg.norm(matched_current_means - previous_means, "fro")

        # Normalize by the number of clusters and features for scale invariance
        normalized_change = change_magnitude / np.sqrt(current_means.size)

        is_converged = normalized_change < convergence_tolerance

        logger.debug(
            f"Convergence check: change={normalized_change:.6f}, tolerance={convergence_tolerance}, converged={is_converged}"
        )

        return normalized_change, is_converged

    except Exception as e:
        logger.error(f"Error in convergence computation: {str(e)}")
        return float("inf"), False


def aggregate_local_cluster_means(
    local_means_list: List[np.ndarray],
    sample_counts_list: List[np.ndarray],
    k_global: int,
    weighted: bool = True,
    previous_global_means: Optional[np.ndarray] = None,
    convergence_tolerance: float = 0.01,
    max_iterations: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, bool, float, bool, Optional[str]]:
    """Complete global aggregation workflow for federated k-means.

    This function implements the complete server-side aggregation process:
    1. Validate and concatenate local cluster means
    2. Perform weighted k-means aggregation
    3. Match clusters to previous round (to handle permutation invariance)
    4. Check for convergence

    Args:
        local_means_list: List of local cluster means from clients
        sample_counts_list: List of sample counts corresponding to local means
        k_global: Target number of global clusters
        weighted: Whether to use weighted aggregation
        previous_global_means: Previous global means for convergence checking
        convergence_tolerance: Tolerance threshold for convergence
        max_iterations: Maximum iterations for k-means aggregation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (global_means, success, change_magnitude, is_converged, error_message) where:
        - global_means: New global cluster means (matched to previous order), shape (k_global, n_features)
        - success: True if aggregation completed successfully
        - change_magnitude: Magnitude of change from previous means
        - is_converged: True if algorithm has converged
        - error_message: Error description if success is False, None otherwise
    """
    try:
        # Step 1: Concatenate and validate local cluster means
        concatenated_means, concatenated_counts = concatenate_local_means(local_means_list, sample_counts_list)

        if concatenated_means.size == 0:
            error_msg = "No valid local cluster means to aggregate"
            logger.error(error_msg)
            return np.array([]).reshape(0, 0), False, float("inf"), False, error_msg

        # Step 2: Perform k-means aggregation
        global_means, aggregation_success, aggregation_error = perform_weighted_kmeans_aggregation(
            concatenated_means, concatenated_counts, k_global, weighted, max_iterations, random_state
        )

        if not aggregation_success:
            return global_means, False, float("inf"), False, aggregation_error

        # Step 3: Match clusters to previous round if available
        if previous_global_means is not None and previous_global_means.shape == global_means.shape:
            global_means = match_clusters_between_rounds(previous_global_means, global_means)

        # Step 4: Check convergence (now using matched clusters)
        change_magnitude, is_converged = compute_convergence_change(
            previous_global_means, global_means, convergence_tolerance
        )

        logger.info(f"Global aggregation completed: change={change_magnitude:.6f}, converged={is_converged}")

        return global_means, True, change_magnitude, is_converged, None

    except Exception as e:
        error_msg = f"Error in global aggregation workflow: {str(e)}"
        logger.error(error_msg)
        return np.array([]).reshape(0, 0), False, float("inf"), False, error_msg

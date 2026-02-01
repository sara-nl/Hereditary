"""Local clustering quality metrics and evaluation utilities."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)


def compute_adjusted_rand_score(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Compute adjusted rand score for clustering evaluation.

    Args:
        true_labels: Ground truth cluster labels, shape (n_samples,)
        predicted_labels: Predicted cluster labels, shape (n_samples,)

    Returns:
        Adjusted rand score between -1 and 1 (higher is better)

    Raises:
        ValueError: If inputs are invalid
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError(f"Label arrays must have same length: {len(true_labels)} vs {len(predicted_labels)}")

    if len(true_labels) == 0:
        return 1.0  # Perfect score for empty arrays

    # Handle case where all labels are the same
    if len(np.unique(true_labels)) == 1 and len(np.unique(predicted_labels)) == 1:
        return 1.0

    try:
        ari = adjusted_rand_score(true_labels, predicted_labels)
        logger.debug(f"Computed adjusted rand score: {ari:.4f}")
        return float(ari)
    except Exception as e:
        logger.error(f"Error computing adjusted rand score: {e}")
        return 0.0


def compute_silhouette_score(data: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Compute silhouette score for clustering evaluation.

    Args:
        data: Data points, shape (n_samples, n_features)
        cluster_labels: Cluster labels, shape (n_samples,)

    Returns:
        Silhouette score between -1 and 1 (higher is better)

    Raises:
        ValueError: If inputs are invalid
    """
    if len(data) != len(cluster_labels):
        raise ValueError(f"Data and labels must have same length: {len(data)} vs {len(cluster_labels)}")

    if len(data) < 2:
        return 0.0  # Cannot compute silhouette score with less than 2 samples

    # Check if we have at least 2 different clusters
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) < 2:
        return 0.0  # Cannot compute silhouette score with only one cluster

    try:
        silhouette = silhouette_score(data, cluster_labels, metric="euclidean")
        logger.debug(f"Computed silhouette score: {silhouette:.4f}")
        return float(silhouette)
    except Exception as e:
        logger.error(f"Error computing silhouette score: {e}")
        return 0.0




def compute_cluster_statistics(data: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """Compute comprehensive statistics for local clustering.

    Args:
        data: Data points, shape (n_samples, n_features)
        cluster_labels: Cluster labels, shape (n_samples,)

    Returns:
        Dictionary containing clustering statistics:
        - num_clusters: Number of unique clusters
        - cluster_sizes: Array of cluster sizes
        - cluster_densities: Array of cluster densities (avg distance to center)
        - total_samples: Total number of data points
        - largest_cluster_size: Size of the largest cluster
        - smallest_cluster_size: Size of the smallest cluster

    Raises:
        ValueError: If inputs are invalid
    """
    if len(data) != len(cluster_labels):
        raise ValueError(f"Data and labels must have same length: {len(data)} vs {len(cluster_labels)}")

    if len(data) == 0:
        return {
            "num_clusters": 0,
            "cluster_sizes": np.array([]),
            "cluster_densities": np.array([]),
            "total_samples": 0,
            "largest_cluster_size": 0,
            "smallest_cluster_size": 0,
        }

    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels)
    cluster_sizes = np.zeros(num_clusters)
    cluster_densities = np.zeros(num_clusters)

    for i, label in enumerate(unique_labels):
        cluster_mask = cluster_labels == label
        cluster_points = data[cluster_mask]
        cluster_size = len(cluster_points)
        cluster_sizes[i] = cluster_size

        if cluster_size > 1:
            # Compute average pairwise distance within cluster
            distances = euclidean_distances(cluster_points)
            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(distances, k=1)
            non_zero_distances = upper_triangle[upper_triangle > 0]
            cluster_densities[i] = np.mean(non_zero_distances) if len(non_zero_distances) > 0 else 0.0
        else:
            cluster_densities[i] = 0.0

    statistics = {
        "num_clusters": num_clusters,
        "cluster_sizes": cluster_sizes,
        "cluster_densities": cluster_densities,
        "total_samples": len(data),
        "largest_cluster_size": int(np.max(cluster_sizes)) if len(cluster_sizes) > 0 else 0,
        "smallest_cluster_size": int(np.min(cluster_sizes)) if len(cluster_sizes) > 0 else 0,
    }

    logger.debug(
        f"Computed cluster statistics: {num_clusters} clusters, "
        f"sizes range [{statistics['smallest_cluster_size']}, {statistics['largest_cluster_size']}]"
    )

    return statistics


def validate_privacy_requirements(cluster_sizes: np.ndarray, privacy_threshold: int) -> Tuple[bool, List[int]]:
    """Validate that all clusters meet privacy requirements.

    Args:
        cluster_sizes: Array of cluster sizes, shape (n_clusters,)
        privacy_threshold: Minimum required samples per cluster

    Returns:
        Tuple of (all_valid, violating_clusters) where:
        - all_valid: True if all clusters meet privacy threshold
        - violating_clusters: List of cluster indices that violate privacy
    """
    if len(cluster_sizes) == 0:
        return True, []

    violating_clusters = []
    for i, size in enumerate(cluster_sizes):
        if size < privacy_threshold:
            violating_clusters.append(i)

    all_valid = len(violating_clusters) == 0

    if not all_valid:
        logger.warning(f"Privacy violation: {len(violating_clusters)} clusters below threshold {privacy_threshold}")

    return all_valid, violating_clusters


def compute_comprehensive_local_metrics(
    data: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    distances: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    privacy_threshold: int = 2,
) -> dict:
    """Compute comprehensive local clustering quality metrics.

    Args:
        data: Data points, shape (n_samples, n_features)
        cluster_labels: Predicted cluster labels, shape (n_samples,)
        cluster_centers: Cluster centers, shape (n_clusters, n_features)
        distances: Precomputed distances to assigned clusters, shape (n_samples,)
        true_labels: Optional ground truth labels for supervised metrics
        privacy_threshold: Privacy threshold for validation

    Returns:
        Dictionary containing all computed metrics:
        - inertia: Within-cluster sum of squared distances
        - silhouette_score: Silhouette coefficient
        - adjusted_rand_score: ARI (only if true_labels provided)
        - cluster_statistics: Detailed cluster statistics
        - privacy_valid: Whether privacy requirements are met
        - privacy_violations: List of clusters violating privacy

    Raises:
        ValueError: If inputs are invalid
    """
    if len(data) != len(cluster_labels):
        raise ValueError(f"Data and labels must have same length: {len(data)} vs {len(cluster_labels)}")

    if true_labels is not None and len(true_labels) != len(data):
        raise ValueError(f"True labels length {len(true_labels)} doesn't match data length {len(data)}")

    metrics = {}

    try:
        # Compute inertia (use precomputed distances if available)
        metrics["inertia"] = float(np.sum(distances**2))

        # Compute silhouette score
        metrics["silhouette_score"] = compute_silhouette_score(data, cluster_labels)

        # Compute adjusted rand score if ground truth is available
        if true_labels is not None:
            metrics["adjusted_rand_score"] = compute_adjusted_rand_score(true_labels, cluster_labels)
        else:
            metrics["adjusted_rand_score"] = None

        # Compute cluster statistics
        cluster_stats = compute_cluster_statistics(data, cluster_labels)
        metrics["cluster_statistics"] = cluster_stats

        # Validate privacy requirements
        privacy_valid, privacy_violations = validate_privacy_requirements(
            cluster_stats["cluster_sizes"], privacy_threshold
        )
        metrics["privacy_valid"] = privacy_valid
        metrics["privacy_violations"] = privacy_violations

        logger.info(
            f"Computed comprehensive metrics: inertia={metrics['inertia']:.4f}, "
            f"silhouette={metrics['silhouette_score']:.4f}, "
            f"clusters={cluster_stats['num_clusters']}, "
            f"privacy_valid={privacy_valid}"
        )

    except Exception as e:
        logger.error(f"Error computing comprehensive metrics: {e}")
        # Return default values on error
        metrics = {
            "inertia": 0.0,
            "silhouette_score": 0.0,
            "adjusted_rand_score": None,
            "cluster_statistics": {
                "num_clusters": 0,
                "cluster_sizes": np.array([]),
                "cluster_densities": np.array([]),
                "total_samples": len(data),
                "largest_cluster_size": 0,
                "smallest_cluster_size": 0,
            },
            "privacy_valid": False,
            "privacy_violations": [],
        }

    return metrics

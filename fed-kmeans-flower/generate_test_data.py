#!/usr/bin/env python3
"""Generate synthetic federated data and save partitions to disk.

This script generates synthetic clustered data and partitions it across
multiple clients for federated k-means experiments. Each client's data
is saved separately for reproducible experiments and external analysis.

Usage:
    python generate_data.py --num-clients 5 --samples-per-client 100 --output-dir ./data
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.datasets import make_blobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""

    n_samples: int
    n_features: int
    n_clusters: int
    cluster_std: float
    center_box: Tuple[float, float]
    random_state: int
    overlap_factor: float = 0.0
    density_variation: float = 0.0


def create_federated_synthetic_dataset(
    config: SyntheticDataConfig, num_clients: int, non_iid_factor: float = 0.5
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Create a federated synthetic dataset with configurable non-IID distribution.

    Args:
        config: Configuration for synthetic data generation
        num_clients: Number of clients to partition data across
        non_iid_factor: Degree of non-IID distribution (0.0 = IID, 1.0 = max non-IID)

    Returns:
        Tuple of (client_partitions, cluster_centers) where:
        - client_partitions: List of (data, labels) tuples for each client
        - cluster_centers: True cluster centers used for generation
    """
    # Adjust cluster std based on overlap factor
    effective_std = config.cluster_std * (1.0 + config.overlap_factor)

    # Generate base synthetic data
    X, y = make_blobs(
        n_samples=config.n_samples,
        n_features=config.n_features,
        centers=config.n_clusters,
        cluster_std=effective_std,
        center_box=config.center_box,
        random_state=config.random_state,
    )

    # Get cluster centers
    cluster_centers = np.array([X[y == i].mean(axis=0) for i in range(config.n_clusters)])

    # Apply density variation if specified
    if config.density_variation > 0:
        X, y = _apply_density_variation(X, y, config.density_variation, config.random_state)

    # Partition data across clients
    if non_iid_factor < 0.01:
        # IID partitioning
        client_partitions = _partition_iid(X, y, num_clients, config.random_state)
    else:
        # Non-IID partitioning
        client_partitions = _partition_non_iid(X, y, num_clients, non_iid_factor, config.random_state)

    return client_partitions, cluster_centers


def _apply_density_variation(
    X: np.ndarray, y: np.ndarray, density_variation: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply density variation to clusters by subsampling."""
    rng = np.random.RandomState(random_state + 1)
    n_clusters = len(np.unique(y))

    # Generate varying densities
    densities = rng.uniform(1.0 - density_variation, 1.0, size=n_clusters)

    # Subsample each cluster
    indices_to_keep = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(y == cluster_id)[0]
        n_keep = int(len(cluster_indices) * densities[cluster_id])
        keep_indices = rng.choice(cluster_indices, size=n_keep, replace=False)
        indices_to_keep.extend(keep_indices)

    indices_to_keep = np.array(indices_to_keep)
    return X[indices_to_keep], y[indices_to_keep]


def _partition_iid(
    X: np.ndarray, y: np.ndarray, num_clients: int, random_state: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data in an IID manner across clients."""
    rng = np.random.RandomState(random_state)
    n_samples = len(X)

    # Shuffle indices
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    # Split into equal partitions
    client_partitions = []
    partition_size = n_samples // num_clients

    for i in range(num_clients):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size if i < num_clients - 1 else n_samples
        client_indices = indices[start_idx:end_idx]

        client_partitions.append((X[client_indices], y[client_indices]))

    return client_partitions


def _partition_non_iid(
    X: np.ndarray, y: np.ndarray, num_clients: int, non_iid_factor: float, random_state: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data in a non-IID manner using Dirichlet distribution."""
    rng = np.random.RandomState(random_state)
    n_clusters = len(np.unique(y))

    # Use Dirichlet distribution to create non-IID partitions
    # Lower alpha = more non-IID
    alpha = (1.0 - non_iid_factor) * 10.0 + 0.1

    client_partitions = [[] for _ in range(num_clients)]

    # For each cluster, distribute samples across clients
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(y == cluster_id)[0]
        rng.shuffle(cluster_indices)

        # Sample from Dirichlet to get proportions
        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(cluster_indices)).astype(int)

        # Adjust to ensure all samples are assigned
        proportions[-1] = len(cluster_indices) - proportions[:-1].sum()

        # Distribute samples
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + proportions[client_id]
            client_partitions[client_id].extend(cluster_indices[start_idx:end_idx])
            start_idx = end_idx

    # Convert to arrays and shuffle each client's data
    result = []
    for client_indices in client_partitions:
        client_indices = np.array(client_indices)
        rng.shuffle(client_indices)
        result.append((X[client_indices], y[client_indices]))

    return result


def save_client_partition(output_dir: Path, client_index: int, data: np.ndarray, labels: np.ndarray) -> None:
    """Save a client's data partition to disk.

    Args:
        output_dir: Directory to save data files
        client_index: Index of the client
        data: Client's data array
        labels: Client's ground truth labels
    """
    client_file = output_dir / f"client_{client_index}.npz"
    np.savez_compressed(client_file, data=data, labels=labels)
    logger.info(f"Saved client {client_index}: {len(data)} samples to {client_file}")


def save_metadata(
    output_dir: Path,
    config: SyntheticDataConfig,
    num_clients: int,
    non_iid_factor: float,
    cluster_centers: np.ndarray,
    partition_stats: Dict[int, Dict[str, Any]],
) -> None:
    """Save dataset metadata for reproducibility and analysis.

    Args:
        output_dir: Directory to save metadata
        config: Synthetic data configuration used
        num_clients: Number of clients
        non_iid_factor: Non-IID factor used
        cluster_centers: True cluster centers
        partition_stats: Statistics for each partition
    """
    metadata = {
        "generation_config": {
            "n_samples": config.n_samples,
            "n_features": config.n_features,
            "n_clusters": config.n_clusters,
            "cluster_std": config.cluster_std,
            "center_box": config.center_box,
            "random_state": config.random_state,
            "overlap_factor": config.overlap_factor,
            "density_variation": config.density_variation,
        },
        "federation_config": {"num_clients": num_clients, "non_iid_factor": non_iid_factor},
        "cluster_centers": cluster_centers.tolist(),
        "partition_statistics": partition_stats,
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")

    # Also save cluster centers separately for easy loading
    centers_file = output_dir / "cluster_centers.npy"
    np.save(centers_file, cluster_centers)
    logger.info(f"Saved cluster centers to {centers_file}")


def compute_partition_statistics(client_partitions: list, num_clients: int) -> Dict[int, Dict[str, Any]]:
    """Compute statistics for each client partition.

    Args:
        client_partitions: List of (data, labels) tuples
        num_clients: Number of clients

    Returns:
        Dictionary mapping client index to statistics
    """
    stats = {}

    for i in range(num_clients):
        data, labels = client_partitions[i]

        unique_labels, label_counts = np.unique(labels, return_counts=True)

        stats[i] = {
            "num_samples": len(data),
            "num_features": data.shape[1] if len(data) > 0 else 0,
            "unique_labels": unique_labels.tolist(),
            "label_distribution": {int(label): int(count) for label, count in zip(unique_labels, label_counts)},
            "feature_means": data.mean(axis=0).tolist() if len(data) > 0 else [],
            "feature_stds": data.std(axis=0).tolist() if len(data) > 0 else [],
        }

    return stats


def generate_and_save_data(
    num_clients: int,
    samples_per_client: int,
    num_features: int,
    num_clusters: int,
    cluster_std: float,
    non_iid_factor: float,
    overlap_factor: float,
    density_variation: float,
    random_state: int,
    output_dir: Path,
) -> None:
    """Generate synthetic federated data and save to disk.

    Args:
        num_clients: Number of federated clients
        samples_per_client: Approximate samples per client
        num_features: Number of features (dimensions)
        num_clusters: Number of clusters
        cluster_std: Standard deviation of clusters
        non_iid_factor: Degree of non-IID distribution (0.0 = IID, 1.0 = max non-IID)
        overlap_factor: Degree of cluster overlap
        density_variation: Variation in cluster densities
        random_state: Random seed for reproducibility
        output_dir: Directory to save generated data
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating data for {num_clients} clients in {output_dir}")

    # Configure synthetic data generation
    total_samples = num_clients * samples_per_client
    config = SyntheticDataConfig(
        n_samples=total_samples,
        n_features=num_features,
        n_clusters=num_clusters,
        cluster_std=cluster_std,
        center_box=(-10.0, 10.0),
        random_state=random_state,
        overlap_factor=overlap_factor,
        density_variation=density_variation,
    )

    logger.info(f"Configuration: {total_samples} total samples, " f"{num_features} features, {num_clusters} clusters")
    logger.info(
        f"Non-IID factor: {non_iid_factor}, "
        f"Overlap factor: {overlap_factor}, "
        f"Density variation: {density_variation}"
    )

    # Generate federated dataset
    client_partitions, cluster_centers = create_federated_synthetic_dataset(
        config=config, num_clients=num_clients, non_iid_factor=non_iid_factor
    )

    # Save each client's partition
    for i in range(num_clients):
        data, labels = client_partitions[i]
        save_client_partition(output_dir, i, data, labels)

    # Compute and save statistics
    partition_stats = compute_partition_statistics(client_partitions, num_clients)
    save_metadata(output_dir, config, num_clients, non_iid_factor, cluster_centers, partition_stats)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Data Generation Summary")
    logger.info("=" * 60)
    logger.info(f"Total clients: {num_clients}")
    logger.info(f"Total samples: {sum(stats['num_samples'] for stats in partition_stats.values())}")
    logger.info(f"Features: {num_features}")
    logger.info(f"Clusters: {num_clusters}")
    logger.info(f"Random seed: {random_state}")
    logger.info("\nPer-client statistics:")
    for i, stats in partition_stats.items():
        logger.info(f"  Client {i}: {stats['num_samples']} samples, " f"labels: {stats['unique_labels']}")
    logger.info("=" * 60)


def main():
    """Main entry point for data generation script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic federated data for k-means clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data generation parameters
    parser.add_argument("--num-clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--samples-per-client", type=int, default=100, help="Approximate number of samples per client")
    parser.add_argument("--num-features", type=int, default=2, help="Number of features (dimensions)")
    parser.add_argument("--num-clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--cluster-std", type=float, default=1.0, help="Standard deviation of clusters")
    parser.add_argument(
        "--non-iid-factor",
        type=float,
        default=0.5,
        help="Degree of non-IID distribution (0.0 = IID, 1.0 = maximum non-IID)",
    )
    parser.add_argument(
        "--overlap-factor",
        type=float,
        default=0.0,
        help="Degree of cluster overlap (0.0 = no overlap, 1.0 = high overlap)",
    )
    parser.add_argument(
        "--density-variation",
        type=float,
        default=0.0,
        help="Variation in cluster densities (0.0 = uniform, 1.0 = max variation)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="./data", help="Directory to save generated data files")

    args = parser.parse_args()

    # Validate arguments
    if args.num_clients <= 0:
        parser.error("num-clients must be positive")
    if args.samples_per_client <= 0:
        parser.error("samples-per-client must be positive")
    if args.num_features <= 0:
        parser.error("num-features must be positive")
    if args.num_clusters <= 0:
        parser.error("num-clusters must be positive")
    if not (0.0 <= args.non_iid_factor <= 1.0):
        parser.error("non-iid-factor must be between 0.0 and 1.0")
    if not (0.0 <= args.overlap_factor <= 1.0):
        parser.error("overlap-factor must be between 0.0 and 1.0")
    if not (0.0 <= args.density_variation <= 1.0):
        parser.error("density-variation must be between 0.0 and 1.0")

    # Generate and save data
    output_dir = Path(args.output_dir)
    generate_and_save_data(
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        num_features=args.num_features,
        num_clusters=args.num_clusters,
        cluster_std=args.cluster_std,
        non_iid_factor=args.non_iid_factor,
        overlap_factor=args.overlap_factor,
        density_variation=args.density_variation,
        random_state=args.random_state,
        output_dir=output_dir,
    )

    logger.info(f"\nData generation complete! Files saved to: {output_dir.absolute()}")
    logger.info(f"\nTo use this data, set data-path in pyproject.toml to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

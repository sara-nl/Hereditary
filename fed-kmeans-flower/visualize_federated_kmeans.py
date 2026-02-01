#!/usr/bin/env python3
"""
Visualization script for federated k-means clustering.

Creates an animated video showing:
1. Client data with global means overlay
2. Local clustering results per client
3. Global aggregation of local means
4. Evolution across all rounds

Usage:
    python visualize_federated_kmeans.py --experiment-dir logs/experiment_20251107_160527
    python visualize_federated_kmeans.py --experiment-dir logs/experiment_20251107_160527 --output video.mp4 --fps 2
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FederatedKMeansVisualizer:
    """Visualizer for federated k-means clustering experiments."""

    def __init__(self, experiment_dir: str):
        """Initialize visualizer with experiment directory.

        Args:
            experiment_dir: Path to experiment directory containing logs
        """
        self.experiment_dir = Path(experiment_dir)
        self.global_means_dir = self.experiment_dir / "global_means"
        self.local_means_dir = self.experiment_dir / "local_means"
        self.pca_dir = self.experiment_dir / "pca"
        self.pca_data_dir = self.experiment_dir / "pca_data"

        # Load experiment summary
        summary_file = self.experiment_dir / "experiment_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                self.summary = json.load(f)
        else:
            self.summary = {}

        # Discover available rounds and clients
        self.rounds = self._discover_rounds()
        self.clients = self._discover_clients()

        # Load PCA components if available
        self.pca_mean = None
        self.principal_components = None
        self.pca_metadata = None
        self.use_pca = False
        self._load_pca_components()

        # Load client data
        self.client_data = self._load_client_data()

        logger.info(f"Loaded experiment: {len(self.rounds)} rounds, {len(self.clients)} clients")
        if self.use_pca:
            logger.info(
                f"Using PCA projection for visualization (explained variance: {self.pca_metadata.get('explained_variance_ratio', [])})"
            )

    def _discover_rounds(self) -> List[int]:
        """Discover available rounds from global means directory."""
        rounds = []
        if self.global_means_dir.exists():
            for file in sorted(self.global_means_dir.glob("round_*_global_means.npy")):
                round_num = int(file.stem.split("_")[1])
                rounds.append(round_num)
        return sorted(rounds)

    def _discover_clients(self) -> List[str]:
        """Discover available clients from local means directory."""
        clients = []
        if self.local_means_dir.exists():
            for client_dir in sorted(self.local_means_dir.iterdir()):
                if client_dir.is_dir() and client_dir.name.startswith("client_"):
                    # Extract client ID, handling both filtered and non-filtered
                    client_id = client_dir.name.replace("client_", "")
                    if not client_id.endswith("_filtered"):
                        clients.append(client_id)
        return sorted(clients)

    def _get_axis_labels(self) -> Tuple[str, str]:
        """Get appropriate axis labels based on whether PCA is used."""
        if self.use_pca and self.pca_metadata:
            variance_ratio = self.pca_metadata.get("explained_variance_ratio", [0, 0])
            xlabel = f"PC1 ({variance_ratio[0]*100:.1f}%)"
            ylabel = f"PC2 ({variance_ratio[1]*100:.1f}%)"
        else:
            xlabel = "Feature 1"
            ylabel = "Feature 2"
        return xlabel, ylabel

    def _get_global_plot_limits(self, global_means: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate global axis limits based on all client data and global means.

        Args:
            global_means: Array of global means with shape (n_clusters, n_features)

        Returns:
            Tuple of ((xmin, xmax), (ymin, ymax)) that can be used with ax.set_xlim and ax.set_ylim
        """
        # Initialize with global means bounds
        x_min, y_min = np.min(global_means, axis=0)[:2]  # Only consider first two dimensions
        x_max, y_max = np.max(global_means, axis=0)[:2]

        # Expand bounds to include all client data
        for client_data in self.client_data.values():
            if client_data.size > 0 and client_data.shape[0] > 0:
                client_x_min, client_y_min = np.min(client_data[:, :2], axis=0)
                client_x_max, client_y_max = np.max(client_data[:, :2], axis=0)

                x_min = min(x_min, client_x_min)
                y_min = min(y_min, client_y_min)
                x_max = max(x_max, client_x_max)
                y_max = max(y_max, client_y_max)

        # Add padding (10% of the range)
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1

        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)

        return x_limits, y_limits

    def _load_pca_components(self) -> None:
        """Load PCA components if available for high-dimensional data."""
        if not self.pca_dir.exists():
            return

        pca_mean_file = self.pca_dir / "pca_mean.npy"
        pca_components_file = self.pca_dir / "principal_components.npy"
        pca_metadata_file = self.pca_dir / "pca_metadata.json"

        if pca_mean_file.exists() and pca_components_file.exists():
            try:
                self.pca_mean = np.load(pca_mean_file)
                self.principal_components = np.load(pca_components_file)

                if pca_metadata_file.exists():
                    with open(pca_metadata_file) as f:
                        self.pca_metadata = json.load(f)

                self.use_pca = True
                logger.info(f"Loaded PCA components: {self.principal_components.shape}")

            except Exception as e:
                logger.warning(f"Failed to load PCA components: {e}")
                self.use_pca = False

    def _load_client_data(self) -> Dict[str, np.ndarray]:
        """Load client data from data directory or PCA-projected data."""
        client_data = {}

        # Get client ID to partition mapping from summary
        client_mapping = self.summary.get("client_mapping", {})

        # First, try to load PCA-projected data if available
        if self.use_pca and self.pca_data_dir.exists():
            logger.info("Loading PCA-projected client data...")
            for client_id in self.clients:
                projected_file = self.pca_data_dir / f"client_{client_id}_projected.npy"
                if projected_file.exists():
                    try:
                        client_data[client_id] = np.load(projected_file)
                        logger.info(f"Loaded PCA-projected data for client {client_id}: {client_data[client_id].shape}")
                    except Exception as e:
                        logger.warning(f"Failed to load PCA-projected data for client {client_id}: {e}")

            if client_data:
                logger.info(f"Using PCA-projected data for {len(client_data)} clients")
                return client_data

        # Otherwise, load original data using partition mapping and apply correct normalization
        data_path = self.summary.get("config", {}).get("data_path", "./test_data")
        data_dir = Path(data_path)

        # Make path absolute if relative
        if not data_dir.is_absolute():
            # Try relative to experiment dir first
            test_path = self.experiment_dir.parent.parent / data_path
            if test_path.exists():
                data_dir = test_path
            else:
                # Try as-is
                data_dir = Path(data_path).absolute()

        logger.info(f"Loading client data from: {data_dir}")

        if data_dir.exists():
            # Check if this is CLEF data by looking for ALS data format
            is_clef_data = self._detect_clef_data()
            
            for client_id in self.clients:
                # Get partition index for this client ID
                partition_index = client_mapping.get(client_id)

                if partition_index is None:
                    logger.warning(f"No partition mapping found for client {client_id}")
                    continue

                # Try different client file naming patterns using partition index
                for pattern in [f"client_{partition_index}.npz", f"client_{partition_index}.npy"]:
                    client_file = data_dir / pattern

                    if client_file.exists():
                        try:
                            if client_file.suffix == ".npz":
                                data = np.load(client_file)
                                raw_data = data["data"]
                            else:
                                raw_data = np.load(client_file)

                            # Apply correct normalization and PCA projection
                            processed_data = self._process_client_data_for_visualization(
                                raw_data, client_id, partition_index, is_clef_data
                            )
                            
                            if processed_data is not None:
                                client_data[client_id] = processed_data
                                logger.info(
                                    f"Loaded and processed data for client {client_id} (partition {partition_index}): {client_data[client_id].shape}"
                                )
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load data for client {client_id} from {client_file}: {e}")
        else:
            logger.warning(f"Data directory does not exist: {data_dir}")

        if not client_data:
            logger.warning("No client data loaded - will visualize means only")

        return client_data

    def _detect_clef_data(self) -> bool:
        """Detect if this experiment used CLEF data based on experiment configuration."""
        data_source = self.summary.get("config", {}).get("data_source", "directory")
        if "ALS" in str(data_source):
            return True
        return False

    def _process_client_data_for_visualization(
        self, raw_data: np.ndarray, client_id: str, partition_index: int, is_clef_data: bool
    ) -> Optional[np.ndarray]:
        """
        Process client data for visualization, applying correct normalization and PCA.
        
        This handles the mismatch between old experiments (all columns normalized) 
        and new experiments (only continuous columns normalized).
        """
        try:
            if is_clef_data and raw_data.shape[1] > 2:
                logger.info(f"Processing CLEF data for client {client_id} with CLEF-specific normalization")
                
                # Load CLEF data properly with correct normalization
                processed_data = self._load_clef_data_correctly(partition_index)
                if processed_data is not None:
                    raw_data = processed_data
                    logger.info(f"Applied CLEF-specific normalization for client {client_id}")
                else:
                    logger.warning(f"Failed to apply CLEF normalization for client {client_id}, using raw data")

            # Project to PCA if needed and components available
            if self.use_pca and raw_data.shape[1] > 2:
                from fed_kmeans_flower.preprocessing import project_to_pca

                projected_data = project_to_pca(raw_data, self.pca_mean, self.principal_components)
                return projected_data
            else:
                return raw_data
                
        except Exception as e:
            logger.error(f"Error processing data for client {client_id}: {e}")
            return None

    def _load_clef_data_correctly(self, partition_index: int) -> Optional[np.ndarray]:
        """
        Load CLEF data with correct CLEF-specific normalization.
        
        This ensures we get the same normalization as the current federated implementation
        (only continuous columns normalized).
        """
        try:
            import os
            import sys
            from fed_kmeans_flower.data.clef_loading import get_data
            from fed_kmeans_flower.data.clef_preprocessing import (
                aggregate_clef_statistics, calculate_clef_local_statistics,
                normalize_clef_data)

            # Add the project root to path to import CLEF modules
            project_root = self.experiment_dir.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            

            # Get CLEF data path from environment or use default
            base_path = os.environ.get("CLEF_DATA_PATH")
            if not base_path:
                logger.warning("CLEF_DATA_PATH not set, cannot load CLEF data correctly")
                return None
                
            # Map partition index to partition letter
            partition_mapping = {0: "T", 1: "L", 2: "U"}
            partition_letter = partition_mapping.get(partition_index)
            
            if partition_letter is None:
                logger.warning(f"Unknown partition index: {partition_index}")
                return None
                
            partition_path = os.path.join(base_path, partition_letter, "datasetC")
            
            # Load data with CLEF-specific preprocessing
            X_train, y_train, X_test, y_test, continuous_indices = get_data(partition_path)
            
            # Simulate federated normalization by computing global stats across all partitions
            all_client_data = []
            all_stats = []
            
            for p_idx, p_letter in partition_mapping.items():
                try:
                    p_path = os.path.join(base_path, p_letter, "datasetC") 
                    X_p_train, _, _, _, _ = get_data(p_path)
                    all_client_data.append(X_p_train)
                    
                    # Calculate local statistics for this partition
                    local_stats = calculate_clef_local_statistics(X_p_train, continuous_indices)
                    all_stats.append(local_stats)
                except Exception as e:
                    logger.warning(f"Failed to load partition {p_letter}: {e}")
            
            # Aggregate statistics to get global normalization parameters
            global_mean, global_std, _ = aggregate_clef_statistics(all_stats)
            normalized_data = normalize_clef_data(X_train, global_mean, global_std, continuous_indices)
            
            logger.info(f"Applied federated CLEF normalization: {len(continuous_indices)} continuous columns normalized")
            return normalized_data

                
        except Exception as e:
            logger.error(f"Error loading CLEF data correctly: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_global_means(self, round_num: int) -> Optional[np.ndarray]:
        """Load global means for a specific round, projecting to PCA if needed."""
        means_file = self.global_means_dir / f"round_{round_num:03d}_global_means.npy"
        if means_file.exists():
            means = np.load(means_file)

            # Project to PCA space if needed
            if self.use_pca and means.shape[1] > 2:
                from fed_kmeans_flower.preprocessing import project_to_pca
                means = project_to_pca(means, self.pca_mean, self.principal_components)

            return means
        return None

    def load_global_metadata(self, round_num: int) -> Dict:
        """Load global means metadata for a specific round."""
        meta_file = self.global_means_dir / f"round_{round_num:03d}_metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                return json.load(f)
        return {}

    def load_local_means(self, client_id: str, round_num: int) -> Optional[np.ndarray]:
        """Load local means for a specific client and round, projecting to PCA if needed."""
        client_dir = self.local_means_dir / f"client_{client_id}"
        means_file = client_dir / f"round_{round_num:03d}_local_means.npy"
        if means_file.exists():
            means = np.load(means_file)

            # Project to PCA space if needed
            if self.use_pca and means.shape[1] > 2:
                from fed_kmeans_flower.preprocessing import project_to_pca
                means = project_to_pca(means, self.pca_mean, self.principal_components)

            return means
        return None

    def load_local_sample_counts(self, client_id: str, round_num: int) -> Optional[np.ndarray]:
        """Load local sample counts for a specific client and round."""
        client_dir = self.local_means_dir / f"client_{client_id}"
        counts_file = client_dir / f"round_{round_num:03d}_sample_counts.npy"
        if counts_file.exists():
            return np.load(counts_file)
        return None

    def create_visualization_frames(self, output_dir: Optional[str] = None) -> List[str]:
        """Create visualization frames for all rounds.

        Args:
            output_dir: Directory to save frames (default: experiment_dir/frames)

        Returns:
            List of frame file paths
        """
        if output_dir is None:
            output_dir = self.experiment_dir / "frames"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_files = []
        frame_idx = 0

        for round_num in self.rounds:
            logger.info(f"Creating frames for round {round_num}")

            # Frame 1: Global means with client data
            frame_file = output_dir / f"frame_{frame_idx:04d}_round_{round_num:03d}_global.png"
            self._create_global_means_frame(round_num, frame_file)
            frame_files.append(str(frame_file))
            frame_idx += 1

            # Frame 2: Local clustering results
            if round_num > 0:  # Skip for initial round
                frame_file = output_dir / f"frame_{frame_idx:04d}_round_{round_num:03d}_local.png"
                self._create_local_means_frame(round_num, frame_file)
                frame_files.append(str(frame_file))
                frame_idx += 1

                # Frame 3: Aggregation visualization
                frame_file = output_dir / f"frame_{frame_idx:04d}_round_{round_num:03d}_aggregation.png"
                self._create_aggregation_frame(round_num, frame_file)
                frame_files.append(str(frame_file))
                frame_idx += 1

        logger.info(f"Created {len(frame_files)} frames in {output_dir}")
        return frame_files

    def _create_global_means_frame(self, round_num: int, output_file: Path) -> None:
        """Create frame showing global means overlaid on client data."""
        try:
            global_means = self.load_global_means(round_num)
            metadata = self.load_global_metadata(round_num)

            if global_means is None:
                logger.warning(f"No global means found for round {round_num}")
                return

            # Determine number of clients to display
            num_clients = len(self.clients)
            if num_clients == 0:
                num_clients = 1  # At least show global means

            # Create figure with subplots for each client
            cols = min(3, num_clients)
            rows = (num_clients + cols - 1) // cols

            fig = plt.figure(figsize=(6 * cols, 5 * rows + 1.5))
            gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3, top=0.88, bottom=0.08)
            # Calculate global axis limits based on all data
            x_limits, y_limits = self._get_global_plot_limits(global_means)
            # Plot each client's data with global means
            for idx, client_id in enumerate(self.clients):
                row = idx // cols
                col = idx % cols
                ax = fig.add_subplot(gs[row, col])

                # Plot client data if available
                if client_id in self.client_data:
                    data = self.client_data[client_id]
                    if data.shape[1] >= 2:
                        ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=20, c="gray", label="Data")

                # Plot global means
                if global_means.shape[1] >= 2:
                    ax.scatter(
                        global_means[:, 0],
                        global_means[:, 1],
                        c="red",
                        s=200,
                        marker="X",
                        edgecolors="black",
                        linewidths=2,
                        label="Global Means",
                        zorder=5,
                    )

                    # Add cluster labels
                    for i, mean in enumerate(global_means):
                        ax.annotate(
                            f"G{i}",
                            (mean[0], mean[1]),
                            fontsize=10,
                            fontweight="bold",
                            ha="center",
                            va="center",
                            color="white",
                        )
                # Set consistent axis limits for all subplots
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)
                ax.set_title(f"Client {client_id}", fontsize=12, fontweight="bold")
                xlabel, ylabel = self._get_axis_labels()
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)

            # Add title with round information
            convergence = metadata.get("convergence_change", "N/A")
            converged = metadata.get("is_converged", False)
            status = "CONVERGED" if converged else "Training"

            fig.suptitle(
                f"Round {round_num}: Global Means ({status})\n"
                f'Convergence Change: {convergence if isinstance(convergence, str) else f"{convergence:.6f}"}',
                fontsize=16,
                fontweight="bold",
                y=0.94,
            )

            plt.savefig(output_file, dpi=100, bbox_inches="tight", facecolor="white", pad_inches=0.1)
            plt.close(fig)
            logger.debug(f"Created global means frame for round {round_num}")
        except Exception as e:
            logger.error(f"Error creating global means frame for round {round_num}: {e}")
            import traceback

            traceback.print_exc()

    def _create_local_means_frame(self, round_num: int, output_file: Path) -> None:
        """Create frame showing local clustering results for each client."""
        try:
            global_means = self.load_global_means(round_num - 1)  # Previous round's global means

            num_clients = len(self.clients)
            cols = min(3, num_clients)
            rows = (num_clients + cols - 1) // cols

            fig = plt.figure(figsize=(6 * cols, 5 * rows + 1.5))
            gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3, top=0.88, bottom=0.08)
            x_limits, y_limits = self._get_global_plot_limits(global_means)

            for idx, client_id in enumerate(self.clients):
                row = idx // cols
                col = idx % cols
                ax = fig.add_subplot(gs[row, col])

                # Plot client data
                if client_id in self.client_data:
                    data = self.client_data[client_id]
                    if data.shape[1] >= 2:
                        ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=20, c="gray", label="Data")

                # Plot previous global means (faded)
                if global_means is not None and global_means.shape[1] >= 2:
                    ax.scatter(
                        global_means[:, 0],
                        global_means[:, 1],
                        c="red",
                        s=100,
                        marker="X",
                        alpha=0.3,
                        edgecolors="black",
                        linewidths=1,
                        label="Prev Global",
                    )

                # Plot local means
                sample_counts = self.load_local_sample_counts(client_id, round_num)
                local_means = self.load_local_means(client_id, round_num)

                if local_means is not None and local_means.shape[1] >= 2:
                    # Size markers by sample count if available
                    sizes = sample_counts * 10 if sample_counts is not None else [150] * len(local_means)

                    ax.scatter(
                        local_means[:, 0],
                        local_means[:, 1],
                        c="blue",
                        s=sizes,
                        marker="o",
                        edgecolors="black",
                        linewidths=2,
                        label="Local Means",
                        zorder=5,
                        alpha=0.8,
                    )

                    # Add cluster labels with sample counts
                    for i, mean in enumerate(local_means):
                        count_text = f"\n({int(sample_counts[i])})" if sample_counts is not None else ""
                        ax.annotate(
                            f"L{i}{count_text}",
                            (mean[0], mean[1]),
                            fontsize=9,
                            fontweight="bold",
                            ha="center",
                            va="center",
                            color="white",
                        )

                ax.set_title(f"Client {client_id}", fontsize=12, fontweight="bold")
                # Set consistent axis limits for all subplots
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)
                xlabel, ylabel = self._get_axis_labels()
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)

            fig.suptitle(
                f"Round {round_num}: Local Clustering Results\n" f"Each client computes local means from global means",
                fontsize=16,
                fontweight="bold",
                y=0.94,
            )

            plt.savefig(output_file, dpi=100, bbox_inches="tight", facecolor="white", pad_inches=0.1)
            plt.close(fig)
            logger.debug(f"Created local means frame for round {round_num}")
        except Exception as e:
            logger.error(f"Error creating local means frame for round {round_num}: {e}")
            import traceback

            traceback.print_exc()

    def _create_aggregation_frame(self, round_num: int, output_file: Path) -> None:
        """Create frame showing aggregation of local means into new global means."""
        try:
            new_global_means = self.load_global_means(round_num)
            old_global_means = self.load_global_means(round_num - 1)

            fig = plt.figure(figsize=(16, 6.5))
            gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3, top=0.85, bottom=0.12)

            # Left: All local means
            ax1 = fig.add_subplot(gs[0, 0])
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.clients)))

            for idx, client_id in enumerate(self.clients):
                local_means = self.load_local_means(client_id, round_num)
                sample_counts = self.load_local_sample_counts(client_id, round_num)

                if local_means is not None and local_means.shape[1] >= 2:
                    sizes = sample_counts * 10 if sample_counts is not None else [100] * len(local_means)
                    ax1.scatter(
                        local_means[:, 0],
                        local_means[:, 1],
                        c=[colors[idx]],
                        s=sizes,
                        marker="o",
                        edgecolors="black",
                        linewidths=1.5,
                        label=f"Client {client_id}",
                        alpha=0.7,
                    )

            ax1.set_title("Local Means from All Clients", fontsize=12, fontweight="bold")
            xlabel, ylabel = self._get_axis_labels()
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            ax1.legend(loc="upper right", fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Middle: Aggregation arrow
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.axis("off")
            ax2.annotate(
                "",
                xy=(0.8, 0.5),
                xytext=(0.2, 0.5),
                arrowprops=dict(arrowstyle="->", lw=5, color="green"),
                xycoords="axes fraction",
            )
            ax2.text(
                0.5,
                0.6,
                "Weighted\nAggregation",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                transform=ax2.transAxes,
            )
            ax2.text(0.5, 0.4, "(K-means++)", ha="center", va="center", fontsize=10, transform=ax2.transAxes)

            # Right: New global means vs old
            ax3 = fig.add_subplot(gs[0, 2])

            # Plot old global means (faded)
            if old_global_means is not None and old_global_means.shape[1] >= 2:
                ax3.scatter(
                    old_global_means[:, 0],
                    old_global_means[:, 1],
                    c="red",
                    s=150,
                    marker="X",
                    alpha=0.3,
                    edgecolors="black",
                    linewidths=1,
                    label="Old Global",
                )

                # Draw arrows showing movement
                if new_global_means is not None and new_global_means.shape[1] >= 2:
                    for old, new in zip(old_global_means, new_global_means):
                        ax3.annotate(
                            "",
                            xy=new[:2],
                            xytext=old[:2],
                            arrowprops=dict(arrowstyle="->", lw=1.5, color="orange", alpha=0.6),
                        )

            # Plot new global means
            if new_global_means is not None and new_global_means.shape[1] >= 2:
                ax3.scatter(
                    new_global_means[:, 0],
                    new_global_means[:, 1],
                    c="red",
                    s=200,
                    marker="X",
                    edgecolors="black",
                    linewidths=2,
                    label="New Global",
                    zorder=5,
                )

                for i, mean in enumerate(new_global_means):
                    ax3.annotate(
                        f"G{i}",
                        (mean[0], mean[1]),
                        fontsize=10,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        color="white",
                    )

            ax3.set_title("Updated Global Means", fontsize=12, fontweight="bold")
            xlabel, ylabel = self._get_axis_labels()
            ax3.set_xlabel(xlabel)
            ax3.set_ylabel(ylabel)
            ax3.legend(loc="upper right", fontsize=9)
            ax3.grid(True, alpha=0.3)

            metadata = self.load_global_metadata(round_num)
            convergence = metadata.get("convergence_change", "N/A")

            fig.suptitle(
                f"Round {round_num}: Global Aggregation\n"
                f'Convergence Change: {convergence if isinstance(convergence, str) else f"{convergence:.6f}"}',
                fontsize=16,
                fontweight="bold",
                y=0.92,
            )

            plt.savefig(output_file, dpi=100, bbox_inches="tight", facecolor="white", pad_inches=0.1)
            plt.close(fig)
            logger.debug(f"Created aggregation frame for round {round_num}")
        except Exception as e:
            logger.error(f"Error creating aggregation frame for round {round_num}: {e}")
            import traceback

            traceback.print_exc()

    def create_video(
        self, output_file: str = "federated_kmeans.mp4", fps: int = 2, cleanup_frames: bool = False
    ) -> str:
        """Create video from visualization frames.

        Args:
            output_file: Output video file path
            fps: Frames per second
            cleanup_frames: Whether to delete frames after creating video

        Returns:
            Path to created video file
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV (cv2) is required for video creation. Install with: pip install opencv-python")
            return None

        # Create frames
        logger.info("Creating visualization frames...")
        frame_files = self.create_visualization_frames()

        if not frame_files:
            logger.error("No frames created")
            return None

        logger.info(f"Created {len(frame_files)} frames")

        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            logger.error(f"Failed to read first frame: {frame_files[0]}")
            return None

        height, width, _ = first_frame.shape
        logger.info(f"Frame dimensions: {width}x{height}")

        # Create video writer with H264 codec (more compatible)
        output_path = self.experiment_dir / output_file

        # Try different codecs in order of preference
        codecs = [
            ("avc1", ".mp4"),  # H.264 (best compatibility)
            ("mp4v", ".mp4"),  # MPEG-4
            ("XVID", ".avi"),  # Xvid (fallback)
        ]

        video_writer = None
        for codec, ext in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_output = str(output_path).replace(".mp4", ext)
                video_writer = cv2.VideoWriter(test_output, fourcc, fps, (width, height))

                # Test if writer is opened successfully
                if video_writer.isOpened():
                    output_path = Path(test_output)
                    logger.info(f"Using codec: {codec}")
                    break
                else:
                    video_writer.release()
                    video_writer = None
            except Exception as e:
                logger.warning(f"Codec {codec} failed: {e}")
                if video_writer:
                    video_writer.release()
                video_writer = None

        if video_writer is None or not video_writer.isOpened():
            logger.error("Failed to create video writer with any codec")
            return None

        logger.info(f"Creating video: {output_path}")

        # Write frames to video
        frames_written = 0
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(frame_file)

            if frame is None:
                logger.warning(f"Failed to read frame {i}: {frame_file}")
                continue

            # Ensure frame has correct dimensions
            if frame.shape[:2] != (height, width):
                logger.warning(
                    f"Frame {i} has different dimensions: {frame.shape[:2]} vs {(height, width)}, resizing..."
                )
                frame = cv2.resize(frame, (width, height))

            video_writer.write(frame)
            frames_written += 1

            if (i + 1) % 5 == 0:
                logger.info(f"Written {i + 1}/{len(frame_files)} frames")

        video_writer.release()

        logger.info(f"Video created: {output_path} ({frames_written} frames written)")

        # Verify video file exists and has size
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Video file size: {output_path.stat().st_size / 1024:.1f} KB")
        else:
            logger.error("Video file is empty or doesn't exist")
            return None

        # Cleanup frames if requested
        if cleanup_frames:
            frames_dir = self.experiment_dir / "frames"
            for frame_file in frame_files:
                Path(frame_file).unlink()
            logger.info("Cleaned up frame files")

        return str(output_path)


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize federated k-means clustering experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create video from experiment
  python visualize_federated_kmeans.py --experiment-dir logs/experiment_20251107_160527
  
  # Create video with custom settings
  python visualize_federated_kmeans.py --experiment-dir logs/experiment_20251107_160527 \\
      --output my_video.mp4 --fps 1 --cleanup
        """,
    )

    parser.add_argument(
        "--experiment-dir", type=str, required=True, help="Path to experiment directory containing logs"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="federated_kmeans.mp4",
        help="Output video filename (default: federated_kmeans.mp4)",
    )

    parser.add_argument("--fps", type=int, default=1, help="Frames per second for video (default: 1)")

    parser.add_argument("--cleanup", action="store_true", help="Delete frame files after creating video")

    args = parser.parse_args()

    # Create visualizer
    visualizer = FederatedKMeansVisualizer(args.experiment_dir)

    # Create video
    video_path = visualizer.create_video(output_file=args.output, fps=args.fps, cleanup_frames=args.cleanup)

    if video_path:
        logger.info(f"✓ Video created successfully: {video_path}")
        return 0
    else:
        logger.error("✗ Failed to create video")
        return 1


if __name__ == "__main__":
    exit(main())

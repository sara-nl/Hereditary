"""Logging utilities for tracking local and global means during federated k-means."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MeansLogger:
    """Logger for tracking local and global cluster means during federated k-means."""

    def __init__(self, base_log_dir: str = "./logs"):
        """Initialize means logger with timestamped experiment directory.

        Args:
            base_log_dir: Base directory for all experiment logs
        """
        self.base_log_dir = Path(base_log_dir)

        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_log_dir / f"experiment_{timestamp}"

        # Create subdirectories for different log types
        self.global_means_dir = self.experiment_dir / "global_means"
        self.local_means_dir = self.experiment_dir / "local_means"
        self.client_logs_dir = self.experiment_dir / "client_logs"
        self.server_logs_dir = self.experiment_dir / "server_logs"

        # Create all directories
        for directory in [self.global_means_dir, self.local_means_dir, self.client_logs_dir, self.server_logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Store experiment metadata
        self.metadata = {
            "experiment_id": timestamp,
            "start_time": datetime.now().isoformat(),
            "base_log_dir": str(self.base_log_dir),
            "experiment_dir": str(self.experiment_dir),
        }

        logger.info(f"Initialized MeansLogger with experiment directory: {self.experiment_dir}")

    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable Python types.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def log_global_means(
        self, round_number: int, global_means: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log global cluster means for a specific round.

        Args:
            round_number: Current round number
            global_means: Global cluster means array
            metadata: Optional metadata to include (convergence, num_clients, etc.)
        """
        try:
            # Save means as numpy file
            means_file = self.global_means_dir / f"round_{round_number:03d}_global_means.npy"
            np.save(means_file, global_means)

            # Save metadata as JSON
            meta_file = self.global_means_dir / f"round_{round_number:03d}_metadata.json"
            meta_data = {
                "round_number": int(round_number),
                "timestamp": datetime.now().isoformat(),
                "shape": list(global_means.shape),
                "num_clusters": int(global_means.shape[0]),
                "num_features": int(global_means.shape[1]),
                "mean_values": global_means.mean(axis=0).tolist(),
                "std_values": global_means.std(axis=0).tolist(),
            }

            if metadata:
                # Convert metadata to JSON-serializable format
                meta_data.update(self._convert_to_json_serializable(metadata))

            with open(meta_file, "w") as f:
                json.dump(meta_data, f, indent=2)

            logger.debug(f"Logged global means for round {round_number}: shape {global_means.shape}")

        except Exception as e:
            logger.error(f"Failed to log global means for round {round_number}: {e}")

    def log_local_means(
        self,
        round_number: int,
        client_id: str,
        local_means: np.ndarray,
        sample_counts: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log local cluster means for a specific client and round.

        Args:
            round_number: Current round number
            client_id: Client identifier
            local_means: Local cluster means array
            sample_counts: Optional sample counts per cluster
            metadata: Optional metadata to include
        """
        try:
            # Create client-specific directory
            client_dir = self.local_means_dir / f"client_{client_id}"
            client_dir.mkdir(parents=True, exist_ok=True)

            # Save means as numpy file
            means_file = client_dir / f"round_{round_number:03d}_local_means.npy"
            np.save(means_file, local_means)

            # Save sample counts if provided
            if sample_counts is not None:
                counts_file = client_dir / f"round_{round_number:03d}_sample_counts.npy"
                np.save(counts_file, sample_counts)

            # Save metadata as JSON
            meta_file = client_dir / f"round_{round_number:03d}_metadata.json"
            meta_data = {
                "round_number": int(round_number),
                "client_id": str(client_id),
                "timestamp": datetime.now().isoformat(),
                "shape": list(local_means.shape),
                "num_clusters": int(local_means.shape[0]),
                "num_features": int(local_means.shape[1]) if local_means.size > 0 else 0,
            }

            if local_means.size > 0:
                meta_data["mean_values"] = local_means.mean(axis=0).tolist()
                meta_data["std_values"] = local_means.std(axis=0).tolist()

            if sample_counts is not None:
                meta_data["sample_counts"] = sample_counts.tolist()
                meta_data["total_samples"] = int(sample_counts.sum())

            if metadata:
                # Convert metadata to JSON-serializable format
                meta_data.update(self._convert_to_json_serializable(metadata))

            with open(meta_file, "w") as f:
                json.dump(meta_data, f, indent=2)

            logger.debug(f"Logged local means for client {client_id}, round {round_number}: shape {local_means.shape}")

        except Exception as e:
            logger.error(f"Failed to log local means for client {client_id}, round {round_number}: {e}")

    def get_server_log_path(self) -> Path:
        """Get the log file path for the server.

        Returns:
            Absolute path to server log file
        """
        return (self.server_logs_dir / "server.log").resolve()

    def save_experiment_summary(self, summary: Dict[str, Any]) -> None:
        """Save experiment summary to JSON file.

        Args:
            summary: Dictionary containing experiment summary data
        """
        try:
            summary_file = self.experiment_dir / "experiment_summary.json"

            # Merge with existing metadata and convert to JSON-serializable format
            full_summary = {**self.metadata, **self._convert_to_json_serializable(summary)}
            full_summary["end_time"] = datetime.now().isoformat()

            with open(summary_file, "w") as f:
                json.dump(full_summary, f, indent=2)

            logger.info(f"Saved experiment summary to {summary_file}")

        except Exception as e:
            logger.error(f"Failed to save experiment summary: {e}")

    def __str__(self) -> str:
        """String representation of the logger."""
        return f"MeansLogger(experiment_dir={self.experiment_dir})"

"""Logging, monitoring, and debugging utilities for federated k-means clustering."""

import json
import logging
import sys
from dataclasses import asdict
from typing import Optional

from .models import FederatedKMeansConfig


class FederatedKMeansLogger:
    """Enhanced logger for federated k-means operations."""

    def __init__(self, name: str, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize the federated k-means logger.

        Args:
            name: Logger name (typically __name__)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            # Ensure parent directory exists
            from pathlib import Path

            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(str(log_path))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_config(self, config: FederatedKMeansConfig):
        """Log configuration details."""
        self.logger.info("=== Federated K-means Configuration ===")
        for key, value in asdict(config).items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=====================================")

    def log_initialization(self, client_id: str, data_shape: tuple, success: bool):
        """Log client initialization."""
        if success:
            self.logger.info(f"Client {client_id} initialized successfully with data shape {data_shape}")
        else:
            self.logger.error(f"Client {client_id} initialization failed")

    def export_metrics(self, filepath: str):
        """Export performance metrics to JSON file."""
        metrics_data = {
            "message": "Performance metrics tracking has been removed. Use means_logger for experiment data."
        }

        with open(filepath, "w") as f:
            json.dump(metrics_data, f, indent=2, default=str)

        self.logger.info(f"Metrics file created at {filepath}")


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None, enable_debug: bool = False
) -> FederatedKMeansLogger:
    """
    Set up logging for the federated k-means system.

    Args:
        log_level: Logging level
        log_file: Optional log file path
        enable_debug: Enable message flow debugging (currently unused)

    Returns:
        FederatedKMeansLogger instance
    """
    main_logger = FederatedKMeansLogger("fed_kmeans_flower", log_level, log_file)
    return main_logger


def log_system_info(logger: FederatedKMeansLogger):
    """Log system information for debugging."""
    import platform

    import psutil

    logger.logger.info("=== System Information ===")
    logger.logger.info(f"Platform: {platform.platform()}")
    logger.logger.info(f"Python version: {platform.python_version()}")
    logger.logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.logger.info("==========================")

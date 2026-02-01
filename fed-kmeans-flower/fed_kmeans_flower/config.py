"""Configuration management and validation for federated k-means clustering."""

from typing import Any, Dict

from .models import FederatedKMeansConfig


def load_config_from_run_config(run_config: Dict[str, Any]) -> FederatedKMeansConfig:
    """
    Load and validate configuration from Flower's run_config.

    Args:
        run_config: Configuration dictionary from Flower context

    Returns:
        Validated FederatedKMeansConfig instance

    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Map run_config keys to FederatedKMeansConfig fields
        config_mapping = {
            "k-global": "k_global",
            "max-iterations": "max_iterations",
            "privacy-threshold": "privacy_threshold",
            "convergence-tolerance": "convergence_tolerance",
            "local-kmeans-iterations": "local_kmeans_iterations",
            "weighted-aggregation": "weighted_aggregation",
            "required-clients": "required_clients",
            "client-timeout": "client_timeout",
            "evaluation-frequency": "evaluation_frequency",
            "recovery-timeout": "recovery_timeout",
            "recovery-retry-delay": "recovery_retry_delay",
        }

        # Convert run_config to config dict
        config_dict = {}
        for run_key, config_key in config_mapping.items():
            if run_key in run_config:
                config_dict[config_key] = run_config[run_key]

        # Create and validate config
        config = FederatedKMeansConfig(**config_dict)

        return config

    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid configuration: {e}")

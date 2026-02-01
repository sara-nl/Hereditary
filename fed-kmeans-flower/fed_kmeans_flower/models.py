"""Core data models and configuration classes for federated k-means clustering."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FederatedKMeansConfig:
    """Configuration for federated k-means clustering algorithm."""

    k_global: int  # Global number of clusters
    max_iterations: int  # Maximum federated rounds
    privacy_threshold: int  # Minimum samples per cluster (default: 2)
    convergence_tolerance: float  # Convergence threshold
    local_kmeans_iterations: int  # Local k-means iterations per round
    weighted_aggregation: bool  # Use sample-weighted aggregation
    required_clients: int  # Minimum clients for training
    client_timeout: float  # Timeout for client responses (seconds)
    evaluation_frequency: int  # Evaluate every N rounds
    max_retries: int = 2  # Number of retries after initial attempt
    retry_delay: float = 0.5  # Seconds between retries
    enable_recovery: bool = True  # Whether to attempt partition recovery
    recovery_timeout: float = 30.0  # Timeout for waiting for new nodes during recovery
    recovery_retry_delay: float = 5.0  # Seconds between discovery attempts during recovery

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate integer parameters
        for name in ["k_global", "max_iterations", "local_kmeans_iterations", "required_clients", "evaluation_frequency"]:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        
        # Validate privacy_threshold (minimum 1)
        if not isinstance(self.privacy_threshold, int) or self.privacy_threshold < 1:
            raise ValueError(f"privacy_threshold must be >= 1, got {self.privacy_threshold}")
        
        # Validate float parameters
        for name in ["convergence_tolerance", "client_timeout", "retry_delay", "recovery_timeout", "recovery_retry_delay"]:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number, got {value}")
        
        # Validate boolean parameters
        if not isinstance(self.enable_recovery, bool):
            raise ValueError(f"enable_recovery must be a boolean, got {type(self.enable_recovery)}")

        # Logical validation
        if self.privacy_threshold > self.k_global:
            raise ValueError(
                f"privacy_threshold ({self.privacy_threshold}) cannot be greater than k_global ({self.k_global}). "
                "Each cluster needs at least privacy_threshold samples, but there are only k_global clusters."
            )

@dataclass
class EvaluationResult:
    """Result returned by clients after evaluation."""

    adjusted_rand_score: float
    silhouette_score: float
    local_inertia: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class PreprocessingResult:
    """Result returned by clients with local statistics."""

    n_samples: int
    sum: np.ndarray
    sum_sq: np.ndarray
    success: bool
    error_message: Optional[str] = None


@dataclass
class ClientState:
    """Internal state maintained by each client."""

    client_id: str
    client_index: int  # Server-assigned unique client index (deprecated, use partition_id)
    partition_id: int  # Partition ID from node configuration
    local_data: Optional[np.ndarray]
    current_means: Optional[np.ndarray]
    data_dimensions: int
    num_samples: int
    is_initialized: bool
    ground_truth_labels: Optional[np.ndarray] = None  # For evaluation
    privacy_threshold: int = 2
    local_kmeans_iterations: int = 1
    log_file_path: Optional[str] = None  # Path to client log file for distributed execution
    data_source: str = "test_data"  # Data source type (e.g., "ALS", "test_data")
    data_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ServerState:
    """Internal state maintained by the server."""

    global_means: Optional[np.ndarray]
    current_round: int
    active_clients: List[str]
    convergence_history: List[float]
    is_converged: bool
    config: FederatedKMeansConfig
    # Preprocessing/normalization parameters
    global_mean: Optional[np.ndarray] = None
    global_std: Optional[np.ndarray] = None
    is_normalized: bool = False
    continuous_indices: Optional[List[int]] = None  # Indices of continuous columns for normalization
    # PCA-related attributes
    data_dimensions: int = 0
    pca_mean: Optional[np.ndarray] = None
    principal_components: Optional[np.ndarray] = None
    explained_variance: Optional[np.ndarray] = None

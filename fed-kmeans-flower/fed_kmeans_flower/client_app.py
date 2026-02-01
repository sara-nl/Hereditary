"""Federated K-means client application using Flower messaging API."""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from flwr.app import (ArrayRecord, ConfigRecord, Context, Message,
                      MetricRecord, RecordDict)
from flwr.clientapp import ClientApp
import matplotlib.pyplot as plt

from fed_kmeans_flower.clustering.local_kmeans import (
    assign_to_global_clusters, perform_complete_local_clustering)
from fed_kmeans_flower.communication import (create_message_content,
                                             extract_message_content)
from fed_kmeans_flower.data.clef_preprocessing import (
    calculate_clef_local_statistics, normalize_clef_data)
from fed_kmeans_flower.data.loader import (get_data_statistics,
                                           load_client_data,
                                           validate_data_format)
from fed_kmeans_flower.evaluation.metrics import \
    compute_comprehensive_local_metrics
from fed_kmeans_flower.logging_utils import (FederatedKMeansLogger,
                                             setup_logging)
from fed_kmeans_flower.models import ClientState
from fed_kmeans_flower.preprocessing import (calculate_local_covariance_stats,
                                             calculate_local_statistics,
                                             project_to_pca)

# Flower ClientApp
app = ClientApp()


def _get_logger(
    client_state: Optional[ClientState] = None, client_id: str = "unknown", log_file_path=None
) -> FederatedKMeansLogger:
    """Get or create a logger for the client.

    In distributed execution, this creates a new logger instance for each handler call
    using the log file path stored in client state. In simulation mode, it reuses
    the same log file.

    Args:
        client_state: Client state containing log file path (if available)
        client_id: Client ID for fallback log file naming
        log_file_path: Explicit log file path (overrides client_state)

    Returns:
        FederatedKMeansLogger instance
    """
    # Priority: explicit path > client_state path > fallback
    log_file = None
    if log_file_path:
        log_file = log_file_path
    elif client_state and client_state.log_file_path:
        log_file = client_state.log_file_path
    else:
        # Fallback for first initialization (should rarely be used)
        log_file = f"client_{client_id}.log"

    return setup_logging(log_level="INFO", log_file=log_file)


def _emergency_logger(message: str, context: Context, logger_var_name: str = "logger") -> None:
    """Emergency logging when regular logger might not be available.

    This function attempts to log an error message using the best available logger:
    1. Try to use the logger variable if it exists in the caller's scope
    2. Fall back to creating a temporary logger
    3. As a last resort, use the standard logging module

    Args:
        message: Error message to log
        context: Flower context for creating fallback logger
        logger_var_name: Name of the logger variable to check (default: "logger")
    """
    import inspect

    try:
        # Get the caller's frame to check for logger variable
        frame = inspect.currentframe().f_back
        if logger_var_name in frame.f_locals:
            frame.f_locals[logger_var_name].logger.error(message)
        else:
            # Create temporary logger as fallback
            temp_logger = _get_logger(None, context.node_id)
            temp_logger.logger.error(message)
    except:
        # Last resort: use standard logging
        logging.getLogger(__name__).error(message)


def _load_client_data(client_id: str, config: dict, partition_id: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load data for the client based on configuration.

    Args:
        client_id: Unique identifier for the client
        config: Configuration dictionary containing data loading parameters
        partition_id: Partition ID for loading the correct data partition

    Returns:
        Tuple of (data, labels) where labels may be None

    Raises:
        ValueError: If data loading fails or configuration is invalid
    """
    dataset_name = config.get("data_source", "test_data")

    logging.getLogger(__name__).info(f"Loading {dataset_name} for partition {partition_id}")

    data, labels, data_info = load_client_data(
        client_id=client_id,
        data_format=dataset_name,
        partition_id=partition_id
    )

    return data, labels, data_info


def _persist_client_state(context: Context, client_state: ClientState) -> None:
    """Persist client state using Flower's Context state management.

    Args:
        context: Flower context for state persistence
        client_state: Current client state to persist
    """
    try:

        # Use client_index as key to prevent collisions in simulation
        state_key = f"client_state_{client_state.client_index}"
        arrays_key = f"client_arrays_{client_state.client_index}"
        config_key = f"client_config_{client_state.client_index}"

        # Create state record for persistence (numeric values)
        state_record = MetricRecord(
            {
                "client_id": float(hash(client_state.client_id) % 1e15),  # Convert to float for MetricRecord
                "client_index": float(client_state.client_index),
                "partition_id": float(client_state.partition_id),
                "data_dimensions": float(client_state.data_dimensions),
                "num_samples": float(client_state.num_samples),
                "is_initialized": float(1.0 if client_state.is_initialized else 0.0),
                "privacy_threshold": float(client_state.privacy_threshold),
                "local_kmeans_iterations": float(client_state.local_kmeans_iterations),
            }
        )

        # Store string values (like log_file_path) in ConfigRecord
        config_record = ConfigRecord(
            {"client_id_str": client_state.client_id, 
            "log_file_path": client_state.log_file_path or "",
            "data_source": client_state.data_source,
            "data_info": json.dumps(client_state.data_info, ensure_ascii=False, separators=(',', ':'))}
        )

        # Store arrays separately using ArrayRecord - pass as list of numpy arrays
        arrays_list = []
        if client_state.local_data is not None:
            arrays_list.append(client_state.local_data)
        if client_state.current_means is not None:
            arrays_list.append(client_state.current_means)
        if hasattr(client_state, "ground_truth_labels") and client_state.ground_truth_labels is not None:
            arrays_list.append(client_state.ground_truth_labels)

        # Use Flower's state management with unique keys per client
        context.state.metric_records[state_key] = state_record
        context.state.config_records[config_key] = config_record
        if arrays_list:
            context.state.array_records[arrays_key] = ArrayRecord(arrays_list)

        logging.getLogger(__name__).debug(
            f"Persisted client state for {client_state.client_id} (index {client_state.client_index})"
        )

    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to persist client state: {e}")


def _restore_client_state(context: Context, client_index: int, client_id: str) -> Optional[ClientState]:
    """Restore client state from Flower's Context state management.

    Args:
        context: Flower context containing persisted state
        client_index: Client index to restore state for
        client_id: Client ID for logging

    Returns:
        Restored ClientState object or None if no state found
    """
    try:
        # Use client_index as key to retrieve correct state
        state_key = f"client_state_{client_index}"
        arrays_key = f"client_arrays_{client_index}"
        config_key = f"client_config_{client_index}"

        # Check if state exists for this client
        if state_key not in context.state.metric_records:
            return None

        # Restore basic state from MetricRecord
        state_record = context.state.metric_records[state_key]

        # Restore string values from ConfigRecord
        if config_key in context.state.config_records:
            config_record = context.state.config_records[config_key]
            restored_log_file_path = config_record.get("log_file_path", "")
            restored_data_source = config_record.get("data_source", None)
            restored_data_info = json.loads(config_record.get("data_info", "{}"))

        # Restore arrays from ArrayRecord (stored as list)
        local_data = None
        current_means = None
        ground_truth_labels = None

        if arrays_key in context.state.array_records:
            array_record = context.state.array_records[arrays_key]
            # Convert ArrayRecord to list of numpy arrays
            arrays_list = array_record.to_numpy_ndarrays()

            # Reconstruct arrays in the order they were stored
            if len(arrays_list) > 0:
                local_data = arrays_list[0]
            if len(arrays_list) > 1:
                current_means = arrays_list[1]
            if len(arrays_list) > 2:
                ground_truth_labels = arrays_list[2]

        # Reconstruct client state - convert floats back to appropriate types
        restored_state = ClientState(
            client_id=client_id,
            client_index=int(state_record["client_index"]),
            partition_id=int(state_record["partition_id"]),
            local_data=local_data,
            current_means=current_means,
            data_dimensions=int(state_record["data_dimensions"]),
            num_samples=int(state_record["num_samples"]),
            is_initialized=bool(state_record["is_initialized"]),
            privacy_threshold=int(state_record["privacy_threshold"]),
            local_kmeans_iterations=int(state_record["local_kmeans_iterations"]),
            log_file_path=restored_log_file_path or f"client_{client_id}.log",
            data_source=restored_data_source,
            data_info=restored_data_info
        )

        # Restore ground truth labels if available
        if ground_truth_labels is not None:
            restored_state.ground_truth_labels = ground_truth_labels

        logging.getLogger(__name__).debug(
            f"Restored client state for {restored_state.client_id} (index {client_index})"
        )
        return restored_state

    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to restore client state for index {client_index}: {e}")
        return None


@app.query("initialize_client")
def initialize_client(msg: Message, context: Context) -> Message:
    """Initialize client with configuration and load local data.

    This handler receives initialization parameters from the server,
    loads local data, validates configuration, and prepares the client
    for federated k-means clustering.

    """
    try:
        # Extract initialization message first
        content = extract_message_content(msg.content)
        config = content.get("config", {})

        # Validate required configuration parameters
        required_params = ["k_global", "max_iterations", "privacy_threshold", "convergence_tolerance"]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required configuration parameter: {param}")

        # Extract partition-id from node_config - this is required
        partition_id = context.node_config.get("partition-id")
        num_partitions = context.node_config.get("num-partitions")

        if partition_id is None:
            raise ValueError(
                "partition-id not found in node_config. Clients can only have one partition, "
                "and it must be properly configured. Please set 'partition-id' in node_config."
            )

        logging.getLogger(__name__).info(f"Using partition-id from node_config: {partition_id}")

        # Validate partition_id is within valid range if num_partitions is specified
        if num_partitions is not None:
            if not (0 <= partition_id < num_partitions):
                raise ValueError(f"partition_id {partition_id} is out of valid range [0, {num_partitions})")

        # Try to restore existing state first
        client_state = _restore_client_state(context, partition_id, context.node_id)

        # Initialize logger with explicit log file path
        client_log_path = os.path.join(
            config["experiment_dir"], "client_logs", f"client_{partition_id}_{context.node_id}.log"
        )
        logger = _get_logger(client_state, context.node_id, client_log_path)

        if client_state and client_state.is_initialized:
            logger.logger.info(
                f"Restored existing client state for {client_state.client_id} (partition {partition_id})"
            )
        else:
            # Create new client state
            client_state = ClientState(
                client_id=context.node_id,
                client_index=partition_id,
                partition_id=partition_id,
                local_data=None,
                current_means=None,
                data_dimensions=0,
                num_samples=0,
                is_initialized=False,
                privacy_threshold=config["privacy_threshold"],
                local_kmeans_iterations=config.get("local_kmeans_iterations", 1),
                log_file_path=client_log_path,
                data_source=config.get("data_source")
            )

            logger.logger.info(
                f"Initializing client {client_state.client_id} (partition {partition_id}) with k_global={config['k_global']}, data_source={client_state.data_source}"
            )

            # Load client data based on configuration
            try:
                data, labels, data_info = _load_client_data(client_state.client_id, config, partition_id)
                client_state.local_data = data
                client_state.data_dimensions = data.shape[1]
                client_state.num_samples = data.shape[0]
                client_state.data_info = data_info

                # Store ground truth labels if available (for evaluation)
                if labels is not None:
                    client_state.ground_truth_labels = labels

                logger.log_initialization(
                    client_state.client_id, (client_state.num_samples, client_state.data_dimensions), True
                )

            except Exception as e:
                raise ValueError(f"Failed to load client data: {e}")

            # Validate minimum sample requirements
            min_samples = max(config["privacy_threshold"], config["k_global"])
            if client_state.num_samples < min_samples:
                raise ValueError(
                    f"Insufficient samples: need at least {min_samples}, " f"got {client_state.num_samples}"
                )

            # Mark as initialized
            client_state.is_initialized = True

        # Compute and validate data statistics
        data_stats = get_data_statistics(client_state.local_data)

        # Persist client state to context
        _persist_client_state(context, client_state)
        # Create response message with data statistics (no raw data)
        response_content = create_message_content(
            metrics={
                "success": True,
                "partition_id": partition_id,
                "num_partitions": num_partitions,
                "data_dimensions": client_state.data_dimensions,
                "num_samples": client_state.num_samples,
                "feature_means": data_stats.feature_means.tolist(),
                "feature_stds": data_stats.feature_stds.tolist(),
                "data_range_min": data_stats.data_range[0].tolist(),
                "data_range_max": data_stats.data_range[1].tolist(),
            }
        )

        logger.logger.info(
            f"Successfully initialized client {client_state.client_id} (partition {partition_id}): "
            f"{client_state.num_samples} samples, {client_state.data_dimensions} dimensions"
        )

        return Message(content=RecordDict(response_content), reply_to=msg)

    except Exception as e:
        # Log error using emergency logger
        _emergency_logger(f"Client initialization failed: {e}", context)

        # Create error response
        response_content = create_message_content(metrics={"success": False, "error_message": str(e)})

        return Message(content=RecordDict(response_content), reply_to=msg)


@app.query("compute_local_statistics")
def compute_local_statistics(msg: Message, context: Context) -> Message:
    """Compute local statistics for federated preprocessing.

    This handler computes local statistics (sum, sum of squares, sample count)
    needed for calculating global mean and variance across all clients.
    For CLEF dataset, only continuous columns are included in statistics.

    """
    try:
        partition_id = context.node_config.get("partition-id")
        client_state = _restore_client_state(context, partition_id, context.node_id)
        # Initialize logger with explicit log file path
        logger = _get_logger(client_state, context.node_id)
        if not client_state or not client_state.is_initialized:
            raise ValueError(f"Client state not found or not initialized for partition {partition_id}")

        if client_state.local_data is None:
            raise ValueError("Client has no local data")

        # Check if this is CLEF dataset (ALS data format)
        data_format = client_state.data_source
        if data_format == "ALS":
            # Use CLEF-specific preprocessing that only normalizes continuous columns
            continuous_indices = client_state.data_info["continuous_indices"]
            
            if continuous_indices is None:
                raise ValueError("Continuous indices not available for CLEF dataset")
            
            local_stats = calculate_clef_local_statistics(client_state.local_data, continuous_indices)
            logger.logger.info(f"Computed CLEF local statistics: {local_stats['n_samples']} samples, "
                             f"{local_stats['n_continuous_features']} continuous features")
        else:
            # Use generic preprocessing for other datasets
            local_stats = calculate_local_statistics(client_state.local_data)
            logger.logger.info(f"Computed local statistics: {local_stats['n_samples']} samples")

        metrics={
            "success": True,
            "n_samples": local_stats["n_samples"],
            "sum": local_stats["sum"],
            "sum_sq": local_stats["sum_sq"],
            "n_continuous_features": local_stats.get("n_continuous_features", client_state.data_dimensions),
        }

        if local_stats.get("continuous_indices"):
            metrics["continuous_indices"] = local_stats["continuous_indices"]

        response_content = create_message_content(
            metrics=metrics
        )

        return Message(content=RecordDict(response_content), reply_to=msg)

    except Exception as e:
        _emergency_logger(f"Failed to compute local statistics: {e}", context)

        # Create error response
        response_content = create_message_content(metrics={"success": False, "error_message": str(e)})

        return Message(content=RecordDict(response_content), reply_to=msg)


@app.query("normalize_data")
def normalize_client_data(msg: Message, context: Context) -> Message:
    """Normalize client data using global mean and standard deviation.

    This handler receives global statistics from the server and normalizes
    the local data. For CLEF dataset, only continuous columns are normalized,
    leaving one-hot encoded categorical columns as 0/1 values.

    """
    try:
        # Extract message content
        content = extract_message_content(msg.content)
        arrays = content.get("arrays", {})
        config = content.get("config", {})
        partition_id = context.node_config.get("partition-id")

        # Restore client state from context
        client_state = _restore_client_state(context, partition_id, context.node_id)

        # Initialize logger with explicit log file path
        logger = _get_logger(client_state, context.node_id)
        if not client_state or not client_state.is_initialized:
            raise ValueError(f"Client state not found or not initialized for partition {partition_id}")

        if client_state.local_data is None:
            raise ValueError("Client has no local data")

        # Extract global statistics
        if "global_mean" not in arrays or "global_std" not in arrays:
            raise ValueError("global_mean or global_std not found in message")

        global_mean = arrays["global_mean"]
        global_std = arrays["global_std"]

        # Check if this is CLEF dataset (ALS data format)
        data_format = client_state.data_source
        
        if data_format == "ALS":
            # Use CLEF-specific normalization that only normalizes continuous columns
            continuous_indices = client_state.data_info["continuous_indices"]
            if continuous_indices is None:
                raise ValueError("Continuous indices not available for CLEF dataset")
            
            # Validate dimensions for continuous features only
            if global_mean.shape[0] != len(continuous_indices):
                raise ValueError(
                    f"Dimension mismatch: global_mean has {global_mean.shape[0]} dimensions, "
                    f"expected {len(continuous_indices)} continuous features"
                )
            
            # Normalize only continuous columns
            client_state.local_data = normalize_clef_data(
                client_state.local_data, global_mean, global_std, continuous_indices
            )
            
            logger.logger.info(
                f"Normalized CLEF data: {client_state.num_samples} samples, "
                f"{len(continuous_indices)} continuous columns normalized, "
                f"{client_state.data_dimensions - len(continuous_indices)} one-hot columns kept as 0/1"
            )
        else:
            # Use generic normalization for other datasets
            # Validate dimensions
            if global_mean.shape[0] != client_state.data_dimensions:
                raise ValueError(
                    f"Dimension mismatch: global_mean has {global_mean.shape[0]} dimensions, "
                    f"client data has {client_state.data_dimensions} dimensions"
                )

            # Normalize all data
            client_state.local_data = (client_state.local_data - global_mean) / global_std
            
            logger.logger.info(
                f"Normalized data: {client_state.num_samples} samples, " 
                f"{client_state.data_dimensions} dimensions"
            )

        # Persist updated client state
        _persist_client_state(context, client_state)

        # Create response message
        response_content = create_message_content(
            metrics={
                "success": True,
                "normalized_samples": client_state.num_samples,
                "normalized_dimensions": client_state.data_dimensions,
            }
        )

        return Message(content=RecordDict(response_content), reply_to=msg)

    except Exception as e:
        _emergency_logger(f"Failed to normalize data: {e}", context)
        response_content = create_message_content(metrics={"success": False, "error_message": str(e)})
        return Message(content=RecordDict(response_content), reply_to=msg)


def _validate_clustering_compatibility(client_state: ClientState, global_means: np.ndarray) -> None:
    """Validate that global means are compatible with client data.

    Args:
        client_state: Current client state
        global_means: Global cluster means to validate

    Raises:
        ValueError: If compatibility check fails
    """
    if client_state.local_data is None:
        raise ValueError("Client has no local data")

    # Validate global_means array properties
    if not isinstance(global_means, np.ndarray):
        raise ValueError("global_means must be a numpy array")

    if len(global_means.shape) != 2:
        raise ValueError(f"global_means must be 2D (k_global, n_features), got shape {global_means.shape}")

    if global_means.shape[0] == 0:
        raise ValueError("global_means cannot be empty")

    if np.any(np.isnan(global_means)) or np.any(np.isinf(global_means)):
        raise ValueError("global_means contains NaN or infinite values")

    # Check dimension compatibility
    if global_means.shape[1] != client_state.data_dimensions:
        raise ValueError(
            f"Dimension mismatch: global means have {global_means.shape[1]} dimensions, "
            f"client data has {client_state.data_dimensions} dimensions"
        )

    logging.getLogger(__name__).debug(
        f"Validated global means: {global_means.shape[0]} clusters, {global_means.shape[1]} dimensions"
    )


def _apply_privacy_filtering(
    local_means: np.ndarray, sample_counts: np.ndarray, privacy_threshold: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply privacy filtering to remove clusters below threshold.

    Args:
        local_means: Local cluster means, shape (n_clusters, n_features)
        sample_counts: Sample counts per cluster, shape (n_clusters,)
        privacy_threshold: Minimum samples required per cluster

    Returns:
        Tuple of (filtered_means, filtered_counts) with small clusters removed
    """
    if len(local_means) == 0 or len(sample_counts) == 0:
        return local_means, sample_counts

    # Find clusters that meet privacy threshold
    valid_mask = sample_counts >= privacy_threshold

    if not np.any(valid_mask):
        logging.getLogger(__name__).warning("No clusters meet privacy threshold, returning empty results")
        return np.array([]).reshape(0, local_means.shape[1]), np.array([])

    filtered_means = local_means[valid_mask]
    filtered_counts = sample_counts[valid_mask]

    removed_clusters = len(local_means) - len(filtered_means)
    if removed_clusters > 0:
        logging.getLogger(__name__).info(
            f"Privacy filtering removed {removed_clusters} clusters " f"(below threshold {privacy_threshold})"
        )

    return filtered_means, filtered_counts


@app.query("perform_local_clustering")
def perform_local_clustering(msg: Message, context: Context) -> Message:
    """Perform local k-means clustering using global cluster means.

    This handler receives global cluster means from the server,
    performs local clustering operations with privacy filtering,
    and returns serialized local results.

    """
    try:
        # Extract clustering message
        content = extract_message_content(msg.content)
        arrays = content.get("arrays", {})
        config = content.get("config", {})

        # Get partition_id from node_config
        partition_id = context.node_config.get("partition-id")

        # Restore client state from context
        client_state = _restore_client_state(context, partition_id, context.node_id)
        if not client_state or not client_state.is_initialized:
            raise ValueError(f"Client state not found or not initialized for partition {partition_id}")

        # Initialize logger with explicit log file path
        logger = _get_logger(client_state, context.node_id)

        # Deserialize global cluster means
        global_means = arrays["global_means"]
        round_number = config.get("round_number", 0)

        # Validate compatibility
        _validate_clustering_compatibility(client_state, global_means)

        # Perform complete local clustering workflow
        local_means, sample_counts, clustering_success, clustering_error = perform_complete_local_clustering(
            local_data=client_state.local_data,
            global_means=global_means,
            privacy_threshold=client_state.privacy_threshold,
            local_iterations=client_state.local_kmeans_iterations,
        )

        if not clustering_success:
            raise ValueError(f"Local clustering failed: {clustering_error}")

        # Apply privacy filtering to results
        filtered_means, filtered_counts = _apply_privacy_filtering(
            local_means, sample_counts, client_state.privacy_threshold
        )

        # Log privacy filtering info
        if len(filtered_means) != len(local_means):
            logger.logger.info(
                f"Privacy filtering removed {len(local_means) - len(filtered_means)} clusters "
                f"(below threshold {client_state.privacy_threshold})"
            )

        # Update and persist client state with current means
        client_state.current_means = filtered_means.copy() if len(filtered_means) > 0 else None
        _persist_client_state(context, client_state)

        # Create response message
        response_content = create_message_content(
            arrays={"local_means": filtered_means, "sample_counts": filtered_counts},
            metrics={
                "success": True,
                "num_active_clusters": len(filtered_means),
                "round_number": round_number,
                "original_clusters": len(local_means),
                "filtered_clusters": len(filtered_means),
            },
        )

        logger.logger.info(
            f"Completed local clustering round {round_number}: "
            f"{len(filtered_means)} clusters after privacy filtering"
        )

        return Message(content=RecordDict(response_content), reply_to=msg)

    except Exception as e:
        _emergency_logger(f"Local clustering failed: {e}", context)

        # Get round number from config if available
        round_num = 0
        if "config" in locals() and config is not None:
            round_num = config.get("round_number", 0)

        response_content = create_message_content(
            metrics={"success": False, "error_message": str(e), "num_active_clusters": 0, "round_number": round_num}
        )
        return Message(content=RecordDict(response_content), reply_to=msg)


def _validate_evaluation_inputs(client_state: ClientState, final_means: np.ndarray) -> None:
    """Validate inputs for evaluation computation.

    Args:
        client_state: Current client state
        final_means: Final global cluster means

    Raises:
        ValueError: If validation fails
    """
    if not client_state.is_initialized:
        raise ValueError("Client not initialized")

    if len(final_means) == 0:
        raise ValueError("Final cluster means are empty")

    # Check dimension compatibility
    if final_means.shape[1] != client_state.data_dimensions:
        raise ValueError(
            f"Dimension mismatch: final means have {final_means.shape[1]} dimensions, "
            f"client data has {client_state.data_dimensions} dimensions"
        )

    # Validate final means
    if np.any(np.isnan(final_means)) or np.any(np.isinf(final_means)):
        raise ValueError("Final cluster means contain NaN or infinite values")


def _compute_evaluation_metrics(client_state: ClientState, final_means: np.ndarray) -> dict:
    """Compute comprehensive evaluation metrics for local clustering quality.

    Args:
        client_state: Current client state with local data
        final_means: Final global cluster means

    Returns:
        Dictionary containing computed evaluation metrics

    Raises:
        ValueError: If metric computation fails
    """
    try:
        # Compute cluster assignments using final means
        cluster_assignments, distances = assign_to_global_clusters(client_state.local_data, final_means)

        # Compute comprehensive metrics for additional insights
        comprehensive_metrics = compute_comprehensive_local_metrics(
            data=client_state.local_data,
            cluster_labels=cluster_assignments,
            cluster_centers=final_means,
            distances=distances,
            true_labels=getattr(client_state, "ground_truth_labels", None),
            privacy_threshold=client_state.privacy_threshold,
        )

        metrics = {
            "adjusted_rand_score": comprehensive_metrics.get("adjusted_rand_score"),
            "silhouette_score": comprehensive_metrics.get("silhouette_score"),
            "local_inertia": comprehensive_metrics.get("inertia"),
            "num_clusters_used": comprehensive_metrics.get("cluster_statistics").get("num_clusters"),
            "total_clusters_available": len(final_means),
            "cluster_statistics": comprehensive_metrics.get("cluster_statistics", {}),
            "privacy_valid": comprehensive_metrics.get("privacy_valid", True),
        }

        ari_value = metrics.get('adjusted_rand_score')
        ari_str = f"{ari_value:.4f}" if ari_value is not None else "N/A"
        logging.getLogger(__name__).info(
            f"Computed evaluation metrics: ARI={ari_str}, "
            f"silhouette={metrics.get('silhouette_score'):.4f}, inertia={metrics.get('local_inertia'):.4f}"
        )

        return metrics

    except Exception as e:
        logging.getLogger(__name__).error(f"Error computing evaluation metrics: {e}")
        raise ValueError(f"Failed to compute evaluation metrics: {e}")


@app.query("compute_local_covariance")
def compute_local_covariance(msg: Message, context: Context) -> Message:
    """Compute local covariance statistics for federated PCA.

    This handler computes local covariance statistics needed for federated PCA
    without sharing raw data.
    """
    try:
        # Get partition_id from node_config
        partition_id = context.node_config.get("partition-id")

        # Restore client state
        client_state = _restore_client_state(context, partition_id, context.node_id)
        if not client_state or not client_state.is_initialized:
            raise ValueError(f"Client state not found or not initialized for partition {partition_id}")
        logger = _get_logger(client_state, context.node_id)

        if client_state.local_data is None:
            raise ValueError("Client has no local data")

        # Calculate local covariance statistics
        local_cov_stats = calculate_local_covariance_stats(client_state.local_data)

        # Create response message - send covariance as array, not in metrics
        response_content = create_message_content(
            arrays={"mean": np.array(local_cov_stats["mean"]), "cov": np.array(local_cov_stats["cov"])},
            metrics={"success": True, "n_samples": local_cov_stats["n_samples"]},
        )

        logger.logger.info(f"Computed local covariance: {local_cov_stats['n_samples']} samples")

        return Message(content=RecordDict(response_content), reply_to=msg)

    except Exception as e:
        _emergency_logger(f"Failed to compute local covariance: {e}", context)
        response_content = create_message_content(metrics={"success": False, "error_message": str(e)})
        return Message(content=RecordDict(response_content), reply_to=msg)


@app.query("project_to_pca")
def project_client_data_to_pca(msg: Message, context: Context) -> Message:
    """Project client data to PCA space for visualization.

    This handler receives PCA components from the server and projects
    local data onto the principal components for 2D visualization.
    """
    try:
        # Extract message content
        content = extract_message_content(msg.content)
        config = content.get("config", {})
        arrays = content.get("arrays", {})

        # Get partition_id from node_config
        partition_id = context.node_config.get("partition-id")

        # Restore client state
        client_state = _restore_client_state(context, partition_id, context.node_id)
        if not client_state or not client_state.is_initialized:
            raise ValueError(f"Client state not found or not initialized for partition {partition_id}")

        # Initialize logger with explicit log file path
        logger = _get_logger(client_state, context.node_id)

        pca_mean = arrays["pca_mean"]
        principal_components = arrays["principal_components"]

        # Validate dimensions
        if pca_mean.shape[0] != client_state.data_dimensions:
            raise ValueError(
                f"Dimension mismatch: pca_mean has {pca_mean.shape[0]} dimensions, "
                f"client data has {client_state.data_dimensions} dimensions"
            )

        # Project data to PCA space
        projected_data = project_to_pca(client_state.local_data, pca_mean, principal_components)

        # Store projected data for visualization, save to experiment directory
        experiment_dir = config.get("experiment_dir")
        if experiment_dir:
            pca_data_dir = Path(experiment_dir) / "pca_data"
            pca_data_dir.mkdir(exist_ok=True)

            # Save projected data
            np.save(pca_data_dir / f"client_{client_state.client_id}_projected.npy", projected_data)

            # Create and save visualization of projected data
            _create_pca_visualization(
                client_state, projected_data, pca_data_dir, logger
            )

            logger.logger.info(f"Saved projected data and visualization to {pca_data_dir}")

        # Create response message
        response_content = create_message_content(
            metrics={
                "success": True,
                "projected_samples": projected_data.shape[0],
                "projected_dimensions": projected_data.shape[1],
            }
        )

        logger.logger.info(f"Projected data to PCA space: {projected_data.shape}")

        return Message(content=RecordDict(response_content), reply_to=msg)

    except Exception as e:
        _emergency_logger(f"Failed to project to PCA: {e}", context)
        response_content = create_message_content(metrics={"success": False, "error_message": str(e)})
        return Message(content=RecordDict(response_content), reply_to=msg)


@app.query("evaluate_convergence")
def evaluate_convergence(msg: Message, context: Context) -> Message:
    """Evaluate clustering quality using final global cluster means.

    This handler computes local clustering quality metrics including
    adjusted rand score, silhouette score, and local inertia for
    evaluation of the federated k-means algorithm performance.
    """
    try:
        # Extract evaluation message
        content = extract_message_content(msg.content)
        arrays = content.get("arrays", {})

        # Get partition_id from node_config
        partition_id = context.node_config.get("partition-id")

        # Restore client state from context
        client_state = _restore_client_state(context, partition_id, context.node_id)
        if not client_state or not client_state.is_initialized:
            raise ValueError(f"Client state not found or not initialized for partition {partition_id}")

        # Initialize logger with explicit log file path
        logger = _get_logger(client_state, context.node_id)

        if "final_means" not in arrays:
            raise ValueError("final_means not found in message arrays")

        final_means = arrays["final_means"]

        # Validate inputs
        _validate_evaluation_inputs(client_state, final_means)

        # Compute comprehensive evaluation metrics
        evaluation_metrics = _compute_evaluation_metrics(client_state, final_means)

        # Create response message with comprehensive metrics
        response_metrics = {
            "success": True,
            "adjusted_rand_score": evaluation_metrics["adjusted_rand_score"] or 0.0,
            "silhouette_score": evaluation_metrics["silhouette_score"],
            "local_inertia": evaluation_metrics["local_inertia"],
            "num_clusters_used": evaluation_metrics["num_clusters_used"],
            "total_clusters_available": evaluation_metrics["total_clusters_available"],
            "privacy_valid": evaluation_metrics["privacy_valid"],
            "has_ground_truth": evaluation_metrics["adjusted_rand_score"] is not None,
        }

        # Add cluster statistics if available
        cluster_stats = evaluation_metrics.get("cluster_statistics", {})
        if cluster_stats:
            response_metrics.update(
                {
                    "num_local_clusters": cluster_stats.get("num_clusters", 0),
                    "largest_cluster_size": cluster_stats.get("largest_cluster_size", 0),
                    "smallest_cluster_size": cluster_stats.get("smallest_cluster_size", 0),
                    "total_samples_evaluated": cluster_stats.get("total_samples", 0),
                }
            )

        response_content = create_message_content(metrics=response_metrics)

        logger.logger.info(
            f"Successfully evaluated clustering quality: "
            f"silhouette={evaluation_metrics['silhouette_score']:.4f}, "
            f"inertia={evaluation_metrics['local_inertia']:.4f}, "
            f"clusters_used={evaluation_metrics['num_clusters_used']}"
        )

        return Message(content=RecordDict(response_content), reply_to=msg)

    except Exception as e:
        _emergency_logger(f"Evaluation failed: {e}", context)
        response_content = create_message_content(
            metrics={
                "success": False,
                "error_message": str(e),
                "adjusted_rand_score": 0.0,
                "silhouette_score": 0.0,
                "local_inertia": 0.0,
            }
        )
        return Message(content=RecordDict(response_content), reply_to=msg)


def _create_pca_visualization(client_state: ClientState, projected_data: np.ndarray, 
                             output_dir: Path, logger) -> None:
    """
    Create and save a visualization of the PCA-projected client data.
    
    This helps debug normalization issues by showing exactly what data
    is being projected and whether one-hot encoded columns are properly preserved.
    """
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: Original data (first 2 dimensions)
        if client_state.local_data.shape[1] >= 2:
            axes[0].scatter(client_state.local_data[:, 0], client_state.local_data[:, 1], 
                          alpha=0.6, s=20, c='blue')
            axes[0].set_title(f'Original Data (First 2 Dims)\nClient {client_state.client_id}')
            axes[0].set_xlabel('Feature 0')
            axes[0].set_ylabel('Feature 1')
            axes[0].grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Shape: {client_state.local_data.shape}\n'
            stats_text += f'Mean: [{client_state.local_data[:, 0].mean():.3f}, {client_state.local_data[:, 1].mean():.3f}]\n'
            stats_text += f'Std: [{client_state.local_data[:, 0].std():.3f}, {client_state.local_data[:, 1].std():.3f}]'
            axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: PCA-projected data
        axes[1].scatter(projected_data[:, 0], projected_data[:, 1], 
                       alpha=0.6, s=20, c='red')
        axes[1].set_title(f'PCA Projected Data\nClient {client_state.client_id}')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].grid(True, alpha=0.3)
        
        # Add PCA statistics
        pca_stats_text = f'Shape: {projected_data.shape}\n'
        pca_stats_text += f'Mean: [{projected_data[:, 0].mean():.3f}, {projected_data[:, 1].mean():.3f}]\n'
        pca_stats_text += f'Std: [{projected_data[:, 0].std():.3f}, {projected_data[:, 1].std():.3f}]'
        axes[1].text(0.02, 0.98, pca_stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the visualization
        viz_file = output_dir / f"client_{client_state.client_id}_pca_analysis.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.logger.info(f"Created PCA visualization: {viz_file}")
        
        # Also save detailed analysis as text
        analysis_file = output_dir / f"client_{client_state.client_id}_analysis.txt"
        with open(analysis_file, 'w') as f:
            f.write(f"Client {client_state.client_id} Data Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original data shape: {client_state.local_data.shape}\n")
            f.write(f"Projected data shape: {projected_data.shape}\n\n")
            
            f.write("Original data statistics (first 5 columns):\n")
            for i in range(min(5, client_state.local_data.shape[1])):
                col_data = client_state.local_data[:, i]
                f.write(f"  Column {i}: mean={col_data.mean():.6f}, std={col_data.std():.6f}, "
                       f"min={col_data.min():.6f}, max={col_data.max():.6f}\n")
                unique_vals = np.unique(col_data)
                if len(unique_vals) <= 5:
                    f.write(f"    Unique values: {unique_vals}\n")
                else:
                    f.write(f"    Unique values count: {len(unique_vals)}\n")
            
            f.write(f"\nPCA projected data statistics:\n")
            f.write(f"  PC1: mean={projected_data[:, 0].mean():.6f}, std={projected_data[:, 0].std():.6f}\n")
            f.write(f"  PC2: mean={projected_data[:, 1].mean():.6f}, std={projected_data[:, 1].std():.6f}\n")
            
        logger.logger.info(f"Created detailed analysis: {analysis_file}")
        
    except Exception as e:
        logger.logger.error(f"Failed to create PCA visualization: {e}")
        import traceback
        traceback.print_exc()

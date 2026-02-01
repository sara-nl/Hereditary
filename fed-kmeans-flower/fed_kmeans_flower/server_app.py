"""Federated K-means server application using Flower messaging API."""

import json
import time
from typing import Optional, Tuple

import numpy as np
from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fed_kmeans_flower.client_communicator import (ClientCommunicator,
                                                   RetryConfig)
from fed_kmeans_flower.client_manager import ClientManager
from fed_kmeans_flower.clustering.global_aggregator import \
    aggregate_local_cluster_means
from fed_kmeans_flower.config import load_config_from_run_config
from fed_kmeans_flower.data.clef_preprocessing import aggregate_clef_statistics
from fed_kmeans_flower.logging_utils import log_system_info, setup_logging
from fed_kmeans_flower.means_logger import MeansLogger
from fed_kmeans_flower.models import (EvaluationResult, PreprocessingResult,
                                      ServerState)
from fed_kmeans_flower.preprocessing import (
    aggregate_covariance_and_compute_pca, aggregate_statistics)

# Global logging instances
main_logger = None
means_logger = None

# Create ServerApp
app = ServerApp()

# Global server state
server_state = None

# Global client manager
client_manager = None

# Global client communicator
client_communicator = None


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the federated k-means server."""
    global server_state, client_manager, main_logger, means_logger, client_communicator

    try:
        # Initialize means logger with timestamped experiment directory
        log_dir = context.run_config.get("log-dir", "./logs")
        means_logger = MeansLogger(base_log_dir=log_dir)

        # Initialize logging and monitoring with paths from means_logger
        main_logger = setup_logging(
            log_level=context.run_config.get("log-level", "INFO"),
            log_file=str(means_logger.get_server_log_path()),
            enable_debug=context.run_config.get("enable-debug", False),
        )

        log_system_info(main_logger)
        config = load_config_from_run_config(context.run_config)
        main_logger.log_config(config)

        # Initialize server state and client manager
        server_state = ServerState(
            global_means=None,
            current_round=0,
            active_clients=[],
            convergence_history=[],
            is_converged=False,
            config=config,
            data_dimensions=0,
            pca_mean=None,
            principal_components=None,
            explained_variance=None,
        )

        client_manager = ClientManager(config)
        data_path = context.run_config.get("data-path", "./data")
        data_source = context.run_config.get("data-source", "directory")

        # Initialize ClientCommunicator with RetryConfig
        retry_config = RetryConfig.from_federated_config(config)
        client_communicator = ClientCommunicator(
            grid=grid,
            server_state=server_state,
            client_manager=client_manager,
            logger=main_logger,
            means_logger=means_logger,
            retry_config=retry_config,
            data_path=data_path,
            data_source=data_source,
        )
        main_logger.logger.info(
            f"ClientCommunicator initialized with timeout={retry_config.timeout}s, max_retries={retry_config.max_retries}"
        )
        main_logger.logger.info(f"Starting federated k-means with {config.k_global} clusters")
        main_logger.logger.info(f"Waiting for {config.required_clients} clients to connect...")

        # Wait for required clients to connect
        if not node_online_loop(grid, context):
            main_logger.logger.error("Failed to get required number of clients")
            return

        # Initialize clients
        if not initialize_clients(grid, context):
            main_logger.logger.error("Failed to initialize clients")
            return

        # Perform preprocessing: calculate global mean and variance
        if not perform_preprocessing(grid, context):
            main_logger.logger.error("Failed to perform preprocessing")
            return

        # Perform federated PCA for visualization (if data has > 2 dimensions)
        if server_state.data_dimensions > 2:
            if not perform_federated_pca(grid, context):
                main_logger.logger.warning("Failed to perform federated PCA - visualization may be limited")
                # Don't fail the entire process, just log warning

        # Run federated clustering rounds
        run_clustering_rounds(grid, context)

        # Final evaluation
        evaluate_model(grid, context, is_final=True)

        # Export performance metrics to experiment directory
        export_performance_metrics(context, config)

    except Exception as e:
        main_logger.logger.error(f"Error in main server execution: {str(e)}")

def export_performance_metrics(context: Context, config):
    global server_state, client_manager, main_logger, means_logger, client_communicator    
    try:
        metrics_file = means_logger.experiment_dir / "federated_kmeans_metrics.json"

        main_logger.export_metrics(str(metrics_file))

        # Build client ID to partition mapping
        partition_mappings = dict(client_manager.client_to_partition)

        # Save experiment summary
        means_logger.save_experiment_summary(
            {
                "config": {
                    "k_global": config.k_global,
                    "max_iterations": config.max_iterations,
                    "privacy_threshold": config.privacy_threshold,
                    "convergence_tolerance": config.convergence_tolerance,
                    "required_clients": config.required_clients,
                    "data_path": context.run_config.get("data-path", "./data"),
                    "data_source": context.run_config.get("data-source", "directory"),
                },
                "results": {
                    "total_rounds": server_state.current_round,
                    "converged": server_state.is_converged,
                    "final_global_means_shape": (
                        list(server_state.global_means.shape) if server_state.global_means is not None else None
                    ),
                    "convergence_history": server_state.convergence_history,
                    "data_dimensions": server_state.data_dimensions,
                    "used_pca": server_state.data_dimensions > 2,
                },
                "partition_mappings": partition_mappings,
            }
        )
    except Exception as e:
        main_logger.logger.warning(f"Failed to export metrics: {e}")

    main_logger.logger.info(f"Federated k-means clustering completed successfully!")
    main_logger.logger.info(f"Experiment logs saved to: {means_logger.experiment_dir}")

def node_online_loop(grid: Grid, context: Context) -> bool:
    """Wait for required number of clients to connect.

    Returns:
        bool: True if required clients connected, False if timeout or error
    """
    global server_state, client_manager, main_logger

    main_logger.logger.info(f"Waiting for {server_state.config.required_clients} clients to connect...")

    # Wait for clients to connect with timeout
    start_time = time.time()
    connection_timeout = 300.0  # 5 minutes timeout for client connections

    while not client_manager.has_minimum_clients():
        # Check for timeout
        if time.time() - start_time > connection_timeout:
            main_logger.logger.error(
                f"Timeout waiting for clients. Got {len(client_manager.get_active_clients())} of {server_state.config.required_clients} required"
            )
            return False

        # Discover available nodes
        try:
            available_nodes = grid.get_node_ids()
            main_logger.logger.debug(f"Available nodes: {available_nodes}")

            # Register new clients
            for node_id in available_nodes:
                if node_id not in client_manager.connected_clients:
                    client_manager.register_client(node_id)

            # Check for client timeouts
            timed_out_clients = client_manager.check_client_timeouts()
            if timed_out_clients:
                main_logger.logger.warning(f"Clients timed out: {timed_out_clients}")

            # Update server state with active clients
            server_state.active_clients = client_manager.get_active_clients()

            # Log status
            status = client_manager.get_status_summary()
            main_logger.logger.info(
                f"Client status: {status['active_clients']}/{server_state.config.required_clients} active clients"
            )

            # Short sleep to avoid busy waiting
            time.sleep(1.0)

        except Exception as e:
            main_logger.logger.error(f"Error in node discovery: {str(e)}")
            time.sleep(5.0)

    main_logger.logger.info(f"Required clients connected: {client_manager.get_active_clients()}")
    return True


def initialize_clients(grid: Grid, context: Context) -> bool:
    """Send initialization messages to all connected clients in parallel.

    Returns:
        bool: True if all clients initialized successfully, False otherwise
    """
    global server_state, client_manager

    main_logger.logger.info("Initializing clients...")

    active_clients = client_manager.get_active_clients()
    if not active_clients:
        main_logger.logger.error("No active clients to initialize")
        return False

    main_logger.logger.info(f"Broadcasting initialization messages to {len(active_clients)} clients")

    # Broadcast initialization messages using ClientCommunicator
    data_source = context.run_config.get("data-source") or context.run_config.get("data_source", "test_data")
    
    broadcast_result = client_communicator.broadcast_to_clients(
        client_ids=active_clients,
        message_type="query.initialize_client",
        config={
            "k_global": server_state.config.k_global,
            "max_iterations": server_state.config.max_iterations,
            "privacy_threshold": server_state.config.privacy_threshold,
            "convergence_tolerance": server_state.config.convergence_tolerance,
            "local_kmeans_iterations": server_state.config.local_kmeans_iterations,
            "data_path": context.run_config.get("data-path", "./data"),
            "data_source": data_source,
            "experiment_dir": str(means_logger.experiment_dir),
        },
        enable_recovery=True,
    )

    # Process all replies
    any_success = False

    for client_id, comm_result in broadcast_result.results.items():
        if not comm_result.success:
            main_logger.logger.error(f"No response from client {client_id}: {comm_result.error_message}")
            client_manager.mark_client_failed(
                client_id, f"No response during initialization: {comm_result.error_message}"
            )
            continue

        try:
            result_data = comm_result.extract_content()
            metrics = result_data.get("metrics", {})

            partition_id = metrics.get("partition_id")
            if partition_id is None:
                main_logger.logger.error(f"Client {client_id} did not provide partition_id")
                client_manager.mark_client_failed(client_id, "Missing partition_id in initialization response")
                continue

            success = metrics.get("success", False)
            data_dimensions = metrics.get("data_dimensions", 0)
            num_samples = metrics.get("num_samples", 0)

            if success:
                client_manager.mark_client_initialized(
                    client_id,
                    partition_id,
                    {"data_dimensions": data_dimensions, "num_samples": num_samples},
                )
                client_manager.update_client_activity(client_id)
                main_logger.logger.info(
                    f"Client {client_id} initialized with partition {partition_id}: {num_samples} samples, {data_dimensions} dimensions"
                )
                any_success = True
            else:
                error_message = metrics.get("error_message")
                main_logger.logger.error(
                    f"Client {client_id} (partition {partition_id}) initialization failed: {error_message}"
                )
                client_manager.mark_client_failed(client_id, f"Initialization failed: {error_message}")

        except Exception as e:
            main_logger.logger.error(f"Error processing initialization response from client {client_id}: {str(e)}")
            client_manager.mark_client_failed(client_id, f"Initialization error: {str(e)}")

    # Check if we have enough successfully initialized clients
    if not client_manager.has_required_clients():
        current_count = len(client_manager.get_initialized_clients())
        main_logger.logger.error(
            f"Insufficient initialized clients: {current_count} < {server_state.config.required_clients}"
        )
        return False

    initialized_clients = client_manager.get_initialized_clients()

    # Set expected number of partitions from config if any client succeeded
    if any_success:
        num_partitions = context.run_config.get("num-partitions")
        if num_partitions is not None:
            client_manager.set_expected_num_partitions(num_partitions)
            main_logger.logger.info(f"Expected number of partitions set to {num_partitions}")

    # Validate data compatibility across clients
    is_compatible, compatibility_error = client_manager.validate_data_compatibility()
    if not is_compatible:
        main_logger.logger.error(f"Data compatibility check failed: {compatibility_error}")
        return False

    # Update server state
    server_state.active_clients = initialized_clients

    # Capture data dimensions from first initialized client
    if client_manager.client_data_stats:
        first_client_stats = next(iter(client_manager.client_data_stats.values()))
        server_state.data_dimensions = first_client_stats.get("data_dimensions", 0)
        main_logger.logger.info(f"Data dimensions: {server_state.data_dimensions}")

    main_logger.logger.info(f"Successfully initialized {len(initialized_clients)} clients in parallel")
    main_logger.logger.info(f"Client data statistics: {client_manager.client_data_stats}")

    return True


def perform_federated_pca(grid: Grid, context: Context) -> bool:
    """Perform federated PCA for visualization of high-dimensional data.

    Returns:
        bool: True if PCA successful, False otherwise
    """
    global server_state, client_manager, main_logger

    main_logger.logger.info("Starting federated PCA computation...")
    main_logger.logger.info("Step 1: Collecting local covariance statistics from clients")

    if not client_manager.ensure_required_clients(main_logger):
        return False

    active_clients = client_manager.get_initialized_clients()

    # Step 1: Request local covariance statistics from all clients using ClientCommunicator
    broadcast_result = client_communicator.broadcast_to_clients(
        client_ids=active_clients, message_type="query.compute_local_covariance", enable_recovery=True
    )

    # Process replies
    local_cov_stats_list = []

    for client_id, comm_result in broadcast_result.results.items():
        if not comm_result.success:
            main_logger.logger.error(f"No response from client {client_id}: {comm_result.error_message}")
            continue

        try:
            result_data = comm_result.extract_content()
            metrics = result_data.get("metrics", {})
            arrays = result_data.get("arrays", {})

            if metrics.get("success", False):
                # Extract mean and cov from arrays
                mean_array = arrays.get("mean", np.array([]))
                cov_array = arrays.get("cov", np.array([]))

                local_cov_stats_list.append(
                    {"n_samples": metrics.get("n_samples", 0), "mean": mean_array.tolist(), "cov": cov_array.tolist()}
                )
                main_logger.logger.info(f"Client {client_id}: received covariance stats")
            else:
                main_logger.logger.error(
                    f"Client {client_id} covariance computation failed: {metrics.get('error_message')}"
                )

        except Exception as e:
            main_logger.logger.error(f"Error processing covariance response from client {client_id}: {str(e)}")

    if len(local_cov_stats_list) < len(active_clients) // 2:
        main_logger.logger.error(
            f"Insufficient covariance stats collected: {len(local_cov_stats_list)}/{len(active_clients)}"
        )
        return False

    # Step 2: Aggregate covariance and compute PCA
    main_logger.logger.info("Step 2: Aggregating covariance and computing PCA")

    try:
        pca_mean, principal_components, explained_variance = aggregate_covariance_and_compute_pca(
            local_cov_stats_list, n_components=2
        )

        # Store PCA components in server state
        server_state.pca_mean = pca_mean
        server_state.principal_components = principal_components
        server_state.explained_variance = explained_variance

        main_logger.logger.info(f"PCA computed: {principal_components.shape[1]} components")
        main_logger.logger.info(f"Explained variance ratio: {explained_variance / np.sum(explained_variance)}")

        # Save PCA components to experiment directory
        pca_dir = means_logger.experiment_dir / "pca"
        pca_dir.mkdir(exist_ok=True)

        np.save(pca_dir / "pca_mean.npy", pca_mean)
        np.save(pca_dir / "principal_components.npy", principal_components)
        np.save(pca_dir / "explained_variance.npy", explained_variance)

        # Save metadata
        pca_metadata = {
            "n_components": int(principal_components.shape[1]),
            "explained_variance": explained_variance.tolist(),
            "explained_variance_ratio": (explained_variance / np.sum(explained_variance)).tolist(),
            "total_variance": float(np.sum(explained_variance)),
            "data_dimensions": int(pca_mean.shape[0]),
        }

        with open(pca_dir / "pca_metadata.json", "w") as f:
            json.dump(pca_metadata, f, indent=2)

        main_logger.logger.info(f"PCA components saved to {pca_dir}")

    except Exception as e:
        main_logger.logger.error(f"Error computing PCA: {str(e)}")
        return False

    # Step 3: Send PCA components to clients for projection using ClientCommunicator
    main_logger.logger.info("Step 3: Broadcasting PCA components to clients")

    pca_broadcast_result = client_communicator.broadcast_to_clients(
        client_ids=active_clients,
        message_type="query.project_to_pca",
        arrays={"pca_mean": pca_mean, "principal_components": principal_components},
        config={"experiment_dir": str(means_logger.experiment_dir)},
        enable_recovery=True,
    )

    # Process projection replies
    projection_success_count = 0

    for client_id, comm_result in pca_broadcast_result.results.items():
        if not comm_result.success:
            main_logger.logger.error(f"No response from client {client_id}: {comm_result.error_message}")
            continue

        try:
            result_data = comm_result.extract_content()
            metrics = result_data.get("metrics", {})

            if metrics.get("success", False):
                projection_success_count += 1
                main_logger.logger.info(f"Client {client_id} projected data to PCA space")
            else:
                main_logger.logger.error(f"Client {client_id} PCA projection failed: {metrics.get('error_message')}")

        except Exception as e:
            main_logger.logger.error(f"Error processing PCA projection response from client {client_id}: {str(e)}")

    if projection_success_count < len(active_clients) // 2:
        main_logger.logger.error(f"Insufficient clients projected: {projection_success_count}/{len(active_clients)}")
        return False

    main_logger.logger.info(
        f"Federated PCA completed: {projection_success_count}/{len(active_clients)} clients projected"
    )
    return True


def perform_preprocessing(grid: Grid, context: Context) -> bool:
    """Perform federated preprocessing to calculate global mean and variance.

    Returns:
        bool: True if preprocessing successful, False otherwise
    """
    global server_state, client_manager, main_logger

    main_logger.logger.info("Starting federated preprocessing...")
    main_logger.logger.info("Step 1: Collecting local statistics from clients")

    if not client_manager.ensure_required_clients(main_logger):
        return False

    active_clients = client_manager.get_initialized_clients()

    # Step 1: Request local statistics from all clients using ClientCommunicator
    main_logger.logger.info(f"Broadcasting statistics request to {len(active_clients)} clients")

    broadcast_result = client_communicator.broadcast_to_clients(
        client_ids=active_clients, message_type="query.compute_local_statistics", enable_recovery=True
    )

    # Process all replies
    local_stats_list = []
    is_clef_dataset = False
    continuous_indices = None

    for client_id, comm_result in broadcast_result.results.items():
        if not comm_result.success:
            main_logger.logger.error(f"No response from client {client_id}: {comm_result.error_message}")
            continue

        try:
            result_data = comm_result.extract_content()
            metrics = result_data.get("metrics", {})

            preprocessing_result = PreprocessingResult(
                n_samples=metrics.get("n_samples", 0),
                sum=np.array(metrics.get("sum", [])),
                sum_sq=np.array(metrics.get("sum_sq", [])),
                success=metrics.get("success", False),
                error_message=metrics.get("error_message"),
            )

            if preprocessing_result.success:
                # Check if this is CLEF dataset based on presence of continuous_indices
                client_continuous_indices = metrics.get("continuous_indices")
                if client_continuous_indices:
                    is_clef_dataset = True
                    if continuous_indices is None:
                        continuous_indices = client_continuous_indices
                    elif continuous_indices != client_continuous_indices:
                        main_logger.logger.error(f"Client {client_id} has different continuous indices")
                        quit()

                local_stats_list.append(
                    {
                        "n_samples": preprocessing_result.n_samples,
                        "sum": preprocessing_result.sum.tolist(),
                        "sum_sq": preprocessing_result.sum_sq.tolist(),
                        "continuous_indices": client_continuous_indices,
                        "n_continuous_features": metrics.get("n_continuous_features", len(preprocessing_result.sum)),
                    }
                )
                main_logger.logger.info(f"Client {client_id}: {preprocessing_result.n_samples} samples")
            else:
                main_logger.logger.error(
                    f"Client {client_id} preprocessing failed: {preprocessing_result.error_message}"
                )

        except Exception as e:
            main_logger.logger.error(f"Error processing response from client {client_id}: {str(e)}")

    # Check if we have enough statistics
    if len(local_stats_list) < len(active_clients) // 2:
        main_logger.logger.error(f"Insufficient statistics collected: {len(local_stats_list)}/{len(active_clients)}")
        return False

    # Step 2: Aggregate statistics to compute global mean and std
    main_logger.logger.info("Step 2: Aggregating statistics to compute global mean and std")

    try:
        if is_clef_dataset:
            main_logger.logger.info("Using CLEF-specific preprocessing (continuous columns only)")
            global_mean, global_std, continuous_indices = aggregate_clef_statistics(local_stats_list)
        else:
            main_logger.logger.info("Using generic preprocessing (all columns)")
            global_mean, global_std = aggregate_statistics(local_stats_list)

        # Store in server state
        server_state.global_mean = global_mean
        server_state.global_std = global_std
        server_state.is_normalized = True
        server_state.continuous_indices = continuous_indices

        main_logger.logger.info(f"Global mean shape: {global_mean.shape}")
        main_logger.logger.info(f"Global std shape: {global_std.shape}")
        main_logger.logger.info(f"Global mean (first 5 dims): {global_mean[:5]}")
        main_logger.logger.info(f"Global std (first 5 dims): {global_std[:5]}")

    except Exception as e:
        main_logger.logger.error(f"Error aggregating statistics: {str(e)}")
        return False

    # Step 3: Send normalization parameters to all clients using ClientCommunicator
    main_logger.logger.info("Step 3: Broadcasting normalization parameters to clients")

    # Prepare config for CLEF dataset
    config = {}
    if is_clef_dataset and continuous_indices is not None:
        config["continuous_indices"] = continuous_indices

    norm_broadcast_result = client_communicator.broadcast_to_clients(
        client_ids=active_clients,
        message_type="query.normalize_data",
        arrays={"global_mean": global_mean, "global_std": global_std},
        config=config,
        enable_recovery=True,
    )

    # Process normalization replies
    normalization_success_count = 0

    for client_id, comm_result in norm_broadcast_result.results.items():
        if not comm_result.success:
            main_logger.logger.error(f"No response from client {client_id}: {comm_result.error_message}")
            continue

        try:
            result_data = comm_result.extract_content()
            metrics = result_data.get("metrics", {})

            if metrics.get("success", False):
                normalization_success_count += 1
                main_logger.logger.info(f"Client {client_id} normalized data successfully")
            else:
                main_logger.logger.error(f"Client {client_id} normalization failed: {metrics.get('error_message')}")

        except Exception as e:
            main_logger.logger.error(f"Error processing normalization response from client {client_id}: {str(e)}")
    # Check if enough clients normalized successfully
    if normalization_success_count < len(active_clients) // 2:
        main_logger.logger.error(
            f"Insufficient clients normalized: {normalization_success_count}/{len(active_clients)}"
        )
        return False

    main_logger.logger.info(
        f"Preprocessing completed successfully: {normalization_success_count}/{len(active_clients)} clients normalized"
    )
    return True


def run_clustering_rounds(grid: Grid, context: Context) -> None:
    """Execute federated k-means clustering rounds with fault tolerance."""
    global server_state, client_manager

    # Initialize global cluster means randomly based on client data dimensions
    data_dimensions = _get_data_dimensions_from_clients()
    if data_dimensions is None:
        main_logger.logger.error("Cannot determine data dimensions from clients")
        return

    # Initialize global means randomly
    np.random.seed(42)  # For reproducibility
    server_state.global_means = np.random.randn(server_state.config.k_global, data_dimensions)

    main_logger.logger.info(f"Starting {server_state.config.max_iterations} clustering rounds...")
    main_logger.logger.info(f"Initial global means shape: {server_state.global_means.shape}")

    # Log initial global means (round 0)
    means_logger.log_global_means(
        round_number=0,
        global_means=server_state.global_means,
        metadata={"initialization": "random", "seed": 42, "num_clients": len(client_manager.get_initialized_clients())},
    )

    for round_num in range(server_state.config.max_iterations):
        server_state.current_round = round_num + 1
        main_logger.logger.info(f"\n=== Round {server_state.current_round}/{server_state.config.max_iterations} ===")

        # Log current system status
        main_logger.logger.info(f"System status: {client_manager.get_status_summary()}")

        # Ensure we have required number of clients
        if not client_manager.ensure_required_clients(main_logger):
            main_logger.logger.error("Insufficient clients available, stopping clustering")
            break

        # Execute clustering round
        success, convergence_change = execute_clustering_round(grid, context)

        if not success:
            main_logger.logger.error(f"Round {server_state.current_round} failed")
            continue

        # Store convergence history
        server_state.convergence_history.append(convergence_change)

        main_logger.logger.info(
            f"Round {server_state.current_round} completed. Convergence change: {convergence_change:.6f}"
        )

        # Log global means after this round
        means_logger.log_global_means(
            round_number=server_state.current_round,
            global_means=server_state.global_means,
            metadata={
                "convergence_change": float(convergence_change),
                "is_converged": server_state.is_converged,
                "num_participating_clients": len(server_state.active_clients),
                "active_clients": server_state.active_clients,
            },
        )

        # Check convergence
        if server_state.is_converged:
            main_logger.logger.info(f"Algorithm converged after {server_state.current_round} rounds!")
            break

        # Evaluate clustering quality periodically
        if server_state.current_round % server_state.config.evaluation_frequency == 0:
            evaluate_model(grid, context, is_final=False)

    # Generate final fault tolerance report
    ft_report = client_manager.get_fault_tolerance_report()
    main_logger.logger.info(f"Clustering completed after {server_state.current_round} rounds")
    main_logger.logger.info(f"Final convergence history: {server_state.convergence_history}")
    main_logger.logger.info(f"Final fault tolerance report: {ft_report}")


def execute_clustering_round(grid: Grid, context: Context) -> Tuple[bool, float]:
    """Execute a single federated clustering round with fault tolerance.

    Returns:
        Tuple of (success, convergence_change)
    """
    global server_state, client_manager

    # Ensure we have required clients at round start
    if not client_manager.ensure_required_clients(main_logger):
        return False, float("inf")

    active_clients = client_manager.get_initialized_clients()
    main_logger.logger.info(f"Starting round with {len(active_clients)} clients")

    # Collect local clustering results using ClientCommunicator (handles recovery automatically)
    broadcast_result = client_communicator.broadcast_to_clients(
        client_ids=active_clients,
        message_type="query.perform_local_clustering",
        arrays={"global_means": server_state.global_means},
        config={"round_number": server_state.current_round},
        enable_recovery=True,
    )

    # Step 4: Check if enough clients successfully responded
    if not broadcast_result.has_minimum_successes(server_state.config.required_clients):
        main_logger.logger.error(
            f"Too many clients failed during round: "
            f"{broadcast_result.success_count} successful < {server_state.config.required_clients} required"
        )
        return False, float("inf")

    # Step 5: Validate results and check if we have enough for aggregation
    valid_client_ids = set()
    local_means_list = []
    sample_counts_list = []

    for client_id, comm_result in broadcast_result.results.items():
        if not comm_result.success:
            main_logger.logger.warning(f"Client {client_id} communication failed: {comm_result.error_message}")
            continue

        try:
            # Extract local clustering result from communication result
            result_data = comm_result.extract_content()
            arrays = result_data.get("arrays", {})
            metrics = result_data.get("metrics", {})

            # Extract individual fields directly
            local_means = arrays.get("local_means", np.array([]))
            sample_counts = arrays.get("sample_counts", np.array([]))
            num_active_clusters = metrics.get("num_active_clusters", 0)
            success = metrics.get("success", False)
            error_message = metrics.get("error_message")

            if success and local_means.size > 0:
                local_means_list.append(local_means)
                sample_counts_list.append(sample_counts)
                valid_client_ids.add(client_id)

                # Record successful participation
                client_manager.record_client_participation(client_id, server_state.current_round)

                # Log local means on server side
                means_logger.log_local_means(
                    round_number=server_state.current_round,
                    client_id=client_id,
                    local_means=local_means,
                    sample_counts=sample_counts,
                    metadata={
                        "num_active_clusters": num_active_clusters,
                        "success": success,
                    },
                )

                main_logger.logger.debug(
                    f"Client {client_id}: {len(local_means)} clusters, {num_active_clusters} active"
                )
            else:
                main_logger.logger.warning(
                    f"Client {client_id} returned invalid result: {error_message}"
                )

        except Exception as e:
            main_logger.logger.error(f"Error processing response from client {client_id}: {str(e)}")

    # Verify we still have enough valid results after validation
    if len(valid_client_ids) < server_state.config.required_clients:
        main_logger.logger.error(
            f"Insufficient valid results after validation: "
            f"{len(valid_client_ids)} < {server_state.config.required_clients}"
        )
        return False, float("inf")

    if not local_means_list:
        main_logger.logger.error("No valid local cluster means received from any client")
        return False, float("inf")

    # Step 6: Perform global aggregation
    main_logger.logger.info(f"Performing global aggregation with {len(local_means_list)} client results")

    previous_means = server_state.global_means.copy()

    global_means, aggregation_success, change_magnitude, is_converged, error_message = aggregate_local_cluster_means(
        local_means_list=local_means_list,
        sample_counts_list=sample_counts_list,
        k_global=server_state.config.k_global,
        weighted=server_state.config.weighted_aggregation,
        previous_global_means=previous_means,
        convergence_tolerance=server_state.config.convergence_tolerance,
        max_iterations=100,
        random_state=42,
    )

    if not aggregation_success:
        main_logger.logger.error(f"Global aggregation failed: {error_message}")
        return False, float("inf")

    # Step 7: Update server state
    server_state.global_means = global_means
    server_state.is_converged = is_converged

    # Update active clients list
    server_state.active_clients = list(valid_client_ids)

    main_logger.logger.info(f"Global aggregation completed. New means shape: {global_means.shape}")
    main_logger.logger.info(f"Convergence change: {change_magnitude:.6f}, converged: {is_converged}")
    main_logger.logger.info(f"Round completed with {len(valid_client_ids)} participating clients")

    # Log fault tolerance status
    ft_report = client_manager.get_fault_tolerance_report()
    main_logger.logger.debug(f"Fault tolerance report: {ft_report['system_health']}")

    return True, change_magnitude


def evaluate_model(grid: Grid, context: Context, is_final: bool = False) -> None:
    """Evaluate clustering model quality.
    
    Args:
        grid: Flower grid for communication
        context: Flower context
        is_final: If True, generates comprehensive final evaluation report with system reliability metrics.
                 If False, generates periodic evaluation summary for current round.
    """
    global server_state, client_manager

    # Log appropriate header based on evaluation type
    if is_final:
        main_logger.logger.info("\nEvaluating final clustering results...")
    else:
        main_logger.logger.info(f"Evaluating clustering quality at round {server_state.current_round}")

    # Ensure we have required clients (with warning for final evaluation)
    if not client_manager.ensure_required_clients(main_logger):
        if is_final:
            main_logger.logger.warning("Insufficient clients for final evaluation")
        return

    active_clients = client_manager.get_initialized_clients()
    final_means = server_state.global_means

    # Broadcast evaluation request to all clients
    broadcast_result = client_communicator.broadcast_to_clients(
        client_ids=active_clients,
        message_type="query.evaluate_convergence",
        arrays={"final_means": final_means},
        enable_recovery=True,
    )

    # Collect evaluation results from clients and aggregate with weighted averaging
    valid_results = []
    client_sample_counts = []

    for client_id, comm_result in broadcast_result.results.items():
        if not comm_result.success:
            error_msg = f"No response from client {client_id}: {comm_result.error_message}" if not is_final else f"Failed to evaluate client {client_id}: {comm_result.error_message}"
            main_logger.logger.error(error_msg)
            continue

        try:
            result_data = comm_result.extract_content()
            metrics = result_data.get("metrics", {})

            evaluation_result = EvaluationResult(
                adjusted_rand_score=metrics.get("adjusted_rand_score", 0.0),
                silhouette_score=metrics.get("silhouette_score", 0.0),
                local_inertia=metrics.get("local_inertia", 0.0),
                success=metrics.get("success", False),
                error_message=metrics.get("error_message"),
            )

            if evaluation_result.success:
                # Add to valid results and get sample count
                valid_results.append(evaluation_result)
                client_stats = client_manager.client_data_stats.get(client_id, {})
                num_samples = client_stats.get("num_samples", 0)
                client_sample_counts.append(num_samples)
                
                eval_type = "final evaluation" if is_final else "evaluation"
                main_logger.logger.info(
                    f"Client {client_id} {eval_type}: ARI={evaluation_result.adjusted_rand_score:.4f}, "
                    f"Silhouette={evaluation_result.silhouette_score:.4f}"
                )
            elif is_final:
                main_logger.logger.warning(f"Client {client_id} evaluation failed: {evaluation_result.error_message}")

        except Exception as e:
            main_logger.logger.error(f"Error processing evaluation from client {client_id}: {str(e)}")

    if valid_results:
        # Use weighted averaging based on number of samples per client
        total_samples = sum(client_sample_counts)
        
        weights = np.array(client_sample_counts) / total_samples
        avg_ari = np.average([r.adjusted_rand_score for r in valid_results], weights=weights)
        avg_silhouette = np.average([r.silhouette_score for r in valid_results], weights=weights)
        avg_inertia = np.average([r.local_inertia for r in valid_results], weights=weights)

        # Generate appropriate summary based on evaluation type
        if is_final:
            main_logger.logger.info("=== FINAL EVALUATION SUMMARY ===")
        else: 
            main_logger.logger.info("=== ROUND EVALUATION SUMMARY ===")
            main_logger.logger.info(f"Round {server_state.current_round} evaluation summary:")


        main_logger.logger.info(f"Total samples across all clients: {total_samples}")
        main_logger.logger.info(f"Average Adjusted Rand Index (weighted): {avg_ari:.4f}")
        main_logger.logger.info(f"Average Silhouette Score (weighted): {avg_silhouette:.4f}")
        main_logger.logger.info(f"Average Local Inertia (weighted): {avg_inertia:.4f}")
            
        if is_final:
            main_logger.logger.info(f"Clients evaluated: {len(valid_results)}/{len(active_clients)}")
            main_logger.logger.info(f"Final global means shape: {server_state.global_means.shape}")

            # Denormalize centroids if data was normalized
            if server_state.is_normalized and server_state.global_mean is not None and server_state.global_std is not None:
                if server_state.continuous_indices:
                    # Only denormalize continuous columns
                    denormalized_means = server_state.global_means.copy()
                    denormalized_means[:, server_state.continuous_indices] = (
                        server_state.global_means[:, server_state.continuous_indices] * server_state.global_std 
                        + server_state.global_mean
                    )
                else:
                    # Denormalize all columns
                    denormalized_means = server_state.global_means * server_state.global_std + server_state.global_mean
                
                main_logger.logger.info(
                    f"Denormalized global means (first centroid, first 5 dims): {denormalized_means[0, :5]}"
                )

            main_logger.logger.info(f"Total rounds completed: {server_state.current_round}")
            main_logger.logger.info(f"Algorithm converged: {server_state.is_converged}")

            # Generate comprehensive system report
            ft_report = client_manager.get_fault_tolerance_report()
            main_logger.logger.info("=== SYSTEM RELIABILITY REPORT ===")
            main_logger.logger.info(f"Total clients seen: {ft_report['total_clients_seen']}")
            main_logger.logger.info(f"Successfully completed: {ft_report['currently_active']}")


def _get_data_dimensions_from_clients() -> Optional[int]:
    """Get data dimensions from initialized clients."""
    global client_manager

    if not client_manager.client_data_stats:
        return None

    # Get dimensions from first initialized client
    for client_id, stats in client_manager.client_data_stats.items():
        if client_id in client_manager.initialized_clients:
            return stats.get("data_dimensions")

    return None

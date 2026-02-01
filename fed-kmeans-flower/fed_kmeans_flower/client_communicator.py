"""Client communication module with retry logic and partition recovery integration."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flwr.app import Message, RecordDict
from flwr.serverapp import Grid

from fed_kmeans_flower.client_manager import ClientManager
from fed_kmeans_flower.communication import (create_message_content,
                                             extract_message_content)
from fed_kmeans_flower.logging_utils import FederatedKMeansLogger
from fed_kmeans_flower.means_logger import MeansLogger
from fed_kmeans_flower.models import FederatedKMeansConfig, ServerState


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int  # Number of retries after initial attempt
    retry_delay: float  # Seconds between retries
    enable_recovery: bool  # Whether to attempt partition recovery
    timeout: float  # Default timeout in seconds
    recovery_timeout: float = 30.0  # Timeout for waiting for new nodes
    recovery_retry_delay: float = 5.0  # Seconds between discovery attempts

    @classmethod
    def from_federated_config(cls, config: FederatedKMeansConfig) -> "RetryConfig":
        """Create RetryConfig from FederatedKMeansConfig.

        Args:
            config: FederatedKMeansConfig instance

        Returns:
            RetryConfig with values from FederatedKMeansConfig
        """
        return cls(
            timeout=config.client_timeout,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            enable_recovery=config.enable_recovery,
            recovery_timeout=config.recovery_timeout,
            recovery_retry_delay=config.recovery_retry_delay
        )


@dataclass
class CommunicationResult:
    """Result of a single-client communication."""

    success: bool
    client_id: str
    response: Optional[Message]
    error_message: Optional[str]
    attempts: int  # Number of attempts made
    recovery_attempted: bool  # Whether partition recovery was attempted
    recovered_client_id: Optional[str]  # New client ID if recovery occurred

    def extract_content(self) -> Optional[Dict[str, Any]]:
        """Extract and deserialize message content.

        Returns:
            Extracted message content or None if no response
        """
        if self.response is None:
            return None
        return extract_message_content(self.response.content)


@dataclass
class BroadcastResult:
    """Result of a multi-client broadcast."""

    results: Dict[str, CommunicationResult]  # client_id -> result
    success_count: int
    failure_count: int
    recovery_count: int  # Number of clients recovered

    def get_successful_results(self) -> Dict[str, CommunicationResult]:
        """Get only successful communication results.

        Returns:
            Dictionary of successful results keyed by client_id
        """
        return {client_id: result for client_id, result in self.results.items() if result.success}

    def get_failed_clients(self) -> List[str]:
        """Get list of client IDs that failed.

        Returns:
            List of client IDs that failed communication
        """
        return [client_id for client_id, result in self.results.items() if not result.success]

    def has_minimum_successes(self, minimum: int) -> bool:
        """Check if minimum number of clients succeeded.

        Args:
            minimum: Minimum number of successful clients required

        Returns:
            True if success_count >= minimum
        """
        return self.success_count >= minimum


class ClientCommunicator:
    """Handles all client communication with retry, timeout, and recovery."""

    def __init__(
        self,
        grid: Grid,
        server_state: ServerState,
        client_manager: ClientManager,
        logger: FederatedKMeansLogger,
        means_logger: MeansLogger,
        retry_config: Optional[RetryConfig] = None,
        data_path: str = "./data",
        data_source: str = "directory",
    ):
        """Initialize the communicator with dependencies.

        Args:
            grid: Flower Grid instance for message transport
            server_state: Server state containing configuration
            client_manager: Client manager for state tracking
            logger: Main logger for federated k-means operations
            means_logger: Logger for tracking means
            retry_config: Optional retry configuration (uses defaults if None)
            data_path: Path to data directory or file
            data_source: Data source type ("directory" or "file")
        """
        self.grid = grid
        self.server_state = server_state
        self.client_manager = client_manager
        self.logger = logger
        self.means_logger = means_logger
        self.retry_config = retry_config or RetryConfig()
        self.data_path = data_path
        self.data_source = data_source

    def send_to_client(
        self,
        client_id: str,
        message_type: str,
        arrays: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        enable_recovery: bool = True,
    ) -> CommunicationResult:
        """Send a message to a single client with retry and recovery.

        Args:
            client_id: Target client ID
            message_type: Type of message (e.g., "query.initialize_client")
            arrays: Optional numpy arrays to send
            config: Optional configuration dictionary
            timeout: Optional timeout override
            enable_recovery: Whether to attempt partition recovery on failure

        Returns:
            CommunicationResult with response data or error information
        """
        actual_timeout = timeout if timeout is not None else self.retry_config.timeout

        # Call _retry_with_recovery with all parameters
        response, error_history, recovered_client_id = self._retry_with_recovery(
            client_id=client_id,
            message_type=message_type,
            arrays=arrays,
            config=config,
            timeout=actual_timeout,
            max_retries=self.retry_config.max_retries,
            enable_recovery=enable_recovery,
        )

        # Calculate number of attempts
        attempts = len(error_history) if error_history else 1

        # Determine if recovery was attempted
        recovery_attempted = recovered_client_id is not None

        # Construct CommunicationResult
        success = response is not None
        error_message = "; ".join(error_history) if error_history else None

        result = CommunicationResult(
            success=success,
            client_id=client_id,
            response=response,
            error_message=error_message,
            attempts=attempts,
            recovery_attempted=recovery_attempted,
            recovered_client_id=recovered_client_id,
        )

        # Log communication outcome at appropriate level with detailed information
        if success:
            if attempts > 1:
                self.logger.logger.info(
                    f"✓ Successfully communicated with client {client_id} ({message_type}) "
                    f"after {attempts} attempt(s){' with recovery' if recovery_attempted else ''}"
                )
            else:
                self.logger.logger.debug(
                    f"✓ Successfully communicated with client {client_id} ({message_type}) on first attempt"
                )

            # Update client activity in ClientManager on success
            self.client_manager.update_client_activity(client_id)

            # Log recovery details if applicable
            if recovery_attempted and recovered_client_id:
                self.logger.logger.info(f"🔄 Client recovered: {client_id} → {recovered_client_id}")
        else:
            if recovery_attempted:
                self.logger.logger.error(
                    f"✗ FAILED: client {client_id} ({message_type}) "
                    f"after {attempts} attempt(s) and recovery attempt. "
                    f"Error: {error_message}"
                )
            else:
                self.logger.logger.warning(
                    f"✗ Failed: client {client_id} ({message_type}) "
                    f"after {attempts} attempt(s). Error: {error_message}"
                )

        return result

    def broadcast_to_clients(
        self,
        client_ids: List[str],
        message_type: str,
        arrays: Optional[Dict[str, np.ndarray]] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        enable_recovery: bool = True,
    ) -> BroadcastResult:
        """Broadcast a message to multiple clients with parallel retry and recovery.

        Args:
            client_ids: List of target client IDs
            message_type: Type of message
            arrays: Optional numpy arrays to send
            config: Optional configuration dictionary
            timeout: Optional timeout override
            enable_recovery: Whether to attempt partition recovery on failures

        Returns:
            BroadcastResult with per-client results and summary
        """
        # Validate input parameters
        if not client_ids:
            self.logger.logger.error("client_ids list cannot be empty")
            return BroadcastResult(results={}, success_count=0, failure_count=0, recovery_count=0)

        if not message_type:
            self.logger.logger.error("message_type cannot be empty")
            return BroadcastResult(
                results={
                    client_id: CommunicationResult(
                        success=False,
                        client_id=client_id,
                        response=None,
                        error_message="message_type cannot be empty",
                        attempts=0,
                        recovery_attempted=False,
                        recovered_client_id=None,
                    )
                    for client_id in client_ids
                },
                success_count=0,
                failure_count=len(client_ids),
                recovery_count=0,
            )

        # Get timeout from parameter or RetryConfig default
        actual_timeout = timeout if timeout is not None else self.retry_config.timeout

        arrays = arrays or {}
        config = config or {}

        self.logger.logger.info(f"📡 Broadcasting {message_type} to {len(client_ids)} clients")

        # Track results for each client
        results: Dict[str, CommunicationResult] = {}

        # Track which clients still need to be contacted
        pending_clients = set(client_ids)

        # Track attempts per client
        client_attempts: Dict[str, int] = {client_id: 0 for client_id in client_ids}

        # Track error messages per client
        client_errors: Dict[str, List[str]] = {client_id: [] for client_id in client_ids}

        # Perform initial broadcast + retries
        for attempt in range(self.retry_config.max_retries + 1):
            if not pending_clients:
                self.logger.logger.debug(f"✓ All clients succeeded, stopping retry loop at attempt {attempt}")
                break  # All clients succeeded

            if attempt > 0:
                self.logger.logger.info(f"⟳ Retry {attempt}/{self.retry_config.max_retries} for {len(pending_clients)} clients")
                time.sleep(self.retry_config.retry_delay)

            # Send messages in parallel to pending clients
            responses_map = self._send_parallel_messages(
                list(pending_clients), message_type, arrays, config, actual_timeout
            )

            # Process responses and update tracking
            newly_succeeded = set()
            for client_id in pending_clients:
                client_attempts[client_id] += 1
                response, error_msg = responses_map.get(client_id, (None, "No response received"))

                if response is not None:
                    # Success - create result and mark as complete
                    results[client_id] = CommunicationResult(
                        success=True,
                        client_id=client_id,
                        response=response,
                        error_message=None,
                        attempts=client_attempts[client_id],
                        recovery_attempted=False,
                        recovered_client_id=None,
                    )
                    newly_succeeded.add(client_id)
                    self.client_manager.update_client_activity(client_id)
                else:
                    client_errors[client_id].append(f"Attempt {client_attempts[client_id]}: {error_msg}")

            pending_clients -= newly_succeeded

        # After all retries, attempt recovery for remaining failed clients
        recovery_count = 0
        if pending_clients and enable_recovery:
            self.logger.logger.info(f"🔄 Attempting partition recovery for {len(pending_clients)} failed clients")

            # Identify partitions for failed clients
            failed_partitions: Dict[int, str] = {}  # partition_id -> original_client_id
            for client_id in pending_clients:
                partition_id = self.client_manager.get_partition_id(client_id)
                if partition_id is not None:
                    failed_partitions[partition_id] = client_id
                    # Mark client as failed
                    self.client_manager.mark_client_failed(
                        client_id, f"Communication failed after {client_attempts[client_id]} attempts"
                    )
                    self.logger.logger.debug(f"Marked client {client_id} (partition {partition_id}) as failed")
                else:
                    self.logger.logger.warning(f"Cannot recover client {client_id}: no partition mapping")

            if failed_partitions:
                # Attempt recovery for all failed partitions
                recovered_clients = self._attempt_broadcast_recovery(
                    failed_partitions, message_type, arrays, config, actual_timeout
                )

                # Process recovery results
                for partition_id, (original_client_id, new_client_id, response, error_msg) in recovered_clients.items():
                    if response is not None:
                        # Recovery succeeded
                        results[original_client_id] = CommunicationResult(
                            success=True,
                            client_id=original_client_id,
                            response=response,
                            error_message=None,
                            attempts=client_attempts[original_client_id] + 1,
                            recovery_attempted=True,
                            recovered_client_id=new_client_id,
                        )
                        pending_clients.discard(original_client_id)
                        recovery_count += 1
                        self.logger.logger.info(
                            f"✓ Successfully recovered partition {partition_id}: "
                            f"{original_client_id} → {new_client_id}"
                        )
                        # Update client activity for new client
                        self.client_manager.update_client_activity(new_client_id)
                    else:
                        # Recovery failed
                        client_errors[original_client_id].append(f"Recovery failed: {error_msg}")
                        self.logger.logger.error(
                            f"✗ Failed to recover partition {partition_id} "
                            f"(original client {original_client_id}): {error_msg}"
                        )

        # Create results for any remaining failed clients
        for client_id in pending_clients:
            error_message = "; ".join(client_errors[client_id])
            results[client_id] = CommunicationResult(
                success=False,
                client_id=client_id,
                response=None,
                error_message=error_message,
                attempts=client_attempts[client_id],
                recovery_attempted=enable_recovery and self.client_manager.get_partition_id(client_id) is not None,
                recovered_client_id=None,
            )

        # Calculate summary statistics
        success_count = sum(1 for r in results.values() if r.success)
        failure_count = len(results) - success_count

        # Construct and return BroadcastResult
        broadcast_result = BroadcastResult(
            results=results, success_count=success_count, failure_count=failure_count, recovery_count=recovery_count
        )

        # Log broadcast summary with detailed statistics
        total_clients = len(results)
        success_rate = (success_count / total_clients * 100) if total_clients > 0 else 0

        if failure_count == 0:
            self.logger.logger.info(
                f"✓ Broadcast complete: {success_count}/{total_clients} succeeded (100%) - {message_type}"
            )
        elif success_count == 0:
            self.logger.logger.error(f"✗ Broadcast FAILED: 0/{total_clients} succeeded (0%) - {message_type}")
        else:
            log_level = self.logger.logger.warning if failure_count > success_count else self.logger.logger.info
            log_level(
                f"⚠ Broadcast partial: {success_count}/{total_clients} succeeded ({success_rate:.1f}%), "
                f"{failure_count} failed, {recovery_count} recovered - {message_type}"
            )

        return broadcast_result

    def _send_parallel_messages(
        self,
        client_ids: List[str],
        message_type: str,
        arrays: Dict[str, np.ndarray],
        config: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Tuple[Optional[Message], Optional[str]]]:
        """Send messages to multiple clients in parallel.

        Args:
            client_ids: List of target client IDs
            message_type: Type of message
            arrays: Dictionary of numpy arrays to send
            config: Configuration dictionary
            timeout: Timeout in seconds

        Returns:
            Dictionary mapping client_id to (response, error_message) tuple
        """
        from flwr.app import RecordDict

        from fed_kmeans_flower.communication import create_message_content

        # Create messages for all target clients
        messages = []
        message_to_client: Dict[int, str] = {}  # message index -> client_id

        for idx, client_id in enumerate(client_ids):
            try:
                # Create message content
                content = create_message_content(arrays=arrays, config=config)

                # Create message
                message = Message(
                    content=RecordDict(content),
                    dst_node_id=client_id,
                    message_type=message_type,
                    ttl=timeout,
                    group_id="",
                )

                messages.append(message)
                message_to_client[len(messages) - 1] = client_id

            except Exception as e:
                error_msg = f"Error creating message for client {client_id}: {str(e)}"
                self.logger.logger.error(error_msg)

        if not messages:
            self.logger.logger.error(f"Failed to create any messages for {len(client_ids)} clients")
            return {client_id: (None, "Failed to create message") for client_id in client_ids}

        try:
            responses = self.grid.send_and_receive(messages, timeout=timeout)
        except Exception as e:
            error_msg = f"Error during parallel transmission: {str(e)}"
            self.logger.logger.error(error_msg, exc_info=True)
            return {client_id: (None, error_msg) for client_id in client_ids}

        # Map responses back to client_ids
        results: Dict[str, Tuple[Optional[Message], Optional[str]]] = {}
        message_id_to_client: Dict[str, str] = {}
        
        for idx, message in enumerate(messages):
            if hasattr(message, "metadata") and hasattr(message.metadata, "message_id"):
                message_id_to_client[message.metadata.message_id] = message_to_client[idx]

        # Match responses to clients
        matched_clients = set()
        for response in responses:
            if response and hasattr(response, "metadata"):
                # Try reply_to_message_id first, then src_node_id
                client_id = None
                if hasattr(response.metadata, "reply_to_message_id"):
                    client_id = message_id_to_client.get(response.metadata.reply_to_message_id)
                if not client_id and hasattr(response.metadata, "src_node_id"):
                    if response.metadata.src_node_id in client_ids:
                        client_id = response.metadata.src_node_id
                
                if client_id:
                    results[client_id] = (response, None)
                    matched_clients.add(client_id)

        # Mark unmatched clients as failed
        for client_id in client_ids:
            if client_id not in matched_clients:
                results[client_id] = (None, "No response received")

        return results

    def _attempt_broadcast_recovery(
        self,
        failed_partitions: Dict[int, str],
        message_type: str,
        arrays: Dict[str, np.ndarray],
        config: Dict[str, Any],
        timeout: float,
    ) -> Dict[int, Tuple[str, Optional[str], Optional[Message], Optional[str]]]:
        """Attempt to recover failed partitions with full initialization and retry communication.

        This method performs a complete recovery flow:
        1. Discover and register new nodes
        2. Initialize new clients with proper partition data
        3. Normalize data if server has global statistics
        4. Apply PCA if server has PCA components
        5. Send the original message to fully initialized clients

        Args:
            failed_partitions: Dictionary mapping partition_id to original client_id
            message_type: Type of message
            arrays: Dictionary of numpy arrays to send
            config: Configuration dictionary
            timeout: Timeout in seconds

        Returns:
            Dictionary mapping partition_id to (original_client_id, new_client_id, response, error_message)
        """
        results: Dict[int, Tuple[str, Optional[str], Optional[Message], Optional[str]]] = {}

        # Discover and register new nodes with retry logic
        start_time = time.time()
        new_nodes = self.client_manager.discover_new_nodes(self.grid)

        while not new_nodes and (time.time() - start_time < self.retry_config.recovery_timeout):
            elapsed = time.time() - start_time
            self.logger.logger.info(
                f"No new nodes discovered for partition recovery. "
                f"Waiting {self.retry_config.recovery_retry_delay}s... "
                f"(elapsed: {elapsed:.1f}s, timeout: {self.retry_config.recovery_timeout}s)"
            )
            time.sleep(self.retry_config.recovery_retry_delay)
            new_nodes = self.client_manager.discover_new_nodes(self.grid)

        if not new_nodes:
            self.logger.logger.warning(
                f"No new nodes discovered for partition recovery after {self.retry_config.recovery_timeout}s"
            )
            for partition_id, original_client_id in failed_partitions.items():
                results[partition_id] = (original_client_id, None, None, "No new nodes available for recovery")
            return results

        self.logger.logger.info(f"Found {len(new_nodes)} new nodes for recovery")

        for node_id in new_nodes:
            self.client_manager.register_client(node_id)

        new_node_list = list(new_nodes)
        partition_list = list(failed_partitions.keys())

        # Map new nodes to partitions (simple 1-to-1 mapping)
        for i, partition_id in enumerate(partition_list):
            if i >= len(new_node_list):
                # Not enough new nodes
                original_client_id = failed_partitions[partition_id]
                self.logger.logger.warning(
                    f"Insufficient new nodes for partition {partition_id} (original client: {original_client_id})"
                )
                results[partition_id] = (original_client_id, None, None, "Insufficient new nodes for recovery")
                continue

            new_client_id = new_node_list[i]
            original_client_id = failed_partitions[partition_id]

            # Initialize the new client with proper configuration
            init_response, init_error = self._initialize_recovered_client(new_client_id, partition_id, timeout)

            if init_response is None:
                recovery_error = f"Initialization failed: {init_error}"
                results[partition_id] = (original_client_id, new_client_id, None, recovery_error)
                self.logger.logger.warning(
                    f"✗ Recovery initialization failed for partition {partition_id} "
                    f"(new client {new_client_id}): {init_error}"
                )
                continue

            # Step 2: Normalize data if server has global statistics
            if self.server_state.is_normalized and self.server_state.global_mean is not None:
                norm_response, norm_error = self._send_single_message(
                    new_client_id,
                    "query.normalize_data",
                    {"global_mean": self.server_state.global_mean, "global_std": self.server_state.global_std},
                    {},
                    timeout,
                )

                if norm_response is None:
                    recovery_error = f"Normalization failed: {norm_error}"
                    results[partition_id] = (original_client_id, new_client_id, None, recovery_error)
                    self.client_manager.mark_client_failed(new_client_id, "Normalization failed during recovery")
                    self.logger.logger.warning(
                        f"✗ Recovery normalization failed for partition {partition_id} "
                        f"(new client {new_client_id}): {norm_error}"
                    )
                    continue

            # Apply PCA if server has PCA components
            if (
                hasattr(self.server_state, "principal_components")
                and self.server_state.principal_components is not None
            ):
                pca_response, pca_error = self._send_single_message(
                    new_client_id,
                    "query.project_to_pca",
                    {
                        "pca_mean": self.server_state.pca_mean,
                        "principal_components": self.server_state.principal_components,
                    },
                    {"experiment_dir": str(self.means_logger.experiment_dir)},
                    timeout,
                )

                if pca_response is None:
                    recovery_error = f"PCA projection failed: {pca_error}"
                    results[partition_id] = (original_client_id, new_client_id, None, recovery_error)
                    self.client_manager.mark_client_failed(new_client_id, "PCA projection failed during recovery")
                    self.logger.logger.warning(
                        f"✗ Recovery PCA projection failed for partition {partition_id} "
                        f"(new client {new_client_id}): {pca_error}"
                    )
                    continue

            # Send the original message to the fully initialized client
            response, error_msg = self._send_single_message(new_client_id, message_type, arrays, config, timeout)

            if response is not None:
                # Success - client is fully recovered and ready
                results[partition_id] = (original_client_id, new_client_id, response, None)
                self.logger.logger.info(
                    f"✓ Recovery successful for partition {partition_id}: " f"{original_client_id} → {new_client_id}"
                )
            else:
                recovery_error = error_msg or "Failed to communicate with recovered client"
                results[partition_id] = (original_client_id, new_client_id, None, recovery_error)
                self.logger.logger.warning(
                    f"✗ Recovery failed for partition {partition_id} " f"(new client {new_client_id}): {error_msg}"
                )

        return results

    def _initialize_recovered_client(
        self, client_id: str, partition_id: int, timeout: float
    ) -> Tuple[Optional[Message], Optional[str]]:
        """Initialize a recovered client with proper configuration.

        Args:
            client_id: New client ID to initialize
            partition_id: Partition ID for the client
            timeout: Timeout in seconds

        Returns:
            Tuple of (response_message, error_message)
        """
        # Get initialization config from server state
        init_config = {
            "k_global": self.server_state.config.k_global,
            "max_iterations": self.server_state.config.max_iterations,
            "privacy_threshold": self.server_state.config.privacy_threshold,
            "convergence_tolerance": self.server_state.config.convergence_tolerance,
            "local_kmeans_iterations": self.server_state.config.local_kmeans_iterations,
            "data_path": self.data_path,
            "data_source": self.data_source,
            "experiment_dir": str(self.means_logger.experiment_dir),
        }

        # Send initialization message
        response, error_msg = self._send_single_message(client_id, "query.initialize_client", {}, init_config, timeout)

        if response is None:
            return None, error_msg

        # Process initialization response
        try:
            result_data = extract_message_content(response.content)
            metrics = result_data.get("metrics", {})

            returned_partition_id = metrics.get("partition_id")
            if returned_partition_id != partition_id:
                error_msg = f"Partition ID mismatch: expected {partition_id}, got {returned_partition_id}"
                return None, error_msg

            success = metrics.get("success", False)
            data_dimensions = metrics.get("data_dimensions", 0)
            num_samples = metrics.get("num_samples", 0)
            error_message = metrics.get("error_message")

            if success:
                self.client_manager.mark_client_initialized(
                    client_id,
                    partition_id,
                    {"data_dimensions": data_dimensions, "num_samples": num_samples},
                )
                self.client_manager.update_client_activity(client_id)
                self.logger.logger.info(
                    f"✓ Initialized recovered client {client_id} with partition {partition_id}: "
                    f"{num_samples} samples, {data_dimensions} dimensions"
                )
                return response, None
            else:
                return None, error_message

        except Exception as e:
            return None, f"Error processing initialization response: {str(e)}"

    def _retry_with_recovery(
        self,
        client_id: str,
        message_type: str,
        arrays: Optional[Dict[str, np.ndarray]],
        config: Optional[Dict[str, Any]],
        timeout: float,
        max_retries: int,
        enable_recovery: bool = True,
    ) -> Tuple[Optional[Message], List[str], Optional[str]]:
        """Internal method to retry a request with partition recovery.

        Args:
            client_id: Target client ID
            message_type: Type of message
            arrays: Optional numpy arrays to send
            config: Optional configuration dictionary
            timeout: Timeout in seconds
            max_retries: Maximum number of retry attempts
            enable_recovery: Whether to attempt partition recovery on failure

        Returns:
            Tuple of (response_message, error_messages, recovered_client_id)
        """
        # Ensure arrays and config are dictionaries
        arrays = arrays or {}
        config = config or {}

        error_messages = []
        recovered_client_id = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                time.sleep(self.retry_config.retry_delay)

            response, error_msg = self._send_single_message(client_id, message_type, arrays, config, timeout)

            if response is not None:
                return response, error_messages, recovered_client_id

            if error_msg:
                error_messages.append(f"Attempt {attempt + 1}: {error_msg}")

        # All attempts failed - mark client as failed if recovery enabled
        if enable_recovery:
            partition_id = self.client_manager.get_partition_id(client_id)
            if partition_id is not None:
                self.client_manager.mark_client_failed(client_id, f"Communication failed after {max_retries + 1} attempts")

        return None, error_messages, recovered_client_id

    def _send_single_message(
        self, client_id: str, message_type: str, arrays: Dict[str, np.ndarray], config: Dict[str, Any], timeout: float
    ) -> Tuple[Optional[Message], Optional[str]]:
        """Internal method to send a single message without retry.

        Args:
            client_id: Target client ID
            message_type: Type of message
            arrays: Dictionary of numpy arrays to send
            config: Configuration dictionary
            timeout: Timeout in seconds

        Returns:
            Tuple of (response_message, error_message)
        """
        try:
            content = create_message_content(arrays=arrays, config=config)
            message = Message(
                content=RecordDict(content), dst_node_id=client_id, message_type=message_type, ttl=timeout, group_id=""
            )

            # Send and receive
            responses = self.grid.send_and_receive([message], timeout=timeout)

            # Validate response
            if not responses or responses[0] is None:
                error_msg = f"No response received from client {client_id}"
                self.logger.logger.warning(f"No response from client {client_id} for {message_type}")
                return None, error_msg

            response = responses[0]
            if not hasattr(response, "content"):
                error_msg = f"Invalid response structure from client {client_id}: missing content"
                self.logger.logger.warning(error_msg)
                return None, error_msg

            return response, None

        except Exception as e:
            error_msg = f"Error communicating with client {client_id}: {str(e)}"
            self.logger.logger.warning(error_msg)
            return None, error_msg

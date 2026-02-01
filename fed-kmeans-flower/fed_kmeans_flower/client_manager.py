import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from flwr.serverapp import Grid

from fed_kmeans_flower.models import FederatedKMeansConfig


# Client management state
class ClientManager:
    """Manages client connections, timeouts, and status monitoring with fault tolerance."""

    def __init__(self, config: FederatedKMeansConfig):
        self.config = config
        self.connected_clients: Set[str] = set()
        self.initialized_clients: Set[str] = set()
        self.client_last_seen: Dict[str, float] = {}
        self.client_data_stats: Dict[str, Dict[str, Any]] = {}
        self.failed_clients: Set[str] = set()
        self.reconnected_clients: Set[str] = set()
        self.client_failure_counts: Dict[str, int] = {}
        self.client_round_participation: Dict[str, List[int]] = {}
        self.max_failures_per_client = 3

        # Partition tracking
        self.client_to_partition: Dict[str, int] = {}  # Maps client_id to partition_id
        self.partition_to_client: Dict[int, str] = {}  # Maps partition_id to client_id
        self.expected_num_partitions: Optional[int] = None

    def register_client(self, client_id) -> None:
        """Register a new client connection."""
        self.connected_clients.add(client_id)
        self.client_last_seen[client_id] = time.time()

        # Check if this is a reconnection
        if client_id in self.failed_clients:
            self.reconnected_clients.add(client_id)
            self.failed_clients.discard(client_id)
            logging.getLogger(__name__).info(
                f"Client {client_id} reconnected. Total clients: {len(self.connected_clients)}"
            )
        else:
            logging.getLogger(__name__).info(
                f"Client {client_id} connected. Total clients: {len(self.connected_clients)}"
            )

    def update_client_activity(self, client_id) -> None:
        """Update client's last seen timestamp."""
        self.client_last_seen[client_id] = time.time()

    def mark_client_initialized(self, client_id: str, partition_id: int, data_stats: Dict[str, Any]) -> None:
        """Mark client as successfully initialized with partition mapping."""

        # Check if partition was previously served by a different client
        if partition_id in self.partition_to_client:
            old_client_id = self.partition_to_client[partition_id]
            if old_client_id != client_id and old_client_id in self.failed_clients:
                logging.getLogger(__name__).info(
                    f"Partition {partition_id} replacement: " f"old client {old_client_id} → new client {client_id}"
                )
                # Remove old client from failed set since it's been replaced
                self.failed_clients.discard(old_client_id)

        # Update partition mappings
        self.client_to_partition[client_id] = partition_id
        self.partition_to_client[partition_id] = client_id

        # Mark as initialized
        self.initialized_clients.add(client_id)
        self.client_data_stats[client_id] = data_stats

        # Reset failure count on successful initialization
        self.client_failure_counts[client_id] = 0

        logging.getLogger(__name__).info(
            f"Client {client_id} initialized with partition {partition_id}. Data: {data_stats}"
        )

    def mark_client_failed(self, client_id: str, reason: str) -> None:
        """Mark client as failed and remove from active sets."""
        # Increment failure count
        self.client_failure_counts[client_id] = self.client_failure_counts.get(client_id, 0) + 1

        # Get partition before removing client
        partition_id = self.client_to_partition.get(client_id)

        # Only permanently fail client if it exceeds max failures
        if self.client_failure_counts[client_id] >= self.max_failures_per_client:
            self.failed_clients.add(client_id)
            self.connected_clients.discard(client_id)
            self.initialized_clients.discard(client_id)

            # Keep partition mapping for potential recovery
            # Don't remove from partition_to_client to allow replacement

            logging.getLogger(__name__).error(
                f"Client {client_id} (partition {partition_id}) permanently failed "
                f"after {self.client_failure_counts[client_id]} failures: {reason}"
            )
        else:
            # Temporary failure - keep in connected but remove from initialized
            self.initialized_clients.discard(client_id)
            logging.getLogger(__name__).warning(
                f"Client {client_id} (partition {partition_id}) temporary failure "
                f"({self.client_failure_counts[client_id]}/{self.max_failures_per_client}): {reason}"
            )

    def record_client_participation(self, client_id, round_number: int) -> None:
        """Record client participation in a clustering round."""
        if client_id not in self.client_round_participation:
            self.client_round_participation[client_id] = []
        self.client_round_participation[client_id].append(round_number)

    def check_client_timeouts(self) -> List:
        """Check for client timeouts and return list of timed out clients."""
        current_time = time.time()
        timed_out_clients = []

        for client_id in list(self.connected_clients):
            last_seen = self.client_last_seen.get(client_id, 0)
            if current_time - last_seen > self.config.client_timeout:
                timed_out_clients.append(client_id)
                self.mark_client_failed(client_id, f"Timeout after {self.config.client_timeout}s")

        return timed_out_clients

    def get_active_clients(self) -> List:
        """Get list of currently active (connected and not failed) clients."""
        return list(self.connected_clients - self.failed_clients)

    def get_initialized_clients(self) -> List:
        """Get list of successfully initialized clients."""
        return list(self.initialized_clients - self.failed_clients)

    def has_minimum_clients(self) -> bool:
        """Check if we have minimum required clients."""
        return len(self.get_active_clients()) >= self.config.required_clients

    def has_required_clients(self) -> bool:
        """Check if we have the required number of initialized clients.

        This is the minimum threshold - experiments cannot proceed below this number.
        """
        return len(self.get_initialized_clients()) >= self.config.required_clients

    def validate_data_compatibility(self) -> Tuple[bool, Optional[str]]:
        """Validate that all initialized clients have compatible data dimensions."""
        if not self.client_data_stats:
            return False, "No client data statistics available"

        dimensions = set()
        for client_id, stats in self.client_data_stats.items():
            if client_id in self.initialized_clients:
                dimensions.add(stats.get("data_dimensions", 0))

        if len(dimensions) > 1:
            return False, f"Incompatible data dimensions across clients: {dimensions}"
        elif len(dimensions) == 0:
            return False, "No valid data dimensions found"

        return True, None

    def discover_new_nodes(self, grid: Grid) -> List[int]:
        """Discover node_ids that are available but not yet registered.

        Returns:
            List of new node_ids that haven't been registered yet
        """
        try:
            available_nodes = grid.get_node_ids()
            new_nodes = [
                node_id
                for node_id in available_nodes
                if node_id not in self.connected_clients and node_id not in self.failed_clients
            ]

            if new_nodes:
                logging.getLogger(__name__).info(f"Discovered {len(new_nodes)} new nodes: {new_nodes}")

            return new_nodes

        except Exception as e:
            logging.getLogger(__name__).error(f"Error discovering new nodes: {str(e)}")
            return []

    def get_failed_partitions(self) -> Set[int]:
        """Get set of partition IDs for clients that have failed.

        Returns:
            Set of partition_ids that belong to failed clients
        """
        failed_partitions = set()
        for client_id in self.failed_clients:
            partition_id = self.client_to_partition.get(client_id)
            if partition_id is not None:
                failed_partitions.add(partition_id)
        return failed_partitions

    def get_partition_id(self, client_id: str) -> Optional[int]:
        """Get the partition ID for a client."""
        return self.client_to_partition.get(client_id)

    def get_client_for_partition(self, partition_id: int) -> Optional[str]:
        """Get the client ID serving a specific partition."""
        return self.partition_to_client.get(partition_id)

    def get_active_partitions(self) -> Set[int]:
        """Get set of currently active partition IDs."""
        active_clients = self.get_initialized_clients()
        return {
            self.client_to_partition[client_id] for client_id in active_clients if client_id in self.client_to_partition
        }

    def get_missing_partitions(self) -> Set[int]:
        """Get set of partition IDs that are not currently active."""
        if self.expected_num_partitions is None:
            return set()

        active_partitions = self.get_active_partitions()
        expected_partitions = set(range(self.expected_num_partitions))
        return expected_partitions - active_partitions

    def has_partition_coverage(self) -> bool:
        """Check if all expected partitions are present."""
        missing = self.get_missing_partitions()
        return len(missing) == 0

    def set_expected_num_partitions(self, num_partitions: int) -> None:
        """Set the expected number of partitions for validation."""
        self.expected_num_partitions = num_partitions
        logging.getLogger(__name__).info(f"Expected number of partitions set to {num_partitions}")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of client manager status."""
        return {
            "connected_clients": len(self.connected_clients),
            "initialized_clients": len(self.initialized_clients),
            "failed_clients": len(self.failed_clients),
            "active_clients": len(self.get_active_clients()),
            "reconnected_clients": len(self.reconnected_clients),
            "has_minimum_clients": self.has_minimum_clients(),
            "has_required_clients": self.has_required_clients(),
            "client_list": list(self.get_active_clients()),
            "partition_mappings": dict(self.client_to_partition),
            "active_partitions": list(self.get_active_partitions()),
            "missing_partitions": list(self.get_missing_partitions()),
            "has_partition_coverage": self.has_partition_coverage(),
            "failure_counts": dict(self.client_failure_counts),
            "participation_summary": {
                client_id: len(rounds) for client_id, rounds in self.client_round_participation.items()
            },
        }

    def get_fault_tolerance_report(self) -> Dict[str, Any]:
        """Generate a detailed fault tolerance report."""
        active_clients = self.get_initialized_clients()

        return {
            "total_clients_seen": len(self.connected_clients | self.failed_clients),
            "currently_active": len(active_clients),
            "permanently_failed": len(self.failed_clients),
            "reconnections": len(self.reconnected_clients),
            "client_reliability": {
                client_id: {
                    "failure_count": self.client_failure_counts.get(client_id, 0),
                    "rounds_participated": len(self.client_round_participation.get(client_id, [])),
                    "is_active": client_id in active_clients,
                    "last_seen": self.client_last_seen.get(client_id, 0),
                }
                for client_id in (self.connected_clients | self.failed_clients)
            },
            "system_health": {
                "meets_minimum_threshold": self.has_required_clients(),
                "required_clients": self.config.required_clients,
                "current_clients": len(active_clients),
                "fault_tolerance_level": len(active_clients) - self.config.required_clients,
            },
        }

    def ensure_required_clients(self, logger) -> bool:
        """Check if we have the required number of clients.

        Args:
            logger: Logger instance for warnings

        Returns:
            bool: True if required clients available, False otherwise
        """
        if self.has_required_clients():
            missing_partitions = self.get_missing_partitions()
            if missing_partitions:
                logger.logger.warning(
                    f"Missing partitions detected: {sorted(missing_partitions)}. "
                    f"Clustering will proceed with available partitions."
                )
            return True

        current_count = len(self.get_initialized_clients())
        logger.logger.warning(f"Insufficient clients: {current_count} < {self.config.required_clients}")
        return False

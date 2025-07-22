"""clef-vertical-messaging-api: A Flower / PyTorch app."""

import pickle
import time
from itertools import cycle
from time import sleep

import numpy as np
import torch
from flwr.common import ConfigRecord, Context, Message, RecordDict
from flwr.server import Grid, ServerApp
from torch.utils.data import DataLoader

from clef_vertical_messaging_api.data import get_labels
from clef_vertical_messaging_api.models import CombinedNetwork
from clef_vertical_messaging_api.network_types import NetworkType
from clef_vertical_messaging_api.task import deserialize_embedding, serialize_gradients
from clef_vertical_messaging_api.utils import get_model_config


def node_online_loop(grid: Grid, required_nodes: int = 2, timeout: float = 60.0) -> list[int]:
    """Wait for a specific number of nodes to become available.

    Args:
        grid: The Flower Grid instance
        required_nodes: Number of nodes to wait for (personal + clinical)
        timeout: Maximum time to wait in seconds

    Returns:
        List of available node IDs

    Raises:
        TimeoutError: If required number of nodes is not found within timeout period
    """
    node_ids = []
    start_time = time.time()
    last_node_count = 0

    while len(node_ids) < required_nodes:
        current_nodes = list(grid.get_node_ids())  # Convert set to list
        if len(current_nodes) > last_node_count:
            print(f"Found {len(current_nodes)}/{required_nodes} nodes")
            last_node_count = len(current_nodes)

        node_ids = current_nodes

        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Timeout waiting for nodes. Found {len(node_ids)}/{required_nodes} " f"nodes after {timeout} seconds."
            )

        sleep(1)

    print(f"All {required_nodes} nodes are ready!")
    return node_ids


def initialize_clients(
    grid: Grid,
    node_ids: list[int],
    context: Context,
    timeout: float = 30.0,
) -> dict[int, NetworkType]:
    """Initialize client nodes with their respective network types and data.

    Args:
        grid: The Flower Grid instance
        node_ids: List of available node IDs
        context: Server context containing configuration
        timeout: Maximum time to wait for replies in seconds

    Returns:
        Dictionary mapping node_id to NetworkType
    """
    # Assign network types to nodes (first node = personal, second = clinical)
    node_assignments = {}
    network_types = [NetworkType.PERSONAL, NetworkType.CLINICAL]

    for i, node_id in enumerate(node_ids[:2]):  # Only use first 2 nodes
        network_type = network_types[i]
        node_assignments[node_id] = network_type

        # Send initialization message
        config_record = ConfigRecord({"network_type": network_type.value})
        message = Message(
            content=RecordDict({"config": config_record}),
            message_type="query.load_data",
            dst_node_id=node_id,
            group_id="clef_vertical",
        )

        print(f"Initializing node {node_id} as {network_type.name}")
        replies = grid.send_and_receive([message], timeout=timeout)

        if not replies or not replies[0].has_content():
            raise Exception(f"Failed to initialize node {node_id}")

        result = replies[0].content["result"]
        if not result["success"]:
            raise Exception(f"Node {node_id} initialization failed: {result.get('error', 'Unknown error')}")

        print(f"Successfully initialized {network_type.name} client on node {node_id}")

    return node_assignments


def training_round(
    grid: Grid,
    node_assignments: dict[int, NetworkType],
    batch_indices: list[int],
    combined_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    targets: torch.Tensor,
    timeout: float = 30.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Execute one training round of vertical federated learning.

    Args:
        grid: The Flower Grid instance
        node_assignments: Mapping of node_id to NetworkType
        batch_indices: Indices of the current batch
        combined_net: The combined network for final prediction
        optimizer: Optimizer for the combined network
        criterion: Loss criterion
        targets: Target values for the batch
        timeout: Maximum time to wait for replies in seconds

    Returns:
        Tuple of (loss_value, personal_gradients, clinical_gradients)
    """
    # Step 1: Send forward pass requests to both clients
    forward_messages = []
    batch_indices_str = ",".join([str(idx) for idx in batch_indices])

    for node_id, network_type in node_assignments.items():
        config_record = ConfigRecord({"batch_indices": batch_indices_str})
        message = Message(
            content=RecordDict({"config": config_record}),
            message_type="query.forward_pass",
            dst_node_id=node_id,
            group_id="clef_vertical",
        )
        forward_messages.append(message)

    # Collect embeddings from both clients
    print(f"Requesting embeddings for batch of size {len(batch_indices)}")
    replies = grid.send_and_receive(forward_messages, timeout=timeout)

    personal_embedding = None
    clinical_embedding = None

    for reply in replies:
        # print(f"Reply type: {type(reply)}")
        # print(f"Reply has_content: {reply.has_content()}")
        if reply.has_content():
            # print(f"Reply content type: {type(reply.content)}")
            # print(f"Reply content keys: {list(reply.content.keys()) if hasattr(reply.content, 'keys') else 'No keys method'}")
            # print(f"Reply content: {reply.content}")

            # Check if this is an error response
            if "result" in reply.content:
                result = reply.content["result"]
                if not result.get("success", True):
                    raise Exception(f"Client forward pass failed: {result.get('error', 'Unknown error')}")

            # Check if this is a successful response with embeddings
            if "embeddings" in reply.content:
                try:
                    embeddings_data = reply.content["embeddings"]
                    # print(f"Successfully accessed embeddings_data: {type(embeddings_data)}")
                    network_type_value = embeddings_data["network_type"]
                    embedding_bytes = embeddings_data["data"]
                    embedding = deserialize_embedding(embedding_bytes)
                    embedding = embedding.clone().detach().requires_grad_(True)

                    if network_type_value == NetworkType.PERSONAL.value:
                        personal_embedding = embedding
                    elif network_type_value == NetworkType.CLINICAL.value:
                        clinical_embedding = embedding
                except Exception as e:
                    print(f"Error processing embeddings: {e}")
                    print(f"Reply content structure: {reply.content}")
                    raise
            else:
                print(f"Warning: Reply has no 'embeddings' key. Keys: {list(reply.content.keys())}")
                print(f"Full reply content: {reply.content}")

    if personal_embedding is None or clinical_embedding is None:
        raise Exception("Failed to collect embeddings from both clients")

    # Step 2: Forward pass through combined network
    combined_net.train()
    prediction = combined_net(personal_embedding, clinical_embedding)
    loss = criterion(prediction, targets)

    # Step 3: Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Step 4: Get gradients for embeddings
    personal_gradients = personal_embedding.grad.numpy()
    clinical_gradients = clinical_embedding.grad.numpy()

    return loss.item(), personal_gradients, clinical_gradients


def send_gradients_to_clients(
    grid: Grid,
    node_assignments: dict[int, NetworkType],
    personal_gradients: np.ndarray,
    clinical_gradients: np.ndarray,
    timeout: float = 30.0,
):
    """Send gradients back to clients for model updates.

    Args:
        grid: The Flower Grid instance
        node_assignments: Mapping of node_id to NetworkType
        personal_gradients: Gradients for personal network
        clinical_gradients: Gradients for clinical network
        timeout: Maximum time to wait for replies in seconds
    """
    backward_messages = []

    for node_id, network_type in node_assignments.items():
        if network_type == NetworkType.PERSONAL:
            gradients_bytes = serialize_gradients(personal_gradients)
        elif network_type == NetworkType.CLINICAL:
            gradients_bytes = serialize_gradients(clinical_gradients)
        else:
            continue

        config_record = ConfigRecord({"gradients": gradients_bytes})
        message = Message(
            content=RecordDict({"config": config_record}),
            message_type="query.backward_pass",
            dst_node_id=node_id,
            group_id="clef_vertical",
        )
        backward_messages.append(message)

    print("Sending gradients to clients for model updates")
    replies = grid.send_and_receive(backward_messages, timeout=timeout)

    # Verify all clients updated successfully
    for reply in replies:
        if reply.has_content():
            result = reply.content["result"]
            if not result["success"]:
                print(f"Warning: Client update failed: {result.get('error', 'Unknown error')}")


def evaluation_round(
    grid: Grid,
    node_assignments: dict[int, NetworkType],
    combined_net: torch.nn.Module,
    criterion: torch.nn.Module,
    test_targets: torch.Tensor,
    timeout: float = 30.0,
) -> float:
    """Execute one evaluation round.

    Args:
        grid: The Flower Grid instance
        node_assignments: Mapping of node_id to NetworkType
        combined_net: The combined network for final prediction
        criterion: Loss criterion
        test_targets: Target values for evaluation
        timeout: Maximum time to wait for replies in seconds

    Returns:
        Evaluation loss
    """
    # Send evaluation requests to both clients
    eval_messages = []
    for node_id in node_assignments.keys():
        message = Message(
            content=RecordDict({}),
            message_type="query.evaluate_model",
            dst_node_id=node_id,
            group_id="clef_vertical",
        )
        eval_messages.append(message)

    print("Requesting evaluation embeddings from clients")
    replies = grid.send_and_receive(eval_messages, timeout=timeout)

    personal_embedding = None
    clinical_embedding = None

    for reply in replies:
        if reply.has_content():
            # Check if this is an error response
            if "result" in reply.content:
                result = reply.content["result"]
                if not result.get("success", True):
                    raise Exception(f"Client evaluation failed: {result.get('error', 'Unknown error')}")

            # Check if this is a successful response with embeddings
            if "embeddings" in reply.content:
                embeddings_data = reply.content["embeddings"]
                network_type_value = embeddings_data["network_type"]
                embedding_bytes = embeddings_data["data"]
                embedding = deserialize_embedding(embedding_bytes)

                if network_type_value == NetworkType.PERSONAL.value:
                    personal_embedding = embedding
                elif network_type_value == NetworkType.CLINICAL.value:
                    clinical_embedding = embedding

    if personal_embedding is None or clinical_embedding is None:
        raise Exception("Failed to collect evaluation embeddings from both clients")

    # Forward pass through combined network
    combined_net.eval()
    with torch.no_grad():
        prediction = combined_net(personal_embedding, clinical_embedding)
        loss = criterion(prediction, test_targets)

    return loss.item()


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main server function to coordinate vertical federated learning."""
    # Get configuration from context
    required_nodes = int(context.run_config.get("required-nodes", 2))
    node_timeout = float(context.run_config.get("node-timeout", 60.0))
    training_timeout = float(context.run_config.get("training-timeout", 30.0))
    num_epochs = int(context.run_config.get("num-epochs", 5))

    # Get model configuration
    model_config = get_model_config(context)

    try:
        # Step 1: Wait for nodes to become available
        print(f"Waiting for {required_nodes} nodes to become available...")
        node_ids = node_online_loop(grid, required_nodes=required_nodes, timeout=node_timeout)

        # Step 2: Initialize clients with their network types
        node_assignments = initialize_clients(grid, node_ids, context, training_timeout)

        # Step 3: Initialize server components
        combined_net = CombinedNetwork(model_config)
        optimizer = torch.optim.Adam(combined_net.parameters(), lr=model_config["learning_rate"])
        criterion = torch.nn.MSELoss()

        # Load labels and create data loader
        train_dataset, test_dataset, y_train_original, y_test_original = get_labels()
        batch_size = model_config["batch_size"]
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Convert batches to list for easier access
        all_batches = []
        for batch in trainloader:
            all_batches.append(batch)

        print(f"Starting vertical federated learning with {len(all_batches)} batches per epoch")

        # Step 4: Training loop
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            epoch_losses = []

            for batch_idx, batch in enumerate(all_batches):
                indices = batch[0].flatten().numpy().tolist()
                targets = batch[1]

                # Execute training round
                loss, personal_gradients, clinical_gradients = training_round(
                    grid, node_assignments, indices, combined_net, optimizer, criterion, targets, training_timeout
                )

                epoch_losses.append(loss)

                # Send gradients back to clients
                send_gradients_to_clients(
                    grid, node_assignments, personal_gradients, clinical_gradients, training_timeout
                )

                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(all_batches)}, Loss: {loss:.4f}")

            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} average training loss: {avg_loss:.4f}")

            # Evaluation
            if epoch % 1 == 0:  # Evaluate every epoch
                eval_loss = evaluation_round(
                    grid, node_assignments, combined_net, criterion, test_dataset.tensors[1], training_timeout
                )
                print(f"Epoch {epoch+1} evaluation loss: {eval_loss:.4f}")

        print("\nVertical federated learning completed successfully!")

    except TimeoutError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

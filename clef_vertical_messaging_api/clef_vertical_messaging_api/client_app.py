"""clef-vertical-messaging-api: A Flower / PyTorch app."""

import pickle

import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import Array, ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict

from clef_vertical_messaging_api.network_types import NetworkType, WeightType
from clef_vertical_messaging_api.task import (
    create_client_model,
    deserialize_gradients,
    get_weights,
    load_client_data,
    serialize_embedding,
    set_weights,
)
from clef_vertical_messaging_api.utils import get_model_config


def get_state_dict_store_name(model_type, weight_type):
    """Get the name of the state dictionary to store in the ArrayRecord.

    Args:
        model_type: Type of model (e.g., PERSONAL, CLINICAL)
        weight_type: Type of weights (e.g., MODEL, OPTIMIZER)

    Returns:
        The name of the state dictionary
    """
    return f"{model_type}_{weight_type}_state_dict"


def store_weights(state_dict, model_type, weight_type, parameter_record):
    """Store model or optimizer weights in the ArrayRecord.

    Args:
        state_dict: The state dictionary to store
        model_type: Type of model (e.g., PERSONAL, CLINICAL)
        weight_type: Type of weights (e.g., MODEL, OPTIMIZER)
        parameter_record: The ArrayRecord to store the weights in
    """
    # Pickle the entire state dictionary and store it as a single array
    key = get_state_dict_store_name(model_type, weight_type)
    state_dict_bytes = pickle.dumps(state_dict)
    state_dict_array = Array(np.frombuffer(state_dict_bytes, dtype=np.uint8))
    parameter_record[key] = state_dict_array


def get_weights_from_context(parameter_record, model_type, weight_type):
    """Load weights from the ArrayRecord.

    Args:
        parameter_record: The ArrayRecord containing the weights
        model_type: Type of model (e.g., PERSONAL, CLINICAL)
        weight_type: Type of weights (e.g., MODEL, OPTIMIZER)

    Returns:
        The state dictionary with the loaded weights
    """
    key = get_state_dict_store_name(model_type, weight_type)
    state_dict_bytes_retrieved = parameter_record[key].numpy().tobytes()
    return pickle.loads(state_dict_bytes_retrieved)


def store_dataset(dataset, dataset_name, parameter_record):
    """Store dataset in the ArrayRecord.

    Args:
        dataset: The dataset tensor to store
        dataset_name: Name for the dataset (e.g., 'train_dataset', 'val_dataset')
        parameter_record: The ArrayRecord to store the dataset in
    """
    dataset_bytes = pickle.dumps(dataset.cpu() if hasattr(dataset, "cpu") else dataset)
    dataset_array = Array(np.frombuffer(dataset_bytes, dtype=np.uint8))
    parameter_record[dataset_name] = dataset_array


def get_dataset_from_context(parameter_record, dataset_name):
    """Load dataset from the ArrayRecord.

    Args:
        parameter_record: The ArrayRecord containing the dataset
        dataset_name: Name of the dataset to retrieve

    Returns:
        The loaded dataset
    """
    dataset_bytes_retrieved = parameter_record[dataset_name].numpy().tobytes()
    return pickle.loads(dataset_bytes_retrieved)


# Create the Flower ClientApp
app = ClientApp()


@app.query("load_data")
def load_data(message: Message, context: Context):
    """Load data and initialize model for specific network type."""
    try:
        # Extract network type from the message
        network_type_value = message.content["config"]["network_type"]
        network_type = NetworkType(network_type_value)

        # Get model configuration
        model_config = get_model_config(context)

        # Load data based on network type
        train_dataset, val_dataset = load_client_data(network_type)

        # Create model
        input_size = train_dataset.shape[1]
        net = create_client_model(network_type, input_size, model_config)

        # Setup device and optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=model_config["learning_rate"])

        # Initialize context storage if not exists
        if "datasets" not in context.state.array_records:
            context.state.array_records["datasets"] = ArrayRecord()
        if "state_dicts" not in context.state.array_records:
            context.state.array_records["state_dicts"] = ArrayRecord()
        if "embeddings" not in context.state.array_records:
            context.state.array_records["embeddings"] = ArrayRecord()
        if "config" not in context.state.array_records:
            context.state.array_records["config"] = ArrayRecord()

        # Store datasets in context
        datasets_rec = context.state.array_records["datasets"]
        store_dataset(train_dataset, f"train_dataset_{network_type.value}", datasets_rec)
        store_dataset(val_dataset, f"val_dataset_{network_type.value}", datasets_rec)

        # Store model and optimizer state dicts
        state_dict_rec = context.state.array_records["state_dicts"]
        store_weights(net.state_dict(), network_type.value, WeightType.MODEL.value, state_dict_rec)
        store_weights(optimizer.state_dict(), network_type.value, WeightType.OPTIMIZER.value, state_dict_rec)

        # Store configuration data
        config_rec = context.state.array_records["config"]
        config_data = {
            "network_type": network_type.value,
            "model_config": model_config,
            "device": str(device),
            "input_size": input_size,
        }
        config_bytes = pickle.dumps(config_data)
        config_array = Array(np.frombuffer(config_bytes, dtype=np.uint8))
        config_rec["client_config"] = config_array

        reply_record = RecordDict(
            {
                "result": ConfigRecord(
                    {
                        "success": True,
                        "network_type": network_type.value,
                        "input_size": input_size,
                        "num_train_samples": len(train_dataset),
                        "num_val_samples": len(val_dataset),
                    }
                )
            }
        )

        print(
            f"Successfully loaded {network_type.name} data and model",
            "train len: ",
            len(train_dataset),
            "val len: ",
            len(val_dataset),
        )
        return Message(reply_record, reply_to=message)

    except Exception as e:
        reply_record = RecordDict({"result": ConfigRecord({"success": False, "error": str(e)})})
        print(f"Error loading data: {e}")
        return Message(reply_record, reply_to=message)


@app.query("forward_pass")
def forward_pass(message: Message, context: Context):
    """Compute and return embeddings for given batch indices."""
    try:
        # Load configuration from context
        config_rec = context.state.array_records["config"]
        config_bytes_retrieved = config_rec["client_config"].numpy().tobytes()
        config_data = pickle.loads(config_bytes_retrieved)

        network_type = NetworkType(config_data["network_type"])
        model_config = config_data["model_config"]
        device = torch.device(config_data["device"])

        # Load datasets from context
        datasets_rec = context.state.array_records["datasets"]
        train_dataset = get_dataset_from_context(datasets_rec, f"train_dataset_{network_type.value}")

        # Load model state from context
        state_dict_rec = context.state.array_records["state_dicts"]
        model_state_dict = get_weights_from_context(state_dict_rec, network_type.value, WeightType.MODEL.value)

        # Recreate model and load state
        input_size = config_data["input_size"]
        net = create_client_model(network_type, input_size, model_config)
        net.to(device)
        net.load_state_dict(model_state_dict)

        # Extract batch indices
        print("network type: ", network_type)
        print("len train: ", len(train_dataset))
        batch_indices_str = message.content["config"]["batch_indices"]
        batch_indices = [int(idx) for idx in batch_indices_str.split(",")]

        # Get batch data
        batch = train_dataset[batch_indices]

        # Set model to training mode
        net.train()

        # Forward pass to compute embeddings
        embeddings = net(batch)

        # Store embeddings and batch indices in context for backward pass
        embeddings_rec = context.state.array_records["embeddings"]
        embedding_numpy = embeddings.cpu().detach().numpy()
        embeddings_rec[f"embedding_{network_type.value}"] = Array(embedding_numpy)
        embeddings_rec[f"batch_idx_{network_type.value}"] = Array(np.array(batch_indices))

        # Serialize embeddings for transmission
        embeddings_bytes = serialize_embedding(embeddings)

        reply_record = RecordDict(
            {"embeddings": ConfigRecord({"data": embeddings_bytes, "network_type": network_type.value})}
        )

        print(f"Computed embeddings for batch of size {len(batch_indices)}")
        return Message(reply_record, reply_to=message)

    except Exception as e:
        reply_record = RecordDict({"result": ConfigRecord({"success": False, "error": str(e)})})
        print(f"Error in forward pass: {e}")
        return Message(reply_record, reply_to=message)


@app.query("backward_pass")
def backward_pass(message: Message, context: Context):
    """Apply gradients and update model."""
    try:
        # Load configuration from context
        config_rec = context.state.array_records["config"]
        config_bytes_retrieved = config_rec["client_config"].numpy().tobytes()
        config_data = pickle.loads(config_bytes_retrieved)

        network_type = NetworkType(config_data["network_type"])
        model_config = config_data["model_config"]
        device = torch.device(config_data["device"])
        input_size = config_data["input_size"]

        # Load datasets from context
        datasets_rec = context.state.array_records["datasets"]
        train_dataset = get_dataset_from_context(datasets_rec, f"train_dataset_{network_type.value}")

        # Load model and optimizer states from context
        state_dict_rec = context.state.array_records["state_dicts"]
        model_state_dict = get_weights_from_context(state_dict_rec, network_type.value, WeightType.MODEL.value)
        optimizer_state_dict = get_weights_from_context(state_dict_rec, network_type.value, WeightType.OPTIMIZER.value)

        # Recreate model and optimizer and load states
        net = create_client_model(network_type, input_size, model_config)
        net.to(device)
        net.load_state_dict(model_state_dict)

        optimizer = torch.optim.Adam(net.parameters(), lr=model_config["learning_rate"])
        optimizer.load_state_dict(optimizer_state_dict)

        # Load batch indices from context
        embeddings_rec = context.state.array_records["embeddings"]
        batch_indices = embeddings_rec[f"batch_idx_{network_type.value}"].numpy()

        # Extract gradients from message
        gradients_bytes = message.content["config"]["gradients"]
        gradients = deserialize_gradients(gradients_bytes)

        # Recalculate embeddings for gradient computation
        batch = train_dataset[batch_indices]
        embeddings_recalculated = net(batch)

        # Apply gradients and update model
        embeddings_recalculated.backward(torch.tensor(gradients))
        optimizer.step()
        optimizer.zero_grad()

        # Store updated model and optimizer states back to context
        store_weights(net.state_dict(), network_type.value, WeightType.MODEL.value, state_dict_rec)
        store_weights(optimizer.state_dict(), network_type.value, WeightType.OPTIMIZER.value, state_dict_rec)

        reply_record = RecordDict({"result": ConfigRecord({"success": True, "network_type": network_type.value})})

        print(f"Successfully applied gradients and updated {network_type.name} model")
        return Message(reply_record, reply_to=message)

    except Exception as e:
        reply_record = RecordDict({"result": ConfigRecord({"success": False, "error": str(e)})})
        print(f"Error in backward pass: {e}")
        return Message(reply_record, reply_to=message)


@app.query("evaluate_model")
def evaluate_model(message: Message, context: Context):
    """Generate embeddings for evaluation dataset."""
    try:
        # Load configuration from context
        config_rec = context.state.array_records["config"]
        config_bytes_retrieved = config_rec["client_config"].numpy().tobytes()
        config_data = pickle.loads(config_bytes_retrieved)

        network_type = NetworkType(config_data["network_type"])
        model_config = config_data["model_config"]
        device = torch.device(config_data["device"])
        input_size = config_data["input_size"]

        # Load validation dataset from context
        datasets_rec = context.state.array_records["datasets"]
        val_dataset = get_dataset_from_context(datasets_rec, f"val_dataset_{network_type.value}")

        # Load model state from context
        state_dict_rec = context.state.array_records["state_dicts"]
        model_state_dict = get_weights_from_context(state_dict_rec, network_type.value, WeightType.MODEL.value)

        # Recreate model and load state
        net = create_client_model(network_type, input_size, model_config)
        net.to(device)
        net.load_state_dict(model_state_dict)

        # Set model to evaluation mode
        net.eval()

        # Compute embeddings for entire validation dataset
        with torch.no_grad():
            embeddings = net(val_dataset)

        # Serialize embeddings for transmission
        embeddings_bytes = serialize_embedding(embeddings)

        reply_record = RecordDict(
            {
                "embeddings": ConfigRecord(
                    {"data": embeddings_bytes, "network_type": network_type.value, "num_samples": len(val_dataset)}
                )
            }
        )

        print(f"Generated evaluation embeddings for {len(val_dataset)} samples")
        return Message(reply_record, reply_to=message)

    except Exception as e:
        reply_record = RecordDict({"result": ConfigRecord({"success": False, "error": str(e)})})
        print(f"Error in evaluation: {e}")
        return Message(reply_record, reply_to=message)

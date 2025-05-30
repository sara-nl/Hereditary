"""clef_vertical: A Flower / PyTorch app."""

import pickle

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Array, ArrayRecord, Context

from clef_vertical.data import get_clinical_data, get_personal_data
from clef_vertical.models import ClinicalNetwork, PersonalNetwork
from clef_vertical.network_types import NetworkType, WeightType
from clef_vertical.utils import get_model_config


def get_gradients_from_parameters(config, network_type):
    """
    Retrieve and deserialize the gradients for the specified network type from the configuration dictionary.

    Args:
        config (dict): The configuration dictionary containing serialized gradients under the keys
                       "personal_gradients" or "clinical_gradients".
        network_type (NetworkType): The type of network (PERSONAL or CLINICAL) for which to retrieve gradients.

    Returns:
        Any: The deserialized gradients object (typically a NumPy array or tensor).

    Raises:
        ValueError: If the provided network_type is not PERSONAL or CLINICAL, or if the expected gradients
                    key is missing from the config.
    """
    gradients = None
    if network_type == NetworkType.PERSONAL:
        gradients = config["personal_gradients"]
    elif network_type == NetworkType.CLINICAL:
        gradients = config["clinical_gradients"]
    else:
        raise ValueError(f"Invalid network_type: {network_type}. Must be PERSONAL or CLINICAL.")
    return pickle.loads(gradients)


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


def get_weights(parameter_record, model_type, weight_type):
    """Load weights from the ArrayRecord.

    Args:
        parameter_record: The ArrayRecord containing the weights
        model_type: Type of model (e.g., PERSONAL, CLINICAL)
        weight_type: Type of weights (e.g., MODEL, OPTIMIZER)

    Returns:
        The state dictionary with the loaded weights
    """
    key = get_state_dict_store_name(model_type, weight_type)
    optimizer_bytes_retrieved = parameter_record[key].numpy().tobytes()
    return pickle.loads(optimizer_bytes_retrieved)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, context, net, model_config, train_dataset, val_dataset, network_type=None):
        self.net = net
        self.model_config = model_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.network_type = network_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=model_config["learning_rate"])
        self.embedding = None
        self.client_state = context.state

        # Create the persistent ArrayRecords to store embeddings and weights if they do not exist yet
        if "embeddings" not in self.client_state.array_records:
            self.client_state.array_records["embeddings"] = ArrayRecord()
            embeddings_rec = self.client_state.array_records["embeddings"]
            embeddings_rec[f"embedding_{str(self.network_type)}"] = Array(np.array([]))
        if "state_dicts" not in self.client_state.array_records:
            self.client_state.array_records["state_dicts"] = ArrayRecord()

        # Store the initial model and optimizer state dicts
        state_dict_rec = self.client_state.array_records["state_dicts"]
        store_key = get_state_dict_store_name(self.network_type.value, WeightType.MODEL.value)
        if store_key not in state_dict_rec:
            print("first time saving model and optimizer state dicts")
            store_weights(self.net.state_dict(), self.network_type.value, WeightType.MODEL.value, state_dict_rec)
            store_weights(
                self.optimizer.state_dict(), self.network_type.value, WeightType.OPTIMIZER.value, state_dict_rec
            )

        # Load the model and optimizer state dicts. The client is re-initialized at every call.
        # so we do not need to load the state dicts in fit or evaluate.
        loaded_model_state_dict = get_weights(state_dict_rec, self.network_type.value, WeightType.MODEL.value)
        loaded_optim_state_dict = get_weights(state_dict_rec, self.network_type.value, WeightType.OPTIMIZER.value)
        self.net.load_state_dict(loaded_model_state_dict)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.model_config["learning_rate"])
        self.optimizer.load_state_dict(loaded_optim_state_dict)

    def fit(self, parameters, config):
        curr_round = config["round"]
        embeddings_rec = self.client_state.array_records["embeddings"]
        state_dict_rec = self.client_state.array_records["state_dicts"]

        self.net.train()

        if curr_round != 1:
            # Process the gradients of the previous round to update the model
            gradients = get_gradients_from_parameters(config, self.network_type)

            # TODO: we can retrieve the old gradient correctly, but calling backward on it does not work
            # thus we recalculate the embedding, but it is not efficient.
            # old_embeddings = embeddings_rec[f"embedding_{str(self.network_type)}"]
            # old_embeddings = torch.tensor(old_embeddings.numpy(), requires_grad=True)
            # old_embeddings.backward(torch.tensor(gradients))

            # Recalculate the embeddings for the old batch
            old_batch_idx = embeddings_rec[f"batch_idx_{str(self.network_type)}"].numpy()
            old_embeddings_recalculated = self.net(self.train_dataset[old_batch_idx])

            # Update the model with the gradients
            old_embeddings_recalculated.backward(torch.tensor(gradients))
            self.optimizer.step()
            self.optimizer.zero_grad()
            store_weights(self.net.state_dict(), self.network_type.value, WeightType.MODEL.value, state_dict_rec)
            store_weights(
                self.optimizer.state_dict(), self.network_type.value, WeightType.OPTIMIZER.value, state_dict_rec
            )

        # process the new batch
        batch_idx = config["batch_idx"].split(",")
        batch_idx = [int(idx) for idx in batch_idx]
        batch = self.train_dataset[batch_idx]

        # Store and return the embedding for the current batch
        embedding = self.net(batch)
        embedding_numpy = embedding.cpu().detach().numpy()
        embeddings_rec[f"embedding_{str(self.network_type)}"] = Array(embedding_numpy)
        embeddings_rec[f"batch_idx_{str(self.network_type)}"] = Array(np.array(batch_idx))

        embedding_bytes = pickle.dumps(embedding.cpu().detach())
        metrics_dict = {"type": self.network_type.value, "embedding": embedding_bytes}

        # returning fake params, since we don't communicate the model in vertical FL
        fake_params = [np.array([0.0]), np.array([0.0])]
        return fake_params, 1, metrics_dict

    def evaluate(self, parameters, config):
        # Create embeddings for the validation dataset
        self.net.eval()
        embeddings = self.net(self.val_dataset)
        embedding_bytes = pickle.dumps(embeddings.cpu().detach())

        # Pack the metrics dictionary with the embedding and return it
        metrics_dict = {"type": self.network_type.value, "embedding": embedding_bytes}
        # returning a fake loss, since the real loss is calculated in the server
        return 0.0, 1, metrics_dict


def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]

    model_config = get_model_config(context)

    if partition_id == NetworkType.PERSONAL.value:
        network_type = NetworkType.PERSONAL
        train_dataset, test_dataset = get_personal_data()
        input_size = train_dataset.shape[1]
        net = PersonalNetwork(input_size, model_config)

    elif partition_id == NetworkType.CLINICAL.value:
        network_type = NetworkType.CLINICAL
        train_dataset, test_dataset = get_clinical_data()
        input_size = train_dataset.shape[1]
        net = ClinicalNetwork(input_size, model_config)
    else:
        raise ValueError(
            f"Invalid partition_id: {partition_id}. Must be {NetworkType.PERSONAL.value} (PERSONAL) or {NetworkType.CLINICAL.value} (CLINICAL)."
        )

    # Return Client instance
    return FlowerClient(context, net, model_config, train_dataset, test_dataset, network_type=network_type).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)

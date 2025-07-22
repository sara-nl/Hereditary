"""clef-vertical-messaging-api: A Flower / PyTorch app."""

import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from clef_vertical_messaging_api.data import get_clinical_data, get_labels, get_personal_data
from clef_vertical_messaging_api.models import ClinicalNetwork, CombinedNetwork, PersonalNetwork
from clef_vertical_messaging_api.network_types import NetworkType, WeightType
from clef_vertical_messaging_api.utils import get_model_config


def load_client_data(network_type):
    """Load data based on network type."""
    if network_type == NetworkType.PERSONAL:
        train_dataset, test_dataset = get_personal_data()
    elif network_type == NetworkType.CLINICAL:
        train_dataset, test_dataset = get_clinical_data()
    else:
        raise ValueError(f"Invalid network_type: {network_type}")

    return train_dataset, test_dataset


def create_client_model(network_type, input_size, model_config):
    """Create model based on network type."""
    if network_type == NetworkType.PERSONAL:
        return PersonalNetwork(input_size, model_config)
    elif network_type == NetworkType.CLINICAL:
        return ClinicalNetwork(input_size, model_config)
    else:
        raise ValueError(f"Invalid network_type: {network_type}")


def serialize_gradients(gradients):
    """Serialize gradients for transmission."""
    return pickle.dumps(gradients)


def deserialize_gradients(gradients_bytes):
    """Deserialize gradients from transmission."""
    return pickle.loads(gradients_bytes)


def serialize_embedding(embedding):
    """Serialize embedding for transmission."""
    return pickle.dumps(embedding.cpu().detach())


def deserialize_embedding(embedding_bytes):
    """Deserialize embedding from transmission."""
    return pickle.loads(embedding_bytes)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

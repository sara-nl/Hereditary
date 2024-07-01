from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

import cifar
import flwr as fl
from flwr.common import Metrics


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 128,
        "current_round": server_round,
        "local_epochs": 2,
    }
    return config



def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}



class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}




# #create a strategy for FedAdam
# def fedadam_strategy(client):
#     return fl.server.strategy.FedAdam(
#         initial_parameters=fl.common.GetParametersIns(client).,
#         fraction_fit=0.5,
#         fraction_evaluate=0.5,
#         min_fit_clients=2,
#         min_evaluate_clients=2,
#         min_available_clients=2,
#         on_fit_config_fn=fit_config,
#         on_evaluate_config_fn=None,
#         evaluate_metrics_aggregation_fn=weighted_average,
#     )


def main() -> None:
    """Load data, start CifarClient."""

    # Load model and data
    model = cifar.Net()
    model.to(DEVICE)
    trainloader, testloader, num_examples = cifar.load_data()

    # Start client
    client = CifarClient(model, trainloader, testloader, num_examples)

    fl.client.start_client(server_address="84.85.31.118:8080", client=client.to_client())


if __name__ == "__main__":
    main()
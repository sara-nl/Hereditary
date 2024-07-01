from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

from strategies import fedavg_strategy, fedadam_strategy

import cifar
import torch


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":


    # Define strategy
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

    fl.server.start_server(
        server_address="localhost:8080", 
        config=fl.server.ServerConfig(num_rounds=1), 
        strategy=strategy)

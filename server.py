from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}



# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=4,
    min_evaluate_clients=2,
    min_available_clients=1,
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)


fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=5), strategy=strategy)

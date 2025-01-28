"""FETS-swin-unet: A Flower / PyTorch app."""

import os
from collections import OrderedDict
from typing import Optional, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import Context, Parameters, Scalar, ndarrays_to_parameters
from flwr.common.typing import FitRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from monai.networks.nets import SwinUNETR

from fets_swin_unet.task import get_weights


def evaluate_metrics_aggregation(eval_metrics):
    """Aggregate evaluation metrics from multiple clients."""
    total_examples = sum(num_examples for num_examples, _ in eval_metrics)
    weighted_metrics = {}

    # Get all metric keys from the first client's metrics
    if eval_metrics and len(eval_metrics[0]) > 1:
        metric_keys = eval_metrics[0][1].keys()

        # Calculate weighted average for each metric
        for key in metric_keys:
            weighted_sum = sum(num_examples * m[key] for num_examples, m in eval_metrics)
            weighted_metrics[key] = weighted_sum / total_examples

    return weighted_metrics


def fit_metrics_aggregation(fit_metrics):
    """Aggregate training metrics from multiple clients.

    fit_metrics structure:
    [(num_examples, {'train_loss': value}), ...]
    """
    # Calculate weighted average of train_loss
    total_examples = sum(num_examples for num_examples, _ in fit_metrics)
    weighted_loss = sum(num_examples * m["train_loss"] for num_examples, m in fit_metrics)

    return {"train_loss": weighted_loss / total_examples}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, roi, run_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize model with roi parameter
        self.net = SwinUNETR(
            img_size=roi,
            in_channels=4,
            out_channels=3,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
        # Create directory for this run
        self.save_dir = f"models/run_{run_id}"
        os.makedirs(self.save_dir, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            params_dict = zip(self.net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)

            # Save the model to disk in run-specific directory
            save_path = os.path.join(self.save_dir, f"model_round_{server_round}.pth")
            torch.save(self.net.state_dict(), save_path)

        return aggregated_parameters, aggregated_metrics


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    roi = [int(i) for i in context.run_config["roi"].split(",")]
    run_id = context.run_id

    # Initialize model
    net = SwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    )

    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        roi=roi,
        run_id=run_id,  # Pass run_id to SaveModelStrategy
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

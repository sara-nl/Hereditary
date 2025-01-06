"""xgboost-CLEF: A Flower / XGBoost app applied to the CLEF dataset."""

from logging import INFO
from typing import Dict, List, Optional

import xgboost as xgb
from xgboost_comprehensive.task import replace_keys

from flwr.common import Context, Parameters, Scalar
from flwr.common.config import unflatten_dict
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic


class CyclicClientManager(SimpleClientManager):
    """Provides a cyclic client selection rule."""

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""

        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # Return all available clients
        return [self.clients[cid] for cid in available_cids]


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (RMSE) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    rmse_aggregated = (
        sum([metrics["RMSE"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"RMSE": rmse_aggregated}
    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def server_fn(context: Context):
    # Read from config
    cfg = replace_keys(unflatten_dict(context.run_config))
    num_rounds = cfg["num_server_rounds"]
    fraction_fit = cfg["fraction_fit"]
    fraction_evaluate = cfg["fraction_evaluate"]
    train_method = cfg["train_method"]
    params = cfg["params"]

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Define strategy
    if train_method == "bagging":
        # Bagging training
        strategy = FedXgbBagging(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            initial_parameters=parameters,
        )
    else:
        # Cyclic training
        strategy = FedXgbCyclic(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            initial_parameters=parameters,
        )

    config = ServerConfig(num_rounds=num_rounds)
    client_manager = CyclicClientManager() if train_method == "cyclic" else None

    return ServerAppComponents(
        strategy=strategy, config=config, client_manager=client_manager
    )


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)

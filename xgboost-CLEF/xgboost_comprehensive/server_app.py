"""xgboost-CLEF: A Flower / XGBoost app applied to the CLEF dataset."""

from logging import INFO, WARNING
from typing import Dict, List, Optional, Union, cast

from xgboost_comprehensive.task import replace_keys

from flwr.common import Context, Parameters, Scalar, FitRes, EvaluateRes
from flwr.common.config import unflatten_dict
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
from flwr.server.strategy.fedxgb_bagging import aggregate
from tensorboardX import SummaryWriter


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
            available_cids = [cid for cid in available_cids if criterion.select(self.clients[cid])]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients" " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # Return all available clients
        return [self.clients[cid] for cid in available_cids]


class MetricsLogger:
    """Handles logging of metrics to TensorBoard."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.writer = SummaryWriter(f"runs/run_{run_id}")

    def log_metrics(self, metrics_type: str, metrics_value: float, round_num: int):
        """Log metrics to TensorBoard."""
        self.writer.add_scalar(f"{metrics_type}/rmse", metrics_value, round_num)
        self.writer.flush()


class CustomFedXgbBagging(FedXgbBagging):
    """Custom FedXgbBagging strategy that includes metrics aggregation."""

    def __init__(self, *args, run_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_logger = MetricsLogger(run_id)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using bagging."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate all the client trees
        global_model = self.global_model
        for _, fit_res in results:
            update = fit_res.parameters.tensors
            for bst in update:
                global_model = aggregate(global_model, bst)

        self.global_model = global_model

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

            # Log training metrics if available
            if "RMSE" in metrics_aggregated:
                self.metrics_logger.log_metrics("train", metrics_aggregated["RMSE"], server_round)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return (
            Parameters(tensor_type="", tensors=[cast(bytes, global_model)]),
            metrics_aggregated,
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ):
        code, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        if "RMSE" in aggregated_metrics:
            self.metrics_logger.log_metrics("eval", aggregated_metrics["RMSE"], server_round)
        return code, aggregated_metrics


class CustomFedXgbCyclic(FedXgbCyclic):
    """Custom FedXgbCyclic strategy that includes metrics aggregation."""

    def __init__(self, *args, run_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_logger = MetricsLogger(run_id)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ):
        code, _ = super().aggregate_fit(server_round, results, failures)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

            # Log training metrics if available
            if "RMSE" in metrics_aggregated:
                self.metrics_logger.log_metrics("train", metrics_aggregated["RMSE"], server_round)

        return code, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ):
        code, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        if "RMSE" in aggregated_metrics:
            self.metrics_logger.log_metrics("eval", aggregated_metrics["RMSE"], server_round)
        return code, aggregated_metrics


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (RMSE) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    rmse_aggregated = sum([metrics["RMSE"] * num for num, metrics in eval_metrics]) / total_num
    metrics_aggregated = {"RMSE": rmse_aggregated}

    # Get the round number from the first client's metrics
    if eval_metrics and len(eval_metrics) > 0:
        metrics = eval_metrics[0][1]
        if "round" in metrics:
            metrics_aggregated["round"] = metrics["round"]

    return metrics_aggregated


def fit_metrics_aggregation(fit_metrics):
    """Return an aggregated metric (RMSE) for training."""
    total_num = sum([num for num, _ in fit_metrics])
    rmse_aggregated = sum([metrics["RMSE"] * num for num, metrics in fit_metrics]) / total_num
    metrics_aggregated = {"RMSE": rmse_aggregated}

    # Get the round number from the first client's metrics
    if fit_metrics and len(fit_metrics) > 0:
        metrics = fit_metrics[0][1]
        if "round" in metrics:
            metrics_aggregated["round"] = metrics["round"]

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
    run_id = context.run_id

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Define strategy
    if train_method == "bagging":
        # Bagging training
        strategy = CustomFedXgbBagging(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            initial_parameters=parameters,
            run_id=run_id,
        )
    else:
        # Cyclic training
        strategy = CustomFedXgbCyclic(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            initial_parameters=parameters,
            run_id=run_id,
        )

    config = ServerConfig(num_rounds=num_rounds)
    client_manager = CyclicClientManager() if train_method == "cyclic" else None

    return ServerAppComponents(strategy=strategy, config=config, client_manager=client_manager)


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)

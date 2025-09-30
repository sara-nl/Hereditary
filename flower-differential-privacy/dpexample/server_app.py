"""dpexample: A Flower with differential privacy app."""

from logging import DEBUG
from typing import List, Tuple

from dpexample.task import (
    get_weights,
    make_net,
    set_weights,
    test,
    load_cifar10_test,
)

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.logger import update_console_handler
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg, DifferentialPrivacyClientSideFixedClipping
from flwr.server.workflow import DefaultWorkflow
import torch
import math
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import Optional

# Global TensorBoard writer
TB_WRITER: Optional[SummaryWriter] = None


def on_fit_config(server_round: int, config: dict):
    """Return training configuration dict for each round with cosine annealing learning rate.
    
    The learning rate follows a cosine annealing schedule from the initial learning rate
    (from config) down to 1% of the initial learning rate over the course of training.
    """
    # Get initial learning rate from run_config, default to 0.01 if not found
    initial_lr = float(config.get("learning-rate", 0.01))
    max_rounds = int(config.get("num-server-rounds", 10))
    
    # Calculate current progress through training (0.0 to 1.0)
    progress = min(server_round / max_rounds, 1.0)
    
    # Cosine annealing schedule
    # eta_min = 1% of initial learning rate
    eta_min = initial_lr * 0.01
    
    # Cosine annealing formula
    lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(math.pi * progress))

    if server_round < 10:
        data_percentage = 1
    elif server_round < 15:
        data_percentage = 0.5
    else:
        data_percentage = 0.25
    data_percentage = 1

    # Log LR if writer is available
    if TB_WRITER is not None:
        TB_WRITER.add_scalar("server/lr", lr, server_round)
    return {"learning_rate": lr, "data_percentage": data_percentage, "server_round": server_round}

# Define metric aggregation function
def weighted_average(eval_type, metrics: List[Tuple[int, Metrics]]) -> Metrics:
    server_round = metrics[0][1].get("server_round")
    # Per-client logging (if available)
    if TB_WRITER is not None:
        for n, m in metrics:
            pid = m.get("partition_id")
            if pid is None:
                continue
            if "accuracy" in m:
                TB_WRITER.add_scalar(f"clients/part-{pid}/{eval_type}/accuracy", m["accuracy"], server_round)
            if "loss" in m:
                TB_WRITER.add_scalar(f"clients/part-{pid}/{eval_type}/loss", m["loss"], server_round)

    total = sum(n for n, _ in metrics)
    if total == 0:
        return {"accuracy": 0.0}

    w_acc = sum(n * m.get("accuracy", 0.0) for n, m in metrics) / total

    # Log weighted metrics
    if TB_WRITER is not None:
        TB_WRITER.add_scalar(f"server/{eval_type}/weighted_accuracy", w_acc, server_round)
        if any(m.get("loss") is not None for _, m in metrics):
            w_loss = sum(n * m.get("loss", 0.0) for n, m in metrics) / total
            TB_WRITER.add_scalar(f"server/{eval_type}/weighted_loss", w_loss, server_round)

    print(f"{eval_type} weighted client acc: ", w_acc)
    return {"accuracy": w_acc}

# Server-side evaluation will be defined inside main() to capture testloader
# Flower ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # TensorBoard writer (global)
    global TB_WRITER
    run_name = context.run_config.get("tensorboard-run-name", "")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{run_name}_{timestamp}" if run_name else f"runs/dp_server_{timestamp}"
    TB_WRITER = SummaryWriter(log_dir=log_dir)

    # Get initial parameters
    ndarrays = get_weights(make_net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Build global test DataLoader for server-side evaluation
    # Use the run_config batch size if available; otherwise default to 128
    test_batch_size = context.run_config.get("batch-size", 128)
    testloader = load_cifar10_test(batch_size=test_batch_size, data_dir="data")

    # Define evaluation function that uses the global CIFAR-10 test set
    def evaluate_agg(server_round, parameters_ndarrays, config):
        net = make_net()
        set_weights(net, parameters_ndarrays)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss, acc = test(net, testloader, device)
        print("server-side eval acc: ", acc, "round: ", server_round)
        # Server-side eval metrics
        if TB_WRITER is not None:
            TB_WRITER.add_scalar("server/eval/accuracy", acc, server_round)
            TB_WRITER.add_scalar("server/eval/loss", loss, server_round)
        return loss, {"accuracy": acc}

    # Define strategy (standard FedAvg, logging happens in weighted_average/evaluate_agg/on_fit_config)
    strategy = FedAvg(
        # Select all available clients
        fraction_fit=1.0,
        min_fit_clients=5,
        # Disable evaluation in demo
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=lambda metrics: weighted_average("evaluate", metrics),
        fit_metrics_aggregation_fn=lambda metrics: weighted_average("fit", metrics),
        initial_parameters=parameters,
        evaluate_fn=evaluate_agg,
        on_fit_config_fn=lambda server_round: on_fit_config(server_round, context.run_config),
    )

    # Wrap the strategy with the DifferentialPrivacyClientSideFixedClipping wrapper
    dp_strategy = DifferentialPrivacyClientSideFixedClipping(
        strategy,
        context.run_config["noise-multiplier"],
        context.run_config["clipping-norm"],
        context.run_config["num-partitions"],
    )

    # Construct the LegacyContext
    num_rounds = context.run_config["num-server-rounds"]
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=dp_strategy,
    )


    # Create the workflow
    workflow = DefaultWorkflow()

    # Execute
    workflow(grid, context)
    if TB_WRITER is not None:
        TB_WRITER.flush()
        TB_WRITER.close()

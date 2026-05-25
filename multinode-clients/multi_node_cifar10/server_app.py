"""Federated CIFAR-10 server – coordinates federated averaging across clients.

The server initialises the global model from a checkpoint file, distributes
weights to clients, and aggregates the returned updates with FedAvg.
"""

import logging
import os
import sys

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from safetensors.torch import load_file as safe_load

logger = logging.getLogger(__name__)


def _get_initial_state_dict() -> dict:
    """Create a fresh ResNet-18 state_dict for shared initialization."""
    cifar_dir = os.path.join(os.path.dirname(__file__), "..", "cifar10")
    if cifar_dir not in sys.path:
        sys.path.insert(0, cifar_dir)
    from model import resnet18

    model = resnet18()
    return model.state_dict()

app = ServerApp()


def _load_initial_checkpoint(ckpt_path: str) -> dict:
    """Load the initial model weights from a .pt or .safetensors file.

    Handles both raw state_dict and ``{"model_state_dict": ...}`` wrapper format.
    If no checkpoint is provided, creates a fresh ResNet-18 so all clients
    share the same initial weights.

    Args:
        ckpt_path: Path to a checkpoint file.

    Returns:
        A state_dict.
    """
    if not ckpt_path or not os.path.isfile(ckpt_path):
        logger.info("No initial checkpoint provided. Initializing fresh ResNet-18.")
        return _get_initial_state_dict()

    if ckpt_path.endswith(".safetensors"):
        state_dict = safe_load(ckpt_path)
    else:
        full_ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = full_ckpt.get("model_state_dict", full_ckpt)

    logger.info("Loaded initial weights from %s (%d keys)", ckpt_path, len(state_dict))

    for k, v in state_dict.items():
        if v.dtype == torch.bfloat16:
            state_dict[k] = v.to(dtype=torch.float32)

    return state_dict


class SaveModelStrategy(FedAvg):
    """Custom FedAvg strategy that saves the global model after every round."""

    def __init__(self, output_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir

    def aggregate_train(self, server_round: int, replies):
        """Aggregate training results and save the global model."""
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is not None:
            # Save weights for this round
            round_dir = os.path.join(self.output_dir, f"round_{server_round}")
            os.makedirs(round_dir, exist_ok=True)
            output_path = os.path.join(round_dir, "global_model.pt")
            state_dict = arrays.to_torch_state_dict()
            torch.save(state_dict, output_path)
            logger.info(
                "Saved global model for round %d to %s", server_round, output_path
            )
        return arrays, metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    num_rounds: int = int(context.run_config["num-server-rounds"])
    fraction_evaluate: float = float(context.run_config.get("fraction-evaluate", 0.0))
    initial_checkpoint_path: str = str(context.run_config.get("initial-checkpoint", ""))

    # --- Setup output directory ---
    experiment_name = str(
        context.run_config.get(
            "experiment-name", os.environ.get("EXPERIMENT_NAME", "fl_cifar10_training")
        )
    )
    base_dir = os.environ.get("OUTPUT_DIR_BASE", os.path.abspath("output"))
    output_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set up file logging
    log_path = os.path.join(output_dir, "server.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    initial_state_dict = _load_initial_checkpoint(initial_checkpoint_path)
    arrays = ArrayRecord(initial_state_dict)

    strategy = SaveModelStrategy(
        output_dir=output_dir,
        fraction_evaluate=fraction_evaluate,
    )

    train_config = ConfigRecord({})

    logger.info("Starting %d federated rounds", num_rounds)
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
    )

    # Save final global model
    output_filename = str(
        context.run_config.get("output-checkpoint", "final_global_model.pt")
    )
    output_path = os.path.join(output_dir, output_filename)
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, output_path)
    logger.info("Saved final global model to %s (%d keys)", output_path, len(state_dict))

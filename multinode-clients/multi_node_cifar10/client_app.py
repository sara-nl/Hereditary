"""Federated CIFAR-10 client – launches multi-node training via sbatch.

The Flower client receives global model weights from the server, saves them
to disk, kicks off a distributed training run via sbatch, and returns the
updated weights.
"""

import logging
import os

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from multi_node_cifar10.task import (
    launch_cifar_training,
    launch_dummy_training,
    load_updated_weights,
    save_global_weights,
)

logger = logging.getLogger(__name__)

app = ClientApp()


def train_slurm(
    msg: Message,
    context: Context,
    global_weights_path: str,
    output_dir: str,
    data_dir: str,
) -> Message:
    """Submit CIFAR training via sbatch ➜ return updated weights."""

    epochs = int(context.run_config.get("epochs", 3))
    lr = float(context.run_config.get("lr", 0.001))
    batch_size = int(context.run_config.get("batch-size", 128))
    max_batches = int(context.run_config.get("max-batches-per-epoch", 32))
    save_freq = int(context.run_config.get("save-freq", 5))
    data_root_config = context.run_config.get("data-root")
    data_root = os.path.abspath(str(data_root_config)) if data_root_config else None

    logger.info(
        "Starting multi-node CIFAR training: %d epochs, batch_size=%d",
        epochs, batch_size,
    )

    final_ckpt_path = launch_cifar_training(
        output_dir=output_dir,
        resume_path=global_weights_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        max_batches_per_epoch=max_batches,
        data_root=data_dir,
        save_freq=save_freq,
    )

    updated_state_dict = load_updated_weights(final_ckpt_path)

    model_record = ArrayRecord(updated_state_dict)
    metrics = MetricRecord({
        "num-examples": 1.0,
        "num-epochs": float(epochs),
        "batch-size": float(batch_size),
    })
    content = RecordDict({"arrays": model_record, "metrics": metrics})

    logger.info("Training complete – returning updated weights to server.")
    return Message(content=content, reply_to=msg)


def train_dummy(
    msg: Message,
    context: Context,
    global_weights_path: str,
    output_dir: str,
) -> Message:
    """Execute dummy python process saving safetensors ➜ return updated weights."""

    logger.info("Starting local dummy training...")

    final_ckpt_path = launch_dummy_training(
        global_weights_path=global_weights_path,
        output_dir=output_dir,
    )

    updated_state_dict = load_updated_weights(final_ckpt_path)

    model_record = ArrayRecord(updated_state_dict)
    metrics = MetricRecord({
        "num-examples": 1.0,
        "num-train-steps": 0.0,
        "is_dummy": 1.0,
    })
    content = RecordDict({"arrays": model_record, "metrics": metrics})

    logger.info("Dummy complete – returning simulated weights to server.")
    return Message(content=content, reply_to=msg)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Router hook that delegates to train_dummy or train_slurm."""
    dummy_run = context.run_config.get("dummy-run", False)

    experiment_name = str(context.run_config.get(
        "experiment-name", os.environ.get("EXPERIMENT_NAME", "fl_cifar10_training")
    ))

    server_round = msg.content["config"]["server-round"]
    partition_id = context.node_config["partition-id"]

    if dummy_run:
        base_dir = os.environ.get("WORKSPACE_BIND", os.path.abspath("output"))
    else:
        base_dir = os.environ.get("OUTPUT_DIR_BASE", os.path.abspath("output"))
    output_dir = os.path.join(base_dir, experiment_name, f"round_{server_round}", f"client_{partition_id}")
    data_dir = os.path.join(base_dir, experiment_name, f"client_{partition_id}", "data")

    global_state_dict = msg.content["arrays"].to_torch_state_dict()

    if global_state_dict:
        global_weights_path = os.path.join(output_dir, "global_weights.pt")
        save_global_weights(global_state_dict, global_weights_path)
    else:
        global_weights_path = ""

    if dummy_run:
        logger.info("Detected 'dummy-run=True'. Routing to dummy test function.")
        return train_dummy(
            msg,
            context,
            global_weights_path,
            output_dir,
        )
    else:
        return train_slurm(
            msg,
            context,
            global_weights_path,
            output_dir,
            data_dir,
        )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Placeholder evaluation – returns dummy metrics."""
    metrics = MetricRecord({
        "eval_loss": 0.0,
        "eval_acc": 0.0,
        "num-examples": 0.0,
    })
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)
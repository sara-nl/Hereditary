# Federated CIFAR-10 Training

This project uses Flower to coordinate federated learning across supercomputers for a CIFAR-10 ResNet-18 model trained with FSDP.

Instead of running local training loops, each client receives global model weights from the central server, saves them to disk, and submits a full distributed training job via `sbatch` using `cifar10/slurm_train.sh`. Once training completes, the client loads the updated weights and returns them to the server for FedAvg aggregation.

## Architecture

```
Server (FedAvg)
  │  global state_dict
  ▼
Client: save_global_weights() → {"model_state_dict": state_dict}
  ▼
Client: launch_cifar_training() → sbatch slurm_train.sh
  ▼
train.py: load_checkpoint() → model_state_dict only, fresh optimizer
  ▼
train.py: save_checkpoint() → {"model_state_dict": ..., "optimizer_state_dict": ..., "epoch": ...}
  ▼
Client: load_updated_weights() → extract model_state_dict, bf16→fp32
  ▼
Server receives updated state_dict for FedAvg
```

## Installation

```bash
pip install -e .
```

## Running the clients

Each client submits its own SLURM job via `sbatch`. The SLURM script (`cifar10/slurm_train.sh`) handles node allocation, environment setup, and launching `torch.distributed.run`.

### Required Environment Variables

| Variable | Description |
|---|---|
| `OUTPUT_DIR_BASE` | Base directory for training outputs. Defaults to `./output`. |

No container or GPU-specific env vars are needed — the SLURM script handles all of that.

### Optional Run Config

Training hyperparameters are passed via `--run-config` to `flwr run`:

| Key | Default | Description |
|---|---|---|
| `epochs` | 3 | Number of training epochs per round |
| `lr` | 0.001 | Learning rate |
| `batch-size` | 128 | Per-rank batch size |
| `max-batches-per-epoch` | 32 | Batches per epoch (0 = full dataset) |
| `data-root` | `./data` | Path to CIFAR-10 data |
| `save-freq` | 5 | Checkpoint save frequency (epochs) |
| `dummy-run` | `false` | Use local dummy training instead of sbatch |
| `experiment-name` | `fl_cifar10_training` | Output folder name |

### Example Workflow

```bash
export OUTPUT_DIR_BASE="/path/to/shared/output"
flwr run . --run-config "epochs=3 lr=0.001 batch-size=128 max-batches-per-epoch=32"
```

## Running the central Server

```bash
export FL_INITIAL_CHECKPOINT="/path/to/start.pt"   # Optional
export FL_OUTPUT_CHECKPOINT="/path/to/aggregate.pt" # Where the final model gets saved
```

## Dummy Mode

For local testing without a SLURM cluster, set `dummy-run=true`:

```bash
flwr run . --run-config "dummy-run=true"
```

This runs a lightweight local training simulation that saves/loads safetensors files.

## Weight Format

- **Server ↔ Client:** Pure `state_dict` (model weights only, no optimizer state)
- **Client → SLURM job:** `{"model_state_dict": state_dict}` saved as `global_weights.pt`
- **SLURM job → Client:** `{"model_state_dict": ..., "optimizer_state_dict": ..., "epoch": ...}` saved as `checkpoint.pt`

The client extracts only `model_state_dict` from the SLURM job output before returning it to the server.

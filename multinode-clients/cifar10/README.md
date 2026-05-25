# CIFAR-10 FSDP Training

Small ResNet-18 trained on CIFAR-10 using Fully Sharded Data Parallel (FSDP). Supports both CPU and GPU.

## Quick Start

### 1. Create and activate the virtual environment

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Single-node CPU test (recommended for debugging)

```bash
python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 train.py --epochs 3 --max-batches-per-epoch 2
```

This spins up 4 CPU processes on the local node and trains for 3 epochs — enough to verify distributed setup, data loading, and checkpointing all work.

### 3. Single-GPU node

```bash
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --standalone train.py --epochs 10
```

### 4. Multi-node via SLURM

```bash
# Fresh training on 2 CPU nodes
sbatch slurm_train.sh

# Resume from a previous checkpoint
RESUME_PATH=./output/checkpoint.pt sbatch slurm_train.sh
```

## Arguments

| Flag | Default | Description |
|---|---|---|
| `--batch-size` | 128 | Per-rank batch size |
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 0.1 | Learning rate |
| `--data-root` | ./data | Path to CIFAR-10 data cache |
| `--output-dir` | ./output | Directory for saved checkpoints |
| `--save-freq` | 5 | Save checkpoint every N epochs |
| `--resume` | None | Path to `checkpoint.pt` to resume from |

## How It Works

- **Auto device detection** — uses `gloo` backend on CPU, `nccl` on GPU
- **FSDP full sharding** — model parameters are partitioned across all ranks
- **DistributedSampler** — each rank sees a disjoint subset of the data
- **Checkpointing** — rank 0 saves `checkpoint.pt` with model + optimizer state

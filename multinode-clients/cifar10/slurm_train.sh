#!/bin/bash
#SBATCH --job-name=cifar10_fsdp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

SCRIPT_DIR="/gpfs/work3/1/hpmlprjs/ESM2/esm2_github_repo/cifar10/"
source "$SCRIPT_DIR/.venv/bin/activate"
export OMP_NUM_THREADS=32
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

export MASTER_ADDR="$MASTER_NODE"
export MASTER_PORT="$MASTER_PORT"

NNODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_PROCID
NPROC_PER_NODE=1

# Set environment variables for torch.distributed env:// rendezvous
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_JOB_NUM_NODES
export LOCAL_RANK=$SLURM_LOCALID

export NCCL_DEBUG=INFO

echo "=== FSDP CIFAR-10 Training ==="
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $NNODES, Rank: $NODE_RANK, Procs/node: $NPROC_PER_NODE"
echo "Hostname: $(hostname)"

# Use torchrun with c10d rendezvous backend for SLURM multi-node
# Each node runs torchrun which coordinates via the master node
srun --ntasks=$NNODES --ntasks-per-node=$NPROC_PER_NODE \
  torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --rdzv_backend c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --node_rank=$SLURM_PROCID \
    /gpfs/work3/1/hpmlprjs/ESM2/esm2_github_repo/cifar10/train.py \
    --batch-size 128 \
    --epochs 3 \
    --lr 0.001 \
    --data-root ./data \
    --output-dir ./output \
    --save-freq 5 \
    ${RESUME_PATH:+--resume "$RESUME_PATH"} \
    --max-batches-per-epoch 32

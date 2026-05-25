#!/bin/bash
#SBATCH --job-name=cifar10_fsdp_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Unset PYTHONPATH and VIRTUAL_ENV to prevent environment leakage from the launcher (Flower client)
unset PYTHONPATH
unset VIRTUAL_ENV

SCRIPT_DIR="/projects/hpmlprjs/ESM2/esm2_github_repo/flower_cifar_multinode/cifar10/"
source "$SCRIPT_DIR/.venv/bin/activate"
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="eno"

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

export MASTER_ADDR="$MASTER_NODE"
export MASTER_PORT="$MASTER_PORT"

NNODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_PROCID
NPROC_PER_NODE=1

# Set environment variables for torch.distributed env// rendezvous
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * NPROC_PER_NODE))
export LOCAL_RANK=$SLURM_LOCALID

export DATA_DIR=/gpfs/work3/1/hpmlprjs/ESM2/esm2_github_repo/esm2_flower_cifar/data/

echo "=== FSDP CIFAR-10 Training on H100 ==="
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $NNODES, Rank: $NODE_RANK, Procs/node: $NPROC_PER_NODE"
echo "Hostname: $(hostname)"

# Use torchrun with c10d rendezvous backend for SLURM multi-node
# Each node runs torchrun which coordinates via the master node
srun --ntasks=$NNODES --ntasks-per-node=1 \
  torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --rdzv_backend c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --node_rank=$SLURM_PROCID \
    $SCRIPT_DIR/train.py \
    --batch-size ${BATCH_SIZE:-128} \
    --epochs ${EPOCHS:-3} \
    --lr ${LR:-0.001} \
    --data-root ${DATA_ROOT:-$DATA_DIR} \
    --output-dir ${OUTPUT_DIR:-./output} \
    --save-freq ${SAVE_FREQ:-5} \
    ${RESUME_PATH:+--resume "$RESUME_PATH"} \
    --max-batches-per-epoch ${MAX_BATCHES:-32}

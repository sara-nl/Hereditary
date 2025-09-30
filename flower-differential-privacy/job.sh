#!/bin/sh
# Define a regular job over 14 nodes. We need to use all GPUs on both nodes, so we set gpus-per-node to 4. SLURM does not allow for less.
#SBATCH --job-name=test     # Job name, also used for wandb
#SBATCH --partition=gpu_a100                    # Partition name
#SBATCH --constraint=scratch-node
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node (typically set to 4 per node when using GPUs)
#SBATCH --gpus-per-node=1                      # Number of GPUs per node
#SBATCH --time=02:00:00                         # Walltime
source ~/git/Hereditary/FETS_venv/bin/activate
cd /home/douwew/git/hereditary-ws5/flower-secure-aggregation/
flwr run .
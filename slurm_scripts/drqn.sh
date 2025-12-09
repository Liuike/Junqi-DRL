#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH -p gpu --gres=gpu:1   
#SBATCH --constraint=geforce3090

# Load Python module

module load python
# export PYTHONPATH=$(pwd):$PYTHONPATH
# source .venv/bin/activate

uv run python -u scripts/train_from_config.py configs/drqn_train.yaml
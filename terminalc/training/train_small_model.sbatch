#!/bin/bash
#SBATCH --job-name=small-lm-train
#SBATCH --partition=condo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/018219422/terminalC
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate terminalc

python terminalc/training/train_small_model.py --epochs 100

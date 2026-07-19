#!/bin/bash
#SBATCH --job-name=tf_abl
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# One training+generation job for a single (variant, dataset).
# Seed is fixed to 42. Scale is --train_sample 500 --sample 500.
#
# Usage (submitted by submit_all.sh; can also run standalone):
#   sbatch --job-name=tf_E3_xsum run_one.sbatch E3 xsum
#
set -euo pipefail

EXP="$1"        # E0 .. E5
DATASET="$2"    # xsum | cnn_dailymail

mkdir -p logs
module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transformer-abl

# Force offline so a compute node never blocks trying to reach the HF hub;
# prefetch.sh must have populated the cache on a login node first.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi   # confirm a GPU was actually allocated

python "Transformer/${EXP}.py" --task all \
    --dataset "$DATASET" \
    --train_sample 500 --sample 500 \
    --seed 42 \
    --device cuda \
    --overwrite
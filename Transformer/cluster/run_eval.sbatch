#!/bin/bash
#SBATCH --job-name=tf_eval
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Evaluate one variant's summaries directory (BERTScore runs on GPU).
#
# Usage:  sbatch --job-name=tf_eval_E3 run_eval.sbatch E3
#
set -euo pipefail

EXP="$1"        # E0 .. E5

mkdir -p logs
module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transformer-abl

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi

# Each variant reads its own summaries/EX and writes results/EX (matches the
# per-variant isolation the code already does via --task evaluate defaults).
python "Transformer/${EXP}.py" --task evaluate
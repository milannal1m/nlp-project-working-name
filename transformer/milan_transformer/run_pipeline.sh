#!/bin/bash
#SBATCH --job-name=milan_transformer
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --time=04:00:00
#SBATCH --output=transformer/milan_transformer/logs/%x_%j.out
#SBATCH --error=transformer/milan_transformer/logs/%x_%j.err
#
# Trains ONE from-scratch summarization transformer (encoder-decoder OR decoder-only)
# on XSum + CNN/DailyMail and evaluates it with the repo's full Evaluator. It's a small
# (~17-27M param) model, so a single GPU with modest CPU/RAM and a few hours is plenty.
#
# The run size (TRAIN_N / EVAL_N / EPOCHS / BATCH) is set by the constants in pipeline.py.
#
# Submit from the repo root (so 'transformer/...' and the log dir resolve):
#   sbatch transformer/milan_transformer/run_pipeline.sh [-m MODEL]
#
#   -m, --model   lecture (encoder-decoder) | milan (decoder-only)   (default: milan)

set -euo pipefail

MODEL="milan"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "Model: $MODEL"

mkdir -p transformer/milan_transformer/logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-project

nvidia-smi

python transformer/milan_transformer/pipeline.py --model "$MODEL"

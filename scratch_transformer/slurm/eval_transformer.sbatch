#!/bin/bash
#SBATCH --job-name=sum-eval-tf
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results/metrics

PY="${NLP_PYTHON:-$HOME/.conda/envs/nlp-env/bin/python}"
[[ -x "$PY" ]] || { echo "FATAL: python interpreter not found at $PY" >&2; exit 1; }

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[$(date)] tf eval on $(hostname)"
"$PY" -m scratch_transformer.evaluate
echo "[$(date)] done."

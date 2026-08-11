#!/bin/bash
#SBATCH --job-name=sum-train-tf
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs scratch_transformer/checkpoints

PY="${NLP_PYTHON:-$HOME/.conda/envs/nlp-env/bin/python}"
[[ -x "$PY" ]] || { echo "FATAL: python interpreter not found at $PY" >&2; exit 1; }

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[$(date)] train task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -m scratch_transformer.train --dataset_index "${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] done."

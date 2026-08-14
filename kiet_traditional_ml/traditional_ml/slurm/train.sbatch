#!/bin/bash
#SBATCH --job-name=tml-train
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180000
#SBATCH --time=24:00:00
#SBATCH --output=traditional_ml/logs/%x_%A_%a.out
#SBATCH --error=traditional_ml/logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p traditional_ml/logs

PY="${TML_PYTHON:-$HOME/.conda/envs/nlp-env/bin/python}"

export TOKENIZERS_PARALLELISM=false

echo "[$(date)] tml train task ${SLURM_ARRAY_TASK_ID} on $(hostname) [$PY]"
"$PY" -m traditional_ml.train
echo "[$(date)] done."

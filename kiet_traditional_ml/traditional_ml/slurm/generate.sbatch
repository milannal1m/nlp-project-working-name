#!/bin/bash
#SBATCH --job-name=tml-gen
#SBATCH --partition=cpu
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=08:00:00
#SBATCH --output=traditional_ml/logs/%x_%A_%a.out
#SBATCH --error=traditional_ml/logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p traditional_ml/logs

PY="${TML_PYTHON:-$HOME/.conda/envs/nlp-env/bin/python}"

DATASETS=(cnn_dailymail xsum)
DS="${DATASETS[${SLURM_ARRAY_TASK_ID}]}"

echo "[$(date)] tml generate ${DS} on $(hostname) [$PY]"
"$PY" -m traditional_ml.generate --datasets "${DS}"
echo "[$(date)] done."

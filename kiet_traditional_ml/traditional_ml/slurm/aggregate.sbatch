#!/bin/bash
#SBATCH --job-name=tml-agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --time=00:20:00
#SBATCH --output=traditional_ml/logs/%x_%j.out
#SBATCH --error=traditional_ml/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p traditional_ml/logs

PY="${TML_PYTHON:-$HOME/.conda/envs/nlp-env/bin/python}"

echo "[$(date)] tml aggregate on $(hostname) [$PY]"
"$PY" -m traditional_ml.aggregate
echo "[$(date)] done."

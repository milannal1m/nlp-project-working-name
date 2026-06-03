#!/bin/bash
#SBATCH --job-name=sum-aggregate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Aggregation — collect every results/metrics/*.json into:
#   results/results.csv, results/results.md, results/charts/*.png
# CPU only.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-env

echo "[$(date)] aggregating on $(hostname)"
python aggregate.py
echo "[$(date)] done. See results/results.md"

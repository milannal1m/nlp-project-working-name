#!/bin/bash
#SBATCH --job-name=sum-gen-base
#SBATCH --partition=cpu
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Baseline generation (Lead-1, Lead-3, TextRank, TF-IDF) — CPU only.
# Array indices 0-3 map to pipeline_config.MODELS[0..3]. Idempotent: skips any
# config whose summaries already exist with >= SAMPLE records.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-env

export TOKENIZERS_PARALLELISM=false

echo "[$(date)] baseline gen task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
python run_generation.py --index "${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] done."

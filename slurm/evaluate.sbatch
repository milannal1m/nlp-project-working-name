#!/bin/bash
#SBATCH --job-name=sum-eval
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Evaluation — one GPU per array task, all running in parallel.
# Array indices 0-19 map to pipeline_config.EVAL_TARGETS (10 models x 2 datasets).
# Each task scores the first SAMPLE records of one summary file and writes
# results/metrics/<label>__<dataset>.json. Metrics: BLEU, ROUGE-L, METEOR,
# BERTScore-F1, SummaC, QAFactEval (ROUGE-1/2 excluded). BERTScore and SummaC
# need a GPU; QAFactEval is best-effort and skipped (null) if not installed.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-env

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[$(date)] eval task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
python run_evaluation.py --index "${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] done."

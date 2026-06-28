#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Evaluation worker: scores every .jsonl in the summaries dir (BLEU, ROUGE-L,
# METEOR, BERTScore, optional QAFactEval) sequentially. Writes
# results/evaluation.{log,csv}. BERTScore runs on GPU, hence the GPU partition.
# Full grid (~44 files) is BERTScore-bound at ~20 min/file -> ~15-17h, so 24h
# gives headroom.
#
# Usage (run from the repo root so 'src/main.py' resolves):
#   sbatch scripts/run_evaluation.sh [--output-dir DIR] [--log-path PATH] [--append]

set -euo pipefail

OUTPUT_DIR="summaries"
LOG_PATH="results/evaluation.log"
APPEND=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --log-path)   LOG_PATH="$2";   shift 2 ;;
        --append)     APPEND="--append_eval"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "Output dir: $OUTPUT_DIR"
echo "Log path:   $LOG_PATH"

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-project

nvidia-smi

python src/main.py \
    --task evaluate \
    --output_dir "$OUTPUT_DIR" \
    --log_path "$LOG_PATH" \
    $APPEND

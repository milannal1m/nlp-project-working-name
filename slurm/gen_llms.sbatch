#!/bin/bash
#SBATCH --job-name=sum-gen-llm
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --array=4-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# LLM generation — one GPU per array task, all running in parallel.
# Array indices 4-9 map to pipeline_config.MODELS[4..9]:
#   4 Llama_None  5 Llama_4bit  6 Llama_8bit
#   7 Phi-3_None  8 Phi-3_4bit  9 Phi-3_8bit
# Idempotent: a config whose summaries already exist (>= SAMPLE) is skipped,
# so the committed Llama_4bit full-test-set file is reused, not regenerated.
#
# Memory note: Llama-3.2-3B / Phi-3-mini in fp16 need ~7-8 GB VRAM; 4bit/8bit
# far less. An A100 (40/80 GB) has ample headroom — no OOM risk at these sizes.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-env

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[$(date)] llm gen task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python run_generation.py --index "${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] done."

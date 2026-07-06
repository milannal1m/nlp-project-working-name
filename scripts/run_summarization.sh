#!/bin/bash
#SBATCH --job-name=summarize
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16000
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Summarization worker: runs ONE (model, quant, prompt) over the given datasets.
# run_experiment.sh submits one job PER DATASET (full test set ~11k articles
# each), so 36h covers the slow cells (P3 = 300 tokens, 8-bit) with headroom.
# Used standalone or submitted in parallel by run_experiment.sh.
#
# Usage (run from the repo root so 'src/main.py' resolves):
#   sbatch scripts/run_summarization.sh [-m MODEL] [-q QUANT] [-p PROMPT] [-d "DS1 DS2"] [-s N] [--skip-existing]
#
#   -m, --model         Model label from the registry (Llama, Phi, Qwen2, ...) (default: Llama)
#   -q, --quant         Quantization: 16bit | 8bit | 4bit                (default: 4bit)
#   -p, --prompt        Prompt name: P1 | P2 | P3                        (default: P1)
#   -d, --datasets      Space-separated dataset list                     (default: all)
#   -s, --sample        Articles per dataset                            (default: full test set)
#   -e, --seed          Random seed (reproducibility)                    (default: 42)
#       --skip-existing Skip datasets whose .jsonl already exists

set -euo pipefail

MODEL="Llama"
QUANT="4bit"
PROMPT="P1"
DATASETS=""
SAMPLE=""
SEED=""
SKIP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)      MODEL="$2";    shift 2 ;;
        -q|--quant)      QUANT="$2";    shift 2 ;;
        -p|--prompt)     PROMPT="$2";   shift 2 ;;
        -d|--datasets)   DATASETS="$2"; shift 2 ;;
        -s|--sample)     SAMPLE="$2";   shift 2 ;;
        -e|--seed)       SEED="$2";     shift 2 ;;
        --skip-existing) SKIP="--skip_existing"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "Model:    $MODEL"
echo "Quant:    $QUANT"
echo "Prompt:   $PROMPT"
echo "Datasets: ${DATASETS:-all}"
echo "Sample:   ${SAMPLE:-full test set}"
echo "Seed:     ${SEED:-42}"

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-project

nvidia-smi

# shellcheck disable=SC2086  # intentional word-splitting for the dataset list
python src/main.py \
    --task summarize \
    --model "$MODEL" \
    --quantization_method "$QUANT" \
    --prompt_name "$PROMPT" \
    ${DATASETS:+--datasets $DATASETS} \
    ${SAMPLE:+--sample "$SAMPLE"} \
    ${SEED:+--seed "$SEED"} \
    $SKIP

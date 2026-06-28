#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Baseline worker: Lead-1, Lead-3, TextRank, TF-IDF (CPU only, no GPU needed).
#
# Usage (run from the repo root so 'src/main.py' resolves):
#   sbatch scripts/run_baselines.sh [-d "DS1 DS2"] [-s N] [-e SEED] [--skip-existing]

set -euo pipefail

DATASETS=""
SAMPLE=""
SEED=""
SKIP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--datasets)   DATASETS="$2"; shift 2 ;;
        -s|--sample)     SAMPLE="$2";   shift 2 ;;
        -e|--seed)       SEED="$2";     shift 2 ;;
        --skip-existing) SKIP="--skip_existing"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "Datasets: ${DATASETS:-all}"
echo "Sample:   ${SAMPLE:-full test set}"
echo "Seed:     ${SEED:-42}"

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-project

# shellcheck disable=SC2086  # intentional word-splitting for the dataset list
python src/main.py \
    --task baselines \
    ${DATASETS:+--datasets $DATASETS} \
    ${SAMPLE:+--sample "$SAMPLE"} \
    ${SEED:+--seed "$SEED"} \
    $SKIP

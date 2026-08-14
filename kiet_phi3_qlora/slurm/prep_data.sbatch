#!/bin/bash
#SBATCH --job-name=lora-prep
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
COMMON="${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"
[[ -f "$COMMON" ]] || { echo "FATAL: cannot find $COMMON" >&2; exit 1; }
source "$COMMON"

banner "lora prep -> ${TOKENIZED_PATH}"

if [[ -d "${TOKENIZED_PATH}" ]]; then
  echo "[skip] tokenized dataset already present: ${TOKENIZED_PATH}"
  exit 0
fi

"$PY" train_lora.py \
  --prepare_only \
  --per_dataset_samples 0 \
  --max_seq_length 2048 \
  --tokenized_path "${TOKENIZED_PATH}"

if [[ ! -d "${TOKENIZED_PATH}" ]]; then
  echo "FATAL: prep finished but ${TOKENIZED_PATH} does not exist" >&2
  exit 1
fi

du -sh "${TOKENIZED_PATH}"
banner "prep done."

#!/bin/bash
#SBATCH --job-name=lora-prep
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage 1/5 — stream both FULL train splits (287,113 + 204,045 = 491,158 pairs),
# tokenize them once at seq len 2048, and cache to disk.
#
# Split out from training on purpose: it is single-threaded, network-bound work.
# Doing it here means the expensive 4-GPU job starts against a local cache and
# never touches the network — one less way for a long job to die.
# Idempotent: exits immediately if the cache already exists.

# Slurm copies the batch script into /var/spool/slurmd, so $0 does NOT resolve
# to the repo — resolve helpers via SLURM_SUBMIT_DIR instead, and fail loudly if
# the helper is missing rather than running on with an unconfigured environment.
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

# Fail loudly rather than let the GPU job start on a missing/half-written cache.
if [[ ! -d "${TOKENIZED_PATH}" ]]; then
  echo "FATAL: prep finished but ${TOKENIZED_PATH} does not exist" >&2
  exit 1
fi

du -sh "${TOKENIZED_PATH}"
banner "prep done."

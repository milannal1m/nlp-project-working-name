#!/bin/bash
#SBATCH --job-name=lora-gen-full
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Stage 3/5 — generate summaries with the fine-tuned adapter over the FULL test
# split. Array task 0 -> cnn_dailymail (11,490), 1 -> xsum (11,334).
#
# One dataset per task rather than both in one job: halves each task's exposure
# to the walltime, and the two run concurrently.
#
# Resumable by construction: run_generation.py counts existing JSONL records and
# appends from there, so a killed task loses only the article in flight.

# Slurm copies the batch script into /var/spool/slurmd, so $0 does NOT resolve
# to the repo — resolve helpers via SLURM_SUBMIT_DIR instead, and fail loudly if
# the helper is missing rather than running on with an unconfigured environment.
set -euo pipefail
COMMON="${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"
[[ -f "$COMMON" ]] || { echo "FATAL: cannot find $COMMON" >&2; exit 1; }
source "$COMMON"

DS="${DATASETS[${SLURM_ARRAY_TASK_ID}]}"
banner "lora gen ${DS} (model index ${MODEL_INDEX})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Gate: never burn a GPU allocation producing summaries from a missing adapter.
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "FATAL: adapter missing at ${ADAPTER_DIR} — training did not complete" >&2
  exit 1
fi

"$PY" run_generation.py --index "${MODEL_INDEX}" --dataset "${DS}"

OUT="summaries/Phi-3-LoRA-Full_4bit_${DS}_summaries.jsonl"
WANT="${FULL_SIZE[$DS]}"
GOT=$(wc -l < "${OUT}" 2>/dev/null || echo 0)
echo "records: ${GOT}/${WANT} in ${OUT}"
if (( GOT < WANT )); then
  echo "FATAL: incomplete generation (${GOT}/${WANT}) — backup job will resume" >&2
  exit 1
fi

banner "gen ${DS} done."

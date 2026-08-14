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

set -euo pipefail
COMMON="${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"
[[ -f "$COMMON" ]] || { echo "FATAL: cannot find $COMMON" >&2; exit 1; }
source "$COMMON"

DS="${DATASETS[${SLURM_ARRAY_TASK_ID}]}"
banner "lora gen ${DS} (model index ${MODEL_INDEX})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

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

#!/bin/bash
#SBATCH --job-name=lora-train-full
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200000
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
COMMON="${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"
[[ -f "$COMMON" ]] || { echo "FATAL: cannot find $COMMON" >&2; exit 1; }
source "$COMMON"

banner "lora train (4x GPU) -> ${ADAPTER_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

if [[ -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "[skip] adapter already trained: ${ADAPTER_DIR}"
  exit 0
fi

if [[ ! -d "${TOKENIZED_PATH}" ]]; then
  echo "FATAL: ${TOKENIZED_PATH} missing — run prep_data.sbatch first" >&2
  exit 1
fi

MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))

srun --ntasks=1 "$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  train_lora.py \
    --tokenized_path "${TOKENIZED_PATH}" \
    --max_seq_length 2048 \
    --output_dir "${ADAPTER_DIR}" \
    --batch_size 4 \
    --grad_accum 2 \
    --save_steps 250

if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "FATAL: training exited 0 but no adapter at ${ADAPTER_DIR}" >&2
  exit 1
fi

ls -la "${ADAPTER_DIR}"
banner "train done."

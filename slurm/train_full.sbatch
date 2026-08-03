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

# Stage 2/5 — QLoRA fine-tune Phi-3-mini on the FULL train splits, 4x H100 DDP.
#
# Sizing from measured evidence (job 5769639: 625 steps / 89.2 min on 1x H100 at
# seq 1024, effective batch 32 => 8.56 s/step):
#   491,158 pairs / batch 32 = 15,349 optimizer steps
#   ~1.5-2x per-step cost at seq 2048, ~3.5x speedup on 4 GPUs => 12-18 h
# On 4x A100 (the partition that actually schedules soonest here — gpu_h100_il
# is backlogged ~15 days, gpu_h100 ~34 h, gpu_a100_il ~8 h) budget ~24 h.
# 48 h requested = ~2x headroom and is legal on gpu_a100_il / gpu_h100_il (48 h
# cap) and on gpu_h100 (72 h cap), so the partition is swappable at submit time.
#
# Effective batch stays 32 exactly as in the validated recipe:
#   per_device 4 x grad_accum 2 x 4 GPUs = 32
#
# Checkpoints every 250 steps; a kill costs <= ~20 min and the backup job resumes.

# Slurm copies the batch script into /var/spool/slurmd, so $0 does NOT resolve
# to the repo — resolve helpers via SLURM_SUBMIT_DIR instead, and fail loudly if
# the helper is missing rather than running on with an unconfigured environment.
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

# Unique rendezvous port so a requeue never collides with a lingering process.
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

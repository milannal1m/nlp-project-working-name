#!/bin/bash
#SBATCH --job-name=lora-eval-full
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --array=22-23
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Stage 4/5 — score the fine-tuned rows. Array indices map to
# pipeline_config.EVAL_TARGETS: 22 = Phi-3-LoRA-Full_4bit/cnn_dailymail,
# 23 = .../xsum. Metrics: BLEU, ROUGE-L, METEOR, BERTScore, SummaC.
#
# On GPU deliberately: BERTScore (roberta-large) and SummaC (albert-xlarge) on
# CPU is what pushed the previous traditional-ML eval to 7h45 of an 8h limit.
#
# Refuses to score a short file — a partial JSONL would silently produce a wrong
# number that looks perfectly plausible in the results table.

# Slurm copies the batch script into /var/spool/slurmd, so $0 does NOT resolve
# to the repo — resolve helpers via SLURM_SUBMIT_DIR instead, and fail loudly if
# the helper is missing rather than running on with an unconfigured environment.
set -euo pipefail
COMMON="${SLURM_SUBMIT_DIR:-.}/slurm/_common.sh"
[[ -f "$COMMON" ]] || { echo "FATAL: cannot find $COMMON" >&2; exit 1; }
source "$COMMON"

banner "lora eval target ${SLURM_ARRAY_TASK_ID}"

read -r LABEL DS FNAME < <("$PY" - "$SLURM_ARRAY_TASK_ID" <<'EOF'
import sys
from pipeline_config import EVAL_TARGETS
t = EVAL_TARGETS[int(sys.argv[1])]
print(t["label"], t["dataset"], t["filename"])
EOF
)
echo "target: ${LABEL} / ${DS} (${FNAME})"

SRC="summaries/${FNAME}"
WANT="${FULL_SIZE[$DS]}"
GOT=$(wc -l < "${SRC}" 2>/dev/null || echo 0)
if (( GOT < WANT )); then
  echo "FATAL: ${SRC} has ${GOT}/${WANT} records — refusing to score partial data" >&2
  exit 1
fi
echo "input verified: ${GOT}/${WANT} records"

"$PY" run_evaluation.py --index "${SLURM_ARRAY_TASK_ID}"

# run_evaluation.py records per-metric failures into an "errors" dict and still
# exits 0, so verify the numbers we actually need are present and non-null.
OUT="results/metrics/${LABEL}__${DS}.json"
"$PY" - "${OUT}" "${WANT}" <<'EOF'
import json, sys

path, want = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    m = json.load(f)

required = ["bleu", "rougeL", "meteor", "bertscore_f1"]
missing = [k for k in required if m.get(k) is None]
if missing:
    sys.exit(f"FATAL: {path} missing/null metrics: {missing} | errors={m.get('errors')}")
if m.get("num_samples", 0) < want:
    sys.exit(f"FATAL: scored only {m.get('num_samples')} of {want} records")
if m.get("summac") is None:
    print(f"  [warn] summac is null | errors={m.get('errors', {}).get('summac')}")
print(f"  verified {path}: " + "  ".join(f"{k}={m[k]:.4f}" for k in required))
print(f"  num_samples={m['num_samples']}  avg_gen_len={m.get('avg_gen_len'):.1f}")
EOF

banner "eval ${LABEL}/${DS} done."

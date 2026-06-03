#!/bin/bash
# Submit the full pipeline as four dependency-chained SLURM stages:
#   1. gen_baselines  (CPU array, 4 tasks)   ┐  run in parallel
#   2. gen_llms       (GPU array, 6 tasks)   ┘
#   3. evaluate       (GPU array, 20 tasks)  ← waits for 1 & 2 (afterany)
#   4. aggregate      (CPU, 1 task)          ← waits for 3 (afterany)
#
# `afterany` is used (not afterok) so the pipeline is robust: a single failed
# generation/eval task does not block the rest, and the report is built from
# whatever metrics succeeded (missing ones show as "—").
#
# Run from the repo root:  bash slurm/submit_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/metrics

# Optional partition overrides (sbatch CLI flags win over the #SBATCH defaults).
# Set CPU_PARTITION / GPU_PARTITION in the environment to override without
# editing the scripts, e.g.  CPU_PARTITION=cpu_il bash slurm/submit_all.sh
CPU_PART_ARG=""
GPU_PART_ARG=""
[[ -n "${CPU_PARTITION:-}" ]] && CPU_PART_ARG="--partition=${CPU_PARTITION}"
[[ -n "${GPU_PARTITION:-}" ]] && GPU_PART_ARG="--partition=${GPU_PARTITION}"

echo "Submitting generation stages..."
JID_BASE=$(sbatch --parsable ${CPU_PART_ARG} slurm/gen_baselines.sbatch)
echo "  gen_baselines : array job $JID_BASE"
JID_LLM=$(sbatch --parsable ${GPU_PART_ARG} slurm/gen_llms.sbatch)
echo "  gen_llms      : array job $JID_LLM"

echo "Submitting evaluation (after generation)..."
JID_EVAL=$(sbatch --parsable ${GPU_PART_ARG} --dependency="afterany:${JID_BASE}:${JID_LLM}" slurm/evaluate.sbatch)
echo "  evaluate      : array job $JID_EVAL"

echo "Submitting aggregation (after evaluation)..."
JID_AGG=$(sbatch --parsable ${CPU_PART_ARG} --dependency="afterany:${JID_EVAL}" slurm/aggregate.sbatch)
echo "  aggregate     : job $JID_AGG"

cat <<EOF

Pipeline submitted. Job IDs:
  gen_baselines = $JID_BASE
  gen_llms      = $JID_LLM
  evaluate      = $JID_EVAL
  aggregate     = $JID_AGG

Monitor:   watch -n 10 squeue --me
Logs:      tail -f logs/sum-*_*.out
Results:   results/results.md , results/results.csv , results/charts/
EOF

#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results/metrics scratch_transformer/checkpoints

CPU_PART_ARG=""
GPU_PART_ARG=""
[[ -n "${CPU_PARTITION:-}" ]] && CPU_PART_ARG="--partition=${CPU_PARTITION}"
[[ -n "${GPU_PARTITION:-}" ]] && GPU_PART_ARG="--partition=${GPU_PARTITION}"

echo "Submitting Transformer training (array 0-1: cnn_dailymail, xsum)..."
JID_TRAIN=$(sbatch --parsable ${GPU_PART_ARG} scratch_transformer/slurm/train_transformer.sbatch)
echo "  train_tf  : array job $JID_TRAIN"

echo "Submitting Transformer generation (after training)..."
JID_GEN=$(sbatch --parsable ${GPU_PART_ARG} --dependency="afterany:${JID_TRAIN}" scratch_transformer/slurm/gen_transformer.sbatch)
echo "  gen_tf    : job $JID_GEN"

echo "Submitting Transformer evaluation (after generation)..."
JID_EVAL=$(sbatch --parsable ${GPU_PART_ARG} --dependency="afterany:${JID_GEN}" scratch_transformer/slurm/eval_transformer.sbatch)
echo "  eval_tf   : job $JID_EVAL"

echo "Submitting aggregation (after evaluation)..."
JID_AGG=$(sbatch --parsable ${CPU_PART_ARG} --dependency="afterany:${JID_EVAL}" slurm/aggregate.sbatch)
echo "  aggregate : job $JID_AGG"

cat <<EOF

Transformer pipeline submitted. Job IDs:
  train_tf  = $JID_TRAIN
  gen_tf    = $JID_GEN
  eval_tf   = $JID_EVAL
  aggregate = $JID_AGG

Monitor:   watch -n 10 squeue --me
Logs:      tail -f logs/sum-train-tf_*.out  logs/sum-gen-tf_*.out  logs/sum-eval-tf_*.out
Results:   results/results.md , results/results.csv , results/charts/
EOF

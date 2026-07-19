#!/bin/bash
#
# submit_all.sh — submit all 12 training jobs (6 variants x 2 datasets, seed 42),
# then 6 evaluation jobs that each wait for their variant's training to finish.
#
# Run from the REPO ROOT after setup_env.sh + prefetch.sh have been run once.
#
set -euo pipefail

VARIANTS="E0 E1 E2 E3 E4 E5"
DATASETS="xsum cnn_dailymail"
CLUSTER_DIR="Transformer/cluster"   # where the .sbatch scripts live

mkdir -p logs

for exp in $VARIANTS; do
    train_ids=()
    for ds in $DATASETS; do
        tag="tf_${exp}_${ds}"
        jid="$(sbatch --parsable --job-name="$tag" \
            "$CLUSTER_DIR/run_one.sbatch" "$exp" "$ds")"
        train_ids+=("$jid")
        echo "  $jid  train  $tag"
    done

    # This variant's evaluation waits for BOTH its dataset training jobs
    # (afterany: run even if one fails, so a partial result is still scored).
    dep="afterany:$(IFS=:; echo "${train_ids[*]}")"
    eid="$(sbatch --parsable --dependency="$dep" \
        --job-name="tf_eval_${exp}" "$CLUSTER_DIR/run_eval.sbatch" "$exp")"
    echo "  $eid  eval   tf_eval_${exp}  (after ${train_ids[*]})"
done

echo "Submitted 12 training + 6 evaluation jobs. Track with: squeue"
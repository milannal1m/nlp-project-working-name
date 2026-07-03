#!/bin/bash
#
# run_experiment.sh — submit the full summarization benchmark in parallel.
#
# Submits one SLURM job per (model, quant, prompt) over the datasets, plus a
# baselines job, then an evaluation job that runs once they have all finished.
# Each job skips datasets it has already produced (--skip-existing), so
# re-running only fills the gaps.
#
# Usage:
#   ./run_experiment.sh
#   ./run_experiment.sh --models Llama --quants "4bit 8bit" --prompts "P1 P2"
#   ./run_experiment.sh --sample 500
#
# Options:
#   --models / --datasets / --quants / --prompts   restrict an axis (space-separated)
#   --sample N                                      articles per dataset (default: full set)
#   --seed N                                        random seed, shared by all jobs (default: 42)
#   --no-setup                                      skip the env setup step
#   --no-baselines                                  skip the baselines job

set -euo pipefail

MODELS="Llama Phi"
DATASETS="cnn_dailymail xsum"
QUANTS="16bit 8bit 4bit"
PROMPTS="P1 P2 P3"
SAMPLE=""
SEED=42
DO_SETUP=1
DO_BASELINES=1

# Route short summarization jobs to the fast short-queue partition. A job whose
# estimated --time is <= SHORT_MAX_MIN minutes goes to SHORT_PARTITION, the rest
# to LONG_PARTITION. (SHORT_MAX_MIN must not exceed the short partition's own max.)
SHORT_PARTITION="gpu_a100_short"
LONG_PARTITION="gpu_a100_il"
SHORT_MAX_MIN=30

hms_to_min() {  # "HH:MM:SS" -> whole minutes (rounding seconds up); 10# avoids octal
    local h m s; IFS=: read -r h m s <<< "$1"
    echo $(( 10#$h * 60 + 10#$m + (10#$s + 59) / 60 ))
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)      MODELS="$2";    shift 2 ;;
        --datasets)    DATASETS="$2";  shift 2 ;;
        --quants)      QUANTS="$2";     shift 2 ;;
        --prompts)     PROMPTS="$2";   shift 2 ;;
        --sample)      SAMPLE="$2";    shift 2 ;;
        --seed)        SEED="$2";      shift 2 ;;
        --no-setup)    DO_SETUP=0;     shift ;;
        --no-baselines) DO_BASELINES=0; shift ;;
        -h|--help)     sed -n '3,21p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p summaries logs
sample_flag=""
[[ -n "$SAMPLE" ]] && sample_flag="-s $SAMPLE"

# Filter already done summaries
file_done() {  # true if this single (model, quant, prompt, dataset) output exists
    local model="$1" quant="$2" prompt="$3" ds="$4"
    [[ -f "summaries/$(python3 src/naming.py "$model" "$prompt" "$quant" "$ds" "$SAMPLE")" ]]
}

# 1. environment (idempotent)
[[ $DO_SETUP -eq 1 ]] && bash scripts/setup_env.sh

# 2 + 3. baselines and the LLM grid — each its own parallel job.
ids=()
if [[ $DO_BASELINES -eq 1 ]]; then
    jid="$(sbatch --parsable scripts/run_baselines.sh -d "$DATASETS" $sample_flag -e "$SEED" --skip-existing)"
    ids+=("$jid")
    echo "  $jid  baselines"
fi

# One job per (model, quant, prompt, dataset) so each stays within the time limit.
for model in $MODELS; do
    for quant in $QUANTS; do
        for prompt in $PROMPTS; do
            for ds in $DATASETS; do
                tag="sum_${model}_${quant}_${prompt}_${ds}"
                if file_done "$model" "$quant" "$prompt" "$ds"; then
                    echo "  skip  $tag (already done)"
                    continue
                fi
                # Per-config --time from historical logs (falls back to 36h if unmeasured).
                jtime="$(python3 src/job_time.py -m "$model" -q "$quant" -p "$prompt" \
                    -d "$ds" ${SAMPLE:+--sample "$SAMPLE"} 2>/dev/null || echo 36:00:00)"
                # Short jobs -> fast short partition; everything else -> the long partition.
                if [[ $(hms_to_min "$jtime") -le $SHORT_MAX_MIN ]]; then
                    jpart="$SHORT_PARTITION"
                else
                    jpart="$LONG_PARTITION"
                fi
                jid="$(sbatch --parsable --job-name="$tag" --partition="$jpart" --time="$jtime" \
                    scripts/run_summarization.sh -m "$model" -q "$quant" -p "$prompt" \
                    -d "$ds" $sample_flag -e "$SEED" --skip-existing)"
                ids+=("$jid")
                echo "  $jid  $tag  ($jpart, --time=$jtime)"
            done
        done
    done
done

# 4. evaluation. Wait for the jobs above (afterany: even if some fail); if no
#    jobs were submitted (all skipped), just evaluate the existing summaries now.
if [[ ${#ids[@]} -gt 0 ]]; then
    sbatch --dependency="afterany:$(IFS=:; echo "${ids[*]}")" scripts/run_evaluation.sh
    echo "Submitted ${#ids[@]} job(s) + evaluation (runs after they finish)."
else
    sbatch scripts/run_evaluation.sh
    echo "Nothing to run — submitted evaluation over existing summaries."
fi

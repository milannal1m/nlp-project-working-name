#!/bin/bash
# Submit every (model x quant x prompt) combination as SEPARATE Slurm jobs
# (currently 3 models x 3 quants x 3 prompts = 27).
#
# Use this instead of run_data_prep.sh when job arrays sit forever in the queue:
# the scheduler sees independent jobs it can start one-by-one as GPUs free up,
# rather than one array it may treat as a single large allocation.
#
# Usage:  bash QA_Evaluation/submit_data_prep_jobs.sh [-s N]
#
#   -s, --sample N   summarize only the first N articles per combo (default: all).
#                    The same N drives the --time estimate, so a short run asks the
#                    scheduler for a correspondingly short slot.
#
# This script only submits; the actual work runs inside each sbatch job below.

set -euo pipefail

SAMPLE=""   # articles per combo; empty = the whole gold file

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--sample) SAMPLE="$2"; shift 2 ;;
        -h|--help)   sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p logs

# Configuration lists (must match run_data_prep.sh)
MODELS=("unsloth/Llama-3.2-3B-Instruct")
QUANTS=("None" "8bit" "4bit")
PROMPTS=("P1" "P2" "P3")

# --- Per-config --time from historical data-prep logs -----------------------
# Data prep is summarization inference over newsqasum_gold.jsonl, so we reuse the
# per-article timings recorded in the previous run's qa_prep_*.out logs (parsed by
# src/job_time.py). Only one model was actually run, so --cross-model lets the
# other model(s) inherit the same (quant, prompt) rate; any (quant, prompt) with no
# logs at all -> 36h. LOGS_DIR is searched recursively for those .out files.
INPUT_FILE="QA_Evaluation/Datasets/newsqasum_gold.jsonl"
LOGS_DIR="logs"                                                # where qa_prep_*.out live (repo-root logs/)
QA_SAMPLE="$(wc -l < "$INPUT_FILE" 2>/dev/null | tr -d ' ')"   # full gold-file size
[[ -z "$QA_SAMPLE" || "$QA_SAMPLE" -eq 0 ]] && QA_SAMPLE=10388

# Article count the --time estimate scales by: what we actually asked for, else all.
EST_N="${SAMPLE:-$QA_SAMPLE}"
if [[ -n "$SAMPLE" ]]; then
    echo "Sampling the first $SAMPLE of $QA_SAMPLE articles per combo."
fi

model_key() {  # HF path -> the model token job_time records for these logs
    case "$(basename "$1" | tr '[:upper:]' '[:lower:]')" in
        *llama*) echo "Llama" ;;
        *phi*)   echo "Phi" ;;
        *qwen*)  echo "Qwen2" ;;
        *)       basename "$1" ;;
    esac
}

SUBMITTED=()   # collect "jobid  name  --time=..." lines for the end-of-run summary

# --- The job body, written once to a temp file ------------------------------
# Every combo runs the SAME script; they differ only in the sbatch flags and the
# variables handed over by --export, so there is nothing to template in here.
#
# Do NOT inline this as a heredoc piped into "$(sbatch ... <<EOF)". Command
# substitution runs the heredoc body through an extra round of backslash
# processing before the delimiter can protect it, which eats the line
# continuations below: the python call collapses onto one line, each '\' becomes
# an escaped space, and the pipeline dies with
#     data_preparation_pipeline.py: error: unrecognized arguments:
# where the "arguments" are five invisible one-space words. Writing the body to
# a file first keeps it verbatim, and lets you cat it when something looks off.
JOB_SCRIPT="$(mktemp)"
trap 'rm -f "$JOB_SCRIPT"' EXIT

cat > "$JOB_SCRIPT" <<'EOF'
#!/bin/bash
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000

module load devel/miniforge/25.3.1-python-3.12
source /opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/etc/profile.d/conda.sh
conda activate nlp-project

echo "Configuration -> Model: $MODEL | Quant: $QUANT | Prompt: $PROMPT"

# SAMPLE is only exported when the submitter asked for one, so a full run passes
# no --sample at all rather than an empty string argparse would reject.
SAMPLE_ARGS=()
if [[ -n "${SAMPLE:-}" ]]; then
    SAMPLE_ARGS=(--sample "$SAMPLE")
fi

python QA_Evaluation/data_preparation_pipeline.py \
    --model_path "$MODEL" \
    --quant "$QUANT" \
    --prompt_id "$PROMPT" \
    --input_file "QA_Evaluation/Datasets/newsqasum_gold.jsonl" \
    --output_dir "QA_Evaluation/Outputs" \
    "${SAMPLE_ARGS[@]}"
EOF

for MODEL in "${MODELS[@]}"; do
  for QUANT in "${QUANTS[@]}"; do
    for PROMPT in "${PROMPTS[@]}"; do

      # Short, filesystem-safe label for job name and log files.
      MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')
      JOB_TAG="${MODEL_TAG}_${QUANT}_${PROMPT}"

      # Estimated wall-clock for this combo (36h if this quant+prompt is unmeasured).
      JTIME="$(python3 src/job_time.py -m "$(model_key "$MODEL")" -q "$QUANT" \
          -p "$PROMPT" -d newsqasum --sample "$EST_N" --cross-model \
          --logs_dir "$LOGS_DIR" 2>/dev/null || true)"
      [[ -z "$JTIME" ]] && JTIME="36:00:00"

      echo "Submitting: $JOB_TAG  (--time=$JTIME)"

      # Submit the shared job body. The per-combo settings ride on the sbatch flags
      # (they take precedence over any #SBATCH directive) and --export hands the
      # three variables to the job; --parsable returns just the job id so we can
      # report it alongside the requested --time.
      EXPORT_VARS="ALL,MODEL=$MODEL,QUANT=$QUANT,PROMPT=$PROMPT"
      [[ -n "$SAMPLE" ]] && EXPORT_VARS="$EXPORT_VARS,SAMPLE=$SAMPLE"

      JID="$(sbatch --parsable \
          --job-name="qa_prep_${JOB_TAG}" \
          --time="$JTIME" \
          --output="logs/qa_prep_${JOB_TAG}_%j.out" \
          --error="logs/qa_prep_${JOB_TAG}_%j.err" \
          --export="$EXPORT_VARS" \
          "$JOB_SCRIPT")"

      echo "  -> submitted job $JID  (--time=$JTIME)"
      SUBMITTED+=("$(printf '%-10s %-30s %s' "$JID" "qa_prep_${JOB_TAG}" "--time=$JTIME")")

    done
  done
done

echo
echo "===== Submitted ${#SUBMITTED[@]} jobs (job id / name / requested time) ====="
printf '  %s\n' "${SUBMITTED[@]}"
echo "Track them with: squeue -u \$USER"

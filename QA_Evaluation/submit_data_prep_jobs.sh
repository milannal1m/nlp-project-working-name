#!/bin/bash
# Submit all 18 (model x quant x prompt) combinations as SEPARATE Slurm jobs.
#
# Use this instead of run_data_prep.sh when job arrays sit forever in the queue:
# the scheduler sees 18 independent jobs it can start one-by-one as GPUs free up,
# rather than one array it may treat as a single large allocation.
#
# Usage:  bash QA_Evaluation/submit_data_prep_jobs.sh
# This script only submits; the actual work runs inside each sbatch job below.

set -euo pipefail

mkdir -p logs

# Configuration lists (must match run_data_prep.sh)
MODELS=("meta-llama/Llama-3.2-3B-Instruct" "microsoft/Phi-3-mini-4k-instruct")
QUANTS=("None" "8bit" "4bit")
PROMPTS=("P1" "P2" "P3")

# --- Per-config --time from historical summarization logs --------------------
# Data prep is summarization inference over newsqasum_gold.jsonl, so we reuse the
# per-article timings recorded in logs/**/sum_*.out by src/job_time.py. Only one
# model was actually run, so --cross-model lets the other model(s) inherit the
# same (quant, prompt) rate; any (quant, prompt) with no logs at all -> 36h.
INPUT_FILE="QA_Evaluation/Datasets/newsqasum_gold.jsonl"
QA_SAMPLE="$(wc -l < "$INPUT_FILE" 2>/dev/null | tr -d ' ')"   # articles to scale by
[[ -z "$QA_SAMPLE" || "$QA_SAMPLE" -eq 0 ]] && QA_SAMPLE=10388

model_key() {  # HF path -> the model token used in sum_*.out log names
    case "$(basename "$1" | tr '[:upper:]' '[:lower:]')" in
        *llama*) echo "Llama" ;;
        *phi*)   echo "Phi" ;;
        *qwen*)  echo "Qwen2" ;;
        *)       basename "$1" ;;
    esac
}
quant_key() { [[ "$1" == "None" ]] && echo "16bit" || echo "$1"; }  # None == 16-bit in logs

SUBMITTED=()   # collect "jobid  name  --time=..." lines for the end-of-run summary

for MODEL in "${MODELS[@]}"; do
  for QUANT in "${QUANTS[@]}"; do
    for PROMPT in "${PROMPTS[@]}"; do

      # Short, filesystem-safe label for job name and log files.
      MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')
      JOB_TAG="${MODEL_TAG}_${QUANT}_${PROMPT}"

      # Estimated wall-clock for this combo (36h if this quant+prompt is unmeasured).
      JTIME="$(python3 src/job_time.py -m "$(model_key "$MODEL")" -q "$(quant_key "$QUANT")" \
          -p "$PROMPT" -d cnn_dailymail --sample "$QA_SAMPLE" --cross-model 2>/dev/null || true)"
      [[ -z "$JTIME" ]] && JTIME="36:00:00"

      echo "Submitting: $JOB_TAG  (--time=$JTIME)"

      # Pipe a self-contained job script to sbatch. --export passes this combo's
      # variables into the job's environment; --parsable returns just the job id
      # so we can report it alongside the requested --time.
      JID="$(sbatch --parsable --export=ALL,MODEL="$MODEL",QUANT="$QUANT",PROMPT="$PROMPT" <<EOF
#!/bin/bash
#SBATCH --job-name=qa_prep_${JOB_TAG}
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=${JTIME}
#SBATCH --output=logs/qa_prep_${JOB_TAG}_%j.out
#SBATCH --error=logs/qa_prep_${JOB_TAG}_%j.err

module load devel/miniforge/25.3.1-python-3.12
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-project

echo "Configuration -> Model: \$MODEL | Quant: \$QUANT | Prompt: \$PROMPT"

python QA_Evaluation/data_preparation_pipeline.py \\
    --model_path "\$MODEL" \\
    --quant "\$QUANT" \\
    --prompt_id "\$PROMPT" \\
    --input_file "QA_Evaluation/Datasets/newsqasum_gold.jsonl" \\
    --output_dir "QA_Evaluation/Outputs"
EOF
)"

      echo "  -> submitted job $JID  (--time=$JTIME)"
      SUBMITTED+=("$(printf '%-10s %-30s %s' "$JID" "qa_prep_${JOB_TAG}" "--time=$JTIME")")

    done
  done
done

echo
echo "===== Submitted ${#SUBMITTED[@]} jobs (job id / name / requested time) ====="
printf '  %s\n' "${SUBMITTED[@]}"
echo "Track them with: squeue -u \$USER"

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

for MODEL in "${MODELS[@]}"; do
  for QUANT in "${QUANTS[@]}"; do
    for PROMPT in "${PROMPTS[@]}"; do

      # Short, filesystem-safe label for job name and log files.
      MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')
      JOB_TAG="${MODEL_TAG}_${QUANT}_${PROMPT}"

      echo "Submitting: $JOB_TAG"

      # Pipe a self-contained job script to sbatch. --export passes this combo's
      # variables into the job's environment.
      sbatch --export=ALL,MODEL="$MODEL",QUANT="$QUANT",PROMPT="$PROMPT" <<EOF
#!/bin/bash
#SBATCH --job-name=qa_prep_${JOB_TAG}
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=28:00:00
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

    done
  done
done

echo "Submitted 18 jobs. Track them with: squeue -u \$USER"

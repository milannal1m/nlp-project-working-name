#!/bin/bash
#SBATCH --job-name=qa_data_prep
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=06:00:00
#SBATCH --output=logs/qa_prep_%A_%a.out
#SBATCH --error=logs/qa_prep_%A_%a.err
#SBATCH --array=0-17  # This tells Slurm to spawn 18 clones (IDs 0 to 17)

# 1. Activate the modern environment (Ensure this matches your team's setup)
module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-project

# 2. Define our configuration lists
MODELS=("meta-llama/Llama-3.2-3B-Instruct" "microsoft/Phi-3-mini-4k-instruct")
QUANTS=("None" "8bit" "4bit")
PROMPTS=("P1" "P2" "P3")

# 3. The Math: Map this clone's specific ID to a combination
# Total combinations: 2 Models * 3 Quants * 3 Prompts = 18 (Index 0-17)
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 9 ))
REMAINING=$(( SLURM_ARRAY_TASK_ID % 9 ))
QUANT_IDX=$(( REMAINING / 3 ))
PROMPT_IDX=$(( REMAINING % 3 ))

# 4. Extract the specific variables for this run
CURRENT_MODEL=${MODELS[$MODEL_IDX]}
CURRENT_QUANT=${QUANTS[$QUANT_IDX]}
CURRENT_PROMPT=${PROMPTS[$PROMPT_IDX]}

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Configuration -> Model: $CURRENT_MODEL | Quant: $CURRENT_QUANT | Prompt: $CURRENT_PROMPT"

# 5. Execute the Python pipeline with the dynamically selected variables
python QA_Evaluation/data_preparation_pipeline.py \
    --model_path "$CURRENT_MODEL" \
    --quant "$CURRENT_QUANT" \
    --prompt_id "$CURRENT_PROMPT" \
    --input_file "QA_Evaluation/Datasets/newsqasum_gold.jsonl" \
    --output_dir "QA_Evaluation/Outputs"
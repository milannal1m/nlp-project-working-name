#!/bin/bash
#SBATCH --job-name=llama-summarization
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-project

python main.py \
    --model_name_or_path unsloth/Llama-3.2-3B-Instruct \
    --quantization_method 4bit &
EXPERIMENT_PID=$!

sleep 3600
nvidia-smi

# Auf Training warten
wait $EXPERIMENT_PID


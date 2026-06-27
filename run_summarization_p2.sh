#!/bin/bash
#SBATCH --job-name=llama-summarization-p2
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp-env

nvidia-smi

python main.py \
    --model_name_or_path unsloth/Llama-3.2-3B-Instruct \
    --quantization_method 4bit \
    --prompt_name P2

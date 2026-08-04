#!/bin/bash
#SBATCH --job-name=qa_eval
#SBATCH --output=logs/qa_eval_%j.out
#SBATCH --error=logs/qa_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# 1. Load Conda
module load devel/miniforge/25.3.1-python-3.12
source /opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/etc/profile.d/conda.sh

# 2. Activate the isolated evaluation environment
conda activate qa-eval

# 3. Run the pipeline
echo "Starting factual consistency evaluation..."
python run_qa.py
echo "Evaluation complete."
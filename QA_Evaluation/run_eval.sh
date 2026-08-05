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
python scripts/run_qa.py
echo "Evaluation complete."

# 4. Reference-based metrics + the LERC aggregation, into one CSV.
# Guard first: a crashed run_qa.py would leave an empty or stale results file, and
# evaluate_all_metrics.py would silently join the old LERC numbers into the new CSV.
if [ ! -s results/final_evaluation_results.jsonl ]; then
    echo "run_qa.py produced no results -- skipping the metrics step." >&2
    exit 1
fi

# evaluate_all_metrics.py needs evaluate/bert_score, which exist only in nlp-project
# (the qa-eval env is Python 3.7 and has neither), hence the second activation. The
# GPU stays allocated for BERTScore.
echo "Starting reference-based metrics (BLEU/ROUGE-L/METEOR/BERTScore)..."
conda activate nlp-project
python scripts/evaluate_all_metrics.py
echo "All metrics written to QA_Evaluation/results/qa_evaluation.csv."
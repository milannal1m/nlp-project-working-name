#!/bin/bash
#
# prefetch.sh — cache everything the evaluator downloads, on a LOGIN node
# (compute nodes have no internet). Run ONCE after setup_env.sh.
#
set -euo pipefail

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate transformer-abl

# BERTScore's default scoring model (the 1.4GB roberta-large you saw locally).
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('roberta-large'); AutoTokenizer.from_pretrained('roberta-large')"

# METEOR / tokenization data that the evaluator pulls at runtime.
python -c "import nltk; [nltk.download(p) for p in ('wordnet','punkt_tab','omw-1.4')]"

# The xsum / cnn_dailymail datasets are also downloaded here so training jobs
# on offline compute nodes read them from the datasets cache.
python -c "from datasets import load_dataset; load_dataset('EdinburghNLP/xsum', split='train', streaming=True)"
python -c "from datasets import load_dataset; load_dataset('abisee/cnn_dailymail','3.0.0', split='train', streaming=True)"

echo "Prefetch done. Model + NLTK data cached under ~/.cache/huggingface and ~/nltk_data."
#!/bin/bash
#
# setup_env.sh — create the conda env for the Transformer ablation experiments.
# Idempotent: safe to call repeatedly; only creates the env if it's missing.
# Run this ONCE on a LOGIN node (compute nodes have no internet).
#
set -euo pipefail

ENV_NAME="transformer-abl"

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists — skipping creation."
else
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"

# CUDA-enabled torch first (its own index), so nothing downgrades it to CPU.
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Only what this baseline + the benchmark evaluator actually need.
pip install \
  "datasets>=2.18.0" \
  "evaluate>=0.4.0" \
  "rouge-score>=0.1.2" \
  "bert-score>=0.3.13" \
  "nltk>=3.8.1" \
  "sentencepiece>=0.1.99" \
  "protobuf>=3.20.0" \
  "absl-py>=1.0.0"

echo "Done. Activate with: conda activate $ENV_NAME"
#!/bin/bash
set -e

ENV_NAME="nlp-project"

module load devel/miniforge/25.3.1-python-3.12
source /opt/bwhpc/common/devel/miniforge/25.3.1-py3.12/etc/profile.d/conda.sh

# Idempotent: only create the env if it doesn't exist yet, so run_experiment.sh
# can call this on every run without erroring on an already-created env.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists — skipping creation."
else
    conda create -n "$ENV_NAME" python=3.11 -y
fi
conda activate "$ENV_NAME"

pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

pip install \
  "transformers>=4.40.0,<5.0.0" \
  "datasets>=2.18.0" \
  "bitsandbytes>=0.43.0" \
  "accelerate>=0.27.0" \
  "sumy==0.12.0" \
  "evaluate>=0.4.0" \
  "rouge-score>=0.1.2" \
  "sentencepiece>=0.1.99" \
  "protobuf>=3.20.0" \
  "absl-py>=1.0.0" \
  "bert-score>=0.3.13" \
  "nltk>=3.8.1"

echo "Done. Activate with: conda activate $ENV_NAME"

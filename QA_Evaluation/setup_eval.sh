#!/bin/bash

echo "Loading Conda..."
module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Creating isolated Python 3.8 environment..."
conda create -n qa-eval-env python=3.8 -y
conda activate qa-eval-env

echo "Installing legacy dependencies..."
python -m pip install "pip==20.2.4" "setuptools<59.0.0" wheel
pip install torch==1.6.0

echo "Bypassing broken PyPI package (extracting raw code)..."
rm -rf temp_repo
git clone https://github.com/salesforce/QAFactEval.git temp_repo
mv temp_repo/qafacteval.py .
mv temp_repo/lerc_quip.py .
rm -rf temp_repo

echo "Bypassing dead Amazon link (downloading base BART files locally)..."
mkdir -p facebook/bart-large
cd facebook/bart-large
wget -qO config.json https://huggingface.co/facebook/bart-large/resolve/main/config.json
wget -qO pytorch_model.bin https://huggingface.co/facebook/bart-large/resolve/main/pytorch_model.bin
wget -qO vocab.json https://huggingface.co/facebook/bart-large/resolve/main/vocab.json
wget -qO merges.txt https://huggingface.co/facebook/bart-large/resolve/main/merges.txt
cd ..

echo "Setup complete. The environment and workarounds are locked in."
#!/bin/bash
# One-time environment setup on bwUniCluster 3.0.
# Creates/updates the `nlp-env` conda environment with everything the pipeline
# needs. Run once from a login node (it only downloads/builds, no GPU needed):
#
#   bash slurm/setup_env.sh
#
# SummaC and QAFactEval are installed separately because they are not on the
# default index; QAFactEval is best-effort (the pipeline records null for the
# QA-Eval metric if it is unavailable).

set -uo pipefail
cd "$(dirname "$0")/.."

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE '(^|/)nlp-env\s'; then
    echo "Creating conda env nlp-env (python 3.11)..."
    conda create -n nlp-env python=3.11 -y
fi
conda activate nlp-env

echo "Installing core requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Installing SummaC (factual consistency)..."
python -m pip install --no-deps git+https://github.com/tingofurro/summac.git || \
    echo "WARN: SummaC install failed — the SummaC metric will be skipped."

echo "Installing QAFactEval (optional QA-Eval metric)..."
if python -m pip install qafacteval; then
    echo "qafacteval installed. NOTE: it also needs model weights in models/qafacteval"
    echo "      (see https://github.com/salesforce/QAFactEval). Skipped if absent."
else
    echo "WARN: qafacteval not installed — QA-Eval will be recorded as null."
fi

echo "Pre-downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)" || true

echo "Pre-fetching base LLM weights into the HF cache (avoids quant-variant races)..."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
python - <<'PY' || echo "WARN: model prefetch failed — tasks will download at runtime instead."
from huggingface_hub import snapshot_download
for repo in ["unsloth/Llama-3.2-3B-Instruct", "microsoft/Phi-3-mini-4k-instruct"]:
    print("prefetch:", repo)
    snapshot_download(repo, ignore_patterns=["*.gguf", "*.onnx", "original/*"])
PY

echo
echo "Done. Sanity check the model matrix with:  python pipeline_config.py"

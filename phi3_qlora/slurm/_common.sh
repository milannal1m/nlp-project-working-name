#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs adapters data

PY="${NLP_PYTHON:-$HOME/.conda/envs/nlp-env/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "FATAL: python interpreter not found at $PY" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

"$PY" - <<'PYCHK' || exit 1
import sys
from importlib.metadata import version
pv = version("peft")
major, minor = (int(x) for x in pv.split(".")[:2])
if (major, minor) >= (0, 19):
    sys.exit(
        f"FATAL: peft {pv} is incompatible with transformers "
        f"{version('transformers')} (missing EmbeddingParallel).\n"
        f"       Fix with: pip install 'peft<0.19'"
    )
print(f"[env] peft {pv} + transformers {version('transformers')}: compatible")
PYCHK

TOKENIZED_PATH="${TOKENIZED_PATH:-data/tokenized_full}"
ADAPTER_DIR="${ADAPTER_DIR:-adapters/phi3-lora-news-full}"
MODEL_INDEX="${MODEL_INDEX:-11}"
DATASETS=(cnn_dailymail xsum)
declare -A FULL_SIZE=([cnn_dailymail]=11490 [xsum]=11334)

banner() { echo "[$(date '+%F %T')] $* on $(hostname)"; }

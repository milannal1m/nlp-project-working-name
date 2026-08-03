#!/bin/bash
# Shared setup for every full_run job. Sourced, not executed.
#
# Deliberately does NOT use `conda activate`: the miniforge shell stub is broken
# in batch shells on some compute nodes here ("No module named 'conda'"), which
# is how earlier jobs died two minutes in. Calling the env interpreter by
# absolute path always works.

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

# Guard: peft >= 0.19 unconditionally imports `EmbeddingParallel` from
# transformers.integrations.tensor_parallel, which transformers 4.57.x does not
# export, so EVERY adapter load raises ImportError — killing both
# resume-from-checkpoint and PeftModel.from_pretrained at generation time.
# Caught by the GPU smoke gate (job 6109680). Fail here in one second rather
# than after hours of queueing and training.
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

# Paths shared across the chain.
TOKENIZED_PATH="${TOKENIZED_PATH:-data/tokenized_full}"
ADAPTER_DIR="${ADAPTER_DIR:-adapters/phi3-lora-news-full}"
MODEL_INDEX="${MODEL_INDEX:-11}"          # pipeline_config.MODELS index
DATASETS=(cnn_dailymail xsum)
declare -A FULL_SIZE=([cnn_dailymail]=11490 [xsum]=11334)

banner() { echo "[$(date '+%F %T')] $* on $(hostname)"; }

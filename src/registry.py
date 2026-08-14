"""Single source of truth for the model registry.

Deliberately dependency-free (stdlib only, no torch): `analysis.py` builds its
job-status report from log filenames alone and must stay importable on a laptop
without the ML stack, and `run_experiment.sh` reads the labels through the CLI
below. `model.py` re-exports MODEL_CONFIGS, so `from model import MODEL_CONFIGS`
keeps working for callers that already have torch loaded.

Add a model here and it becomes available to the experiment grid, the job-status
report and the LaTeX tables at once.
"""

import os

# Short label -> HuggingFace id (or local path). The label is used in output
# filenames and as the --model value in the scripts.
#
# Qwen2 uses an offline local copy on the cluster; when that folder is absent
# (e.g. running post-hoc analysis on a laptop) we fall back to the public Hub id
# so the tokenizer/config still resolves instead of erroring out.
_QWEN2_LOCAL = "./Qwen2-1.5B-Instruct"
MODEL_CONFIGS = {
    "Llama": "unsloth/Llama-3.2-3B-Instruct",
    "Phi":   "microsoft/Phi-3-mini-4k-instruct",
    "Qwen2": _QWEN2_LOCAL if os.path.isdir(_QWEN2_LOCAL) else "Qwen/Qwen2-1.5B-Instruct",
}

# Label -> the compact name printed in the LaTeX tables and figures. The full
# official names (Llama-3.2-3B-Instruct, Phi-3-mini-4k-instruct) belong in the
# paper's setup section, not in a table header.
MODEL_LABELS = {
    "Llama": "Llama-3.2-3B",
    "Phi":   "Phi-3-mini",
    "Qwen2": "Qwen2-1.5B",
}


if __name__ == "__main__":
    # CLI: print the labels space-separated, for run_experiment.sh's default grid.
    print(" ".join(MODEL_CONFIGS))

"""Single source of truth for summary output filenames.

Both main.py (to skip/write outputs) and run_experiment.sh (to decide which
combinations still need to run) rely on this exact convention. If you change
the pattern here, the bash orchestrator picks it up automatically because it
calls this module instead of hard-coding the pattern.
"""

import sys


def sample_tag(sample) -> str:
    """'full' for the whole split, otherwise the article count (e.g. '500').

    Accepts None, "", "full", an int, or a numeric string — all normalised here
    so callers (Python and the shell) produce identical filenames.
    """
    if sample is None or str(sample).strip().lower() in ("", "full"):
        return "full"
    return str(sample).strip()


def summary_filename(model_label: str, prompt_name: str, quant: str,
                     dataset_name: str, sample) -> str:
    """Filename for one (model, prompt, quant, dataset, sample) summarization run."""
    return (f"{model_label}_{prompt_name}_{quant}_{dataset_name}_"
            f"{sample_tag(sample)}_summaries.jsonl")


def baseline_filename(prefix: str, dataset_name: str, sample) -> str:
    """Filename for one extractive-baseline run (Lead-1/Lead-3/TextRank/TFIDF)."""
    return f"{prefix}_{dataset_name}_{sample_tag(sample)}_summaries.jsonl"


if __name__ == "__main__":
    # CLI: python naming.py MODEL PROMPT QUANT DATASET SAMPLE  -> prints filename
    if len(sys.argv) != 6:
        print("usage: python naming.py MODEL_LABEL PROMPT QUANT DATASET SAMPLE",
              file=sys.stderr)
        sys.exit(2)
    print(summary_filename(*sys.argv[1:6]))

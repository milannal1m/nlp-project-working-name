from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------
# Global run settings
# --------------------------------------------------------------------------
# SAMPLE caps how many records per dataset are generated AND evaluated.
# Set to an int (e.g. 500) for a quick run, or None to use the FULL test split.
SAMPLE: Optional[int] = None
DATASETS: list[str] = ["cnn_dailymail", "xsum"]
OUTPUT_DIR: str = "summaries"
RESULTS_DIR: str = "results"
METRICS_DIR: str = "results/metrics"
CHARTS_DIR: str = "results/charts"

# Canonical HuggingFace *test* split sizes. Used so that a full run (SAMPLE=None)
# can still skip already-complete summary files: a file is "done" once it has
# this many records. Add an entry when you add a dataset; an unknown dataset
# falls back to "never skip" (always regenerate) so we never reuse a partial file.
FULL_SPLIT_SIZES: dict[str, int] = {
    "cnn_dailymail": 11490,
    "xsum": 11334,
}


def target_count(dataset: str, sample: Optional[int]) -> int:
    """How many records make a complete run for ``dataset`` given ``sample``.

    With an explicit ``sample`` it is just that number; with ``sample=None``
    (full run) it is the dataset's full test-split size, or a huge sentinel for
    an unknown dataset so its file is never wrongly treated as complete.
    """
    if sample is not None:
        return sample
    return FULL_SPLIT_SIZES.get(dataset, 10**12)


@dataclass
class ModelSpec:
    """One experiment configuration.

    Attributes:
        label: Output prefix, e.g. ``Llama_4bit`` -> ``Llama_4bit_{dataset}_summaries.jsonl``.
        kind: ``baseline`` (CPU) or ``llm`` (GPU).
        baseline: For baselines, which one: ``lead-1``/``lead-3``/``textrank``/``tfidf``.
        model_path: HuggingFace id for LLMs.
        quant: ``None`` | ``4bit`` | ``8bit`` for LLMs.
    """

    label: str
    kind: str  # "baseline" | "llm"
    baseline: Optional[str] = None
    model_path: Optional[str] = None
    quant: str = "None"


# --------------------------------------------------------------------------
# Model matrix
# --------------------------------------------------------------------------
BASELINES: list[ModelSpec] = [
    ModelSpec(label="Lead-1", kind="baseline", baseline="lead-1"),
    ModelSpec(label="Lead-3", kind="baseline", baseline="lead-3"),
    ModelSpec(label="TextRank", kind="baseline", baseline="textrank"),
    ModelSpec(label="TFIDF", kind="baseline", baseline="tfidf"),
]

_LLM_BASES = [
    ("Llama", "unsloth/Llama-3.2-3B-Instruct"),
    ("Phi-3", "microsoft/Phi-3-mini-4k-instruct"),
]
_QUANTS = ["None", "4bit", "8bit"]

LLMS: list[ModelSpec] = [
    ModelSpec(label=f"{name}_{quant}", kind="llm", model_path=path, quant=quant)
    for name, path in _LLM_BASES
    for quant in _QUANTS
]

MODELS: list[ModelSpec] = BASELINES + LLMS


# Build the JSONL filename for one (model label, dataset) summary file.
def output_filename(label: str, dataset: str) -> str:
    return f"{label}_{dataset}_summaries.jsonl"


# Flat list of every (model, dataset) summary file the pipeline produces /
# evaluates. The evaluation SLURM array indexes into this list by task id.
EVAL_TARGETS: list[dict] = [
    {
        "label": spec.label,
        "dataset": dataset,
        "filename": output_filename(spec.label, dataset),
    }
    for spec in MODELS
    for dataset in DATASETS
]


if __name__ == "__main__":
    print(f"Sample per dataset: {SAMPLE if SAMPLE is not None else 'full split'}")
    print(f"Datasets: {DATASETS}")
    print(f"\n{len(MODELS)} model configs:")
    for i, m in enumerate(MODELS):
        extra = m.baseline if m.kind == "baseline" else f"{m.model_path} ({m.quant})"
        print(f"  [{i:2d}] {m.label:14s} {m.kind:9s} {extra}")
    print(f"\n{len(EVAL_TARGETS)} evaluation targets (model x dataset).")
    print(f"  baselines (CPU): {len(BASELINES)}  |  llms (GPU): {len(LLMS)}")

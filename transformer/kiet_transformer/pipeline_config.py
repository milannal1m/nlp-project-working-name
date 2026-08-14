from dataclasses import dataclass, field
from typing import Optional

SAMPLE: Optional[int] = None
DATASETS: list[str] = ["cnn_dailymail", "xsum"]
OUTPUT_DIR: str = "summaries"
RESULTS_DIR: str = "results"
METRICS_DIR: str = "results/metrics"
CHARTS_DIR: str = "results/charts"


FULL_SPLIT_SIZES: dict[str, int] = {
    "cnn_dailymail": 11490,
    "xsum": 11334,
}


def target_count(dataset: str, sample: Optional[int]) -> int:
    if sample is not None:
        return sample
    return FULL_SPLIT_SIZES.get(dataset, 10**12)


@dataclass
class ModelSpec:

    label: str
    kind: str
    baseline: Optional[str] = None
    model_path: Optional[str] = None
    quant: str = "None"
    adapter_path: Optional[str] = None
    prompt_name: Optional[str] = None


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

FINETUNED: list[ModelSpec] = [
    ModelSpec(
        label="Phi-3-LoRA_4bit",
        kind="llm",
        model_path="microsoft/Phi-3-mini-4k-instruct",
        quant="4bit",
        adapter_path="adapters/phi3-lora-news",
    ),
    ModelSpec(
        label="Phi-3-LoRA-Full_4bit",
        kind="llm",
        model_path="microsoft/Phi-3-mini-4k-instruct",
        quant="4bit",
        adapter_path="adapters/phi3-lora-news-full",
    ),
]

PROMPT_VARIANTS: list[ModelSpec] = [
    ModelSpec(
        label=f"Phi-3-LoRA-Full_4bit_{p}",
        kind="llm",
        model_path="microsoft/Phi-3-mini-4k-instruct",
        quant="4bit",
        adapter_path="adapters/phi3-lora-news-full",
        prompt_name=p,
    )
    for p in ("P1", "P2", "P3")
]

P4_VARIANTS: list[ModelSpec] = [
    ModelSpec(
        label="Phi-3-LoRA-Full_4bit_P4",
        kind="llm",
        model_path="microsoft/Phi-3-mini-4k-instruct",
        quant="4bit",
        adapter_path="adapters/phi3-lora-news-full",
        prompt_name="P4",
    ),
] + [
    ModelSpec(
        label=f"Phi-3_{quant}_P4",
        kind="llm",
        model_path="microsoft/Phi-3-mini-4k-instruct",
        quant=quant,
        prompt_name="P4",
    )
    for quant in _QUANTS
]

MODELS: list[ModelSpec] = BASELINES + LLMS + FINETUNED + PROMPT_VARIANTS + P4_VARIANTS


def output_filename(label: str, dataset: str) -> str:
    return f"{label}_{dataset}_summaries.jsonl"


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
    print(
        f"  baselines (CPU): {len(BASELINES)}  |  "
        f"llms (GPU): {len(LLMS) + len(FINETUNED) + len(PROMPT_VARIANTS) + len(P4_VARIANTS)}"
    )

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

    """Records needed for a complete run: ``sample``, or the full test split.

    An unknown dataset returns a huge sentinel so its file is never mistaken
    for complete.
    """

    if sample is not None:

        return sample

    return FULL_SPLIT_SIZES.get(dataset, 10**12)


@dataclass

class ModelSpec:

    """One experiment configuration.

    ``label`` prefixes the output files; ``kind`` is ``baseline`` (CPU) or
    ``llm`` (GPU); ``adapter_path`` attaches a LoRA adapter over the base.
    """


    label: str

    kind: str

    baseline: Optional[str] = None

    model_path: Optional[str] = None

    quant: str = "None"

    adapter_path: Optional[str] = None


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


MODELS: list[ModelSpec] = BASELINES + LLMS + FINETUNED


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

    print(f"  baselines (CPU): {len(BASELINES)}  |  llms (GPU): {len(LLMS) + len(FINETUNED)}")

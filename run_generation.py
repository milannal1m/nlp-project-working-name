"""Generate summaries for a single model configuration (one SLURM array task).

The SLURM array passes an index via ``--index`` (or ``$SLURM_ARRAY_TASK_ID``)
that selects one entry from ``pipeline_config.MODELS``. Each task writes one
JSONL file per dataset. Generation is idempotent: if an output file already has
at least ``--sample`` lines it is skipped, so the existing full-test-set
summaries are reused instead of regenerated.

Usage (driven by SLURM, but runnable locally too):
    python run_generation.py --index 4 --sample 500
"""

import argparse
import json
import os
import time

import nltk

from pipeline_config import MODELS, DATASETS, OUTPUT_DIR, SAMPLE, output_filename
from dataset import extract_fields, load_datasets_streaming

nltk.download("punkt_tab", quiet=True)


def _has_enough(path: str, sample: int) -> bool:
    """True if ``path`` already holds at least ``sample`` JSONL records."""
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) >= sample


def _generate_llm(spec, datasets: dict, output_dir: str, sample: int) -> None:
    """Run an autoregressive LLM over every dataset and write JSONL summaries."""
    import torch
    from model import RunConfig, SummarizationModel

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Warning: no GPU found, running on CPU (very slow)", flush=True)

    config = RunConfig(
        model_name_or_path=spec.model_path,
        quantization_method=spec.quant,
        output_dir=output_dir,
        model_label=spec.label,
    )
    model = SummarizationModel(config)

    for dataset_name, data in datasets.items():
        output_path = os.path.join(output_dir, output_filename(spec.label, dataset_name))
        if _has_enough(output_path, sample):
            print(f"[skip] {output_path} already has >= {sample} samples", flush=True)
            continue
        print("=" * 80, flush=True)
        print(f"[{spec.label}] {dataset_name} (up to {sample}) -> {output_path}", flush=True)
        start = time.time()
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            for idx, item in enumerate(data, start=1):
                news_text, ref_summary, qa_pairs = extract_fields(dataset_name, item)
                generated_text, input_len, generated_len = model.summarize(news_text)
                record = {
                    "news": news_text,
                    "reference_summary": ref_summary,
                    "generated_summary": generated_text,
                }
                if qa_pairs is not None:
                    record["qa_pairs"] = qa_pairs
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if idx == 1 or idx % 25 == 0:
                    print(
                        f"  sample {idx}/{sample} | in={input_len} | gen={generated_len}",
                        flush=True,
                    )
        print(f"  Done in {(time.time() - start) / 60:.2f} min -> {output_path}", flush=True)


def _generate_baseline(spec, datasets: dict, output_dir: str, sample: int) -> None:
    """Run a non-neural baseline (lead-n / textrank / tf-idf)."""
    # Skip cheaply if both dataset outputs already exist.
    if all(
        _has_enough(os.path.join(output_dir, output_filename(spec.label, ds)), sample)
        for ds in datasets
    ):
        print(f"[skip] {spec.label}: outputs already present", flush=True)
        return

    if spec.baseline in ("lead-1", "lead-3"):
        from baseline_lead import run_lead

        run_lead(int(spec.baseline.split("-")[1]), datasets, output_dir, sample)
    elif spec.baseline == "textrank":
        from baseline_textrank import run_textrank

        run_textrank(datasets, output_dir, sample)
    elif spec.baseline == "tfidf":
        from baseline_tfidf import run_tfidf

        run_tfidf(datasets, output_dir, sample)
    else:
        raise ValueError(f"Unknown baseline: {spec.baseline}")


# Parse CLI args: --index into MODELS, --sample size, --output_dir.
def parse_args():
    parser = argparse.ArgumentParser(description="Generate summaries for one model config")
    parser.add_argument(
        "--index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)),
        help="Index into pipeline_config.MODELS (defaults to $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


# Entry point: pick MODELS[index], load datasets, run baseline or LLM generation.
def main() -> None:
    args = parse_args()
    if not (0 <= args.index < len(MODELS)):
        raise SystemExit(
            f"--index {args.index} out of range (0..{len(MODELS) - 1}). "
            "Set --index or $SLURM_ARRAY_TASK_ID."
        )

    spec = MODELS[args.index]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Config [{args.index}] {spec.label} ({spec.kind}) ===", flush=True)

    datasets = load_datasets_streaming(sample=args.sample)
    if spec.kind == "baseline":
        _generate_baseline(spec, datasets, args.output_dir, args.sample)
    else:
        _generate_llm(spec, datasets, args.output_dir, args.sample)
    print(f"=== Done: {spec.label} ===", flush=True)


if __name__ == "__main__":
    main()

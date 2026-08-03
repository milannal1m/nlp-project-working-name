import argparse

import json

import os

import time


import nltk


from pipeline_config import MODELS, DATASETS, OUTPUT_DIR, SAMPLE, output_filename, target_count

from dataset import extract_fields, load_datasets_streaming


nltk.download("punkt_tab", quiet=True)


def _count_records(path: str) -> int:

    """Number of JSONL records already written to ``path`` (0 if it does not exist)."""

    if not os.path.exists(path):

        return 0

    with open(path, "r", encoding="utf-8") as f:

        return sum(1 for _ in f)


def _has_enough(path: str, sample: int) -> bool:

    """True if ``path`` already holds at least ``sample`` JSONL records."""

    return _count_records(path) >= sample


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

        adapter_path=getattr(spec, "adapter_path", None),

    )

    model = SummarizationModel(config)


    for dataset_name, data in datasets.items():

        target = target_count(dataset_name, sample)

        output_path = os.path.join(output_dir, output_filename(spec.label, dataset_name))

        existing = _count_records(output_path)

        if existing >= target:

            print(f"[skip] {output_path} already has >= {target} samples", flush=True)

            continue

        mode = "a" if existing > 0 else "w"

        if existing > 0:

            print(f"[resume] {output_path} has {existing}/{target}; continuing from {existing + 1}", flush=True)

        print("=" * 80, flush=True)

        print(f"[{spec.label}] {dataset_name} (target {target}) -> {output_path}", flush=True)

        start = time.time()

        with open(output_path, mode, encoding="utf-8", newline="\n") as f:

            for idx, item in enumerate(data, start=1):

                if idx <= existing:

                    continue

                news_text, ref_summary, qa_pairs = extract_fields(dataset_name, item)

                generated_text, input_len, generated_len = model.summarize(

                    news_text, dataset=dataset_name

                )

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

                        f"  sample {idx}/{target} | in={input_len} | gen={generated_len}",

                        flush=True,

                    )

        print(f"  Done in {(time.time() - start) / 60:.2f} min -> {output_path}", flush=True)


def _generate_baseline(spec, datasets: dict, output_dir: str, sample: int) -> None:

    """Run a non-neural baseline (lead-n / textrank / tf-idf)."""


    if all(

        _has_enough(

            os.path.join(output_dir, output_filename(spec.label, ds)),

            target_count(ds, sample),

        )

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

    parser.add_argument(

        "--dataset",

        type=str,

        default=None,

        choices=DATASETS,

        help="Restrict to one dataset (lets a config be split across SLURM jobs)",

    )

    return parser.parse_args()


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

    if args.dataset:

        datasets = {args.dataset: datasets[args.dataset]}

        print(f"    restricted to dataset: {args.dataset}", flush=True)

    if spec.kind == "baseline":

        _generate_baseline(spec, datasets, args.output_dir, args.sample)

    else:

        _generate_llm(spec, datasets, args.output_dir, args.sample)

    print(f"=== Done: {spec.label} ===", flush=True)


if __name__ == "__main__":

    main()

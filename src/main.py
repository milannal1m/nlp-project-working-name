import argparse
import glob
import json
import os
import time

import nltk
import torch
from transformers import set_seed

nltk.download('punkt_tab', quiet=True)

from dataset import DATASET_CONFIGS, extract_fields, load_datasets_streaming
from model import MODEL_CONFIGS, RunConfig, SummarizationModel
from prompts import PROMPT_CONFIGS
from naming import summary_filename, baseline_filename
from baselines.lead import run_lead
from baselines.textrank import run_textrank
from baselines.tfidf import run_tfidf
from evaluator import Evaluator

DATASET_NAMES = list(DATASET_CONFIGS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarization experiment worker: summarize, run baselines, or evaluate."
    )
    parser.add_argument(
        "--task",
        choices=["summarize", "baselines", "evaluate", "all"],
        default="all",
        help="summarize: one model/quant/prompt over the datasets. "
             "baselines: Lead/TextRank/TF-IDF. evaluate: score every .jsonl. "
             "all: baselines + one model + evaluate (single-machine convenience).",
    )

    # --- model selection (for summarize / all) ---
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Model label from the registry (resolves to a HuggingFace id).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Explicit HF id / local path. Use with --model_label. "
             "Overrides --model if both are given.",
    )
    parser.add_argument(
        "--model_label",
        type=str,
        default=None,
        help="Short label used in output filenames (required with --model_name_or_path).",
    )

    parser.add_argument(
        "--quantization_method",
        type=str,
        choices=["16bit", "8bit", "4bit", "None"],
        default="16bit",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        choices=list(PROMPT_CONFIGS.keys()),
        default="P1",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=DATASET_NAMES,
        help="Subset of datasets to run. Defaults to all configured datasets.",
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Articles per dataset. Omit to use the full test set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset shuffling and generation (reproducibility).")
    parser.add_argument("--output_dir", type=str, default="summaries")
    parser.add_argument("--log_path", type=str, default="results/evaluation.log")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a (model/quant/prompt/dataset) run if its .jsonl already exists.",
    )
    parser.add_argument(
        "--append_eval",
        action="store_true",
        help="Append to the eval log/csv instead of overwriting them (task=evaluate).",
    )

    return parser.parse_args()


def resolve_model(args) -> tuple[str, str]:
    """Return (model_name_or_path, model_label) from the CLI args."""
    if args.model_name_or_path:
        label = args.model_label
        if not label:
            raise SystemExit("--model_name_or_path requires --model_label")
        return args.model_name_or_path, label
    if args.model:
        return MODEL_CONFIGS[args.model], args.model
    raise SystemExit("Provide --model <label> or --model_name_or_path <path> --model_label <label>")


def output_path_for(output_dir, model_label, prompt_name, quant, dataset_name, sample) -> str:
    return os.path.join(
        output_dir, summary_filename(model_label, prompt_name, quant, dataset_name, sample)
    )


def run_summarize(args) -> None:
    model_name_or_path, model_label = resolve_model(args)

    # Decide which datasets still need work before paying to load the model.
    pending = []
    for dataset_name in args.datasets:
        out_path = output_path_for(
            args.output_dir, model_label, args.prompt_name,
            args.quantization_method, dataset_name, args.sample,
        )
        if args.skip_existing and os.path.exists(out_path):
            print(f"[skip] exists: {out_path}", flush=True)
        else:
            pending.append(dataset_name)

    if not pending:
        print("Nothing to summarize — all outputs already exist.", flush=True)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_cfg = PROMPT_CONFIGS[args.prompt_name]
    config = RunConfig(
        model_name_or_path=model_name_or_path,
        model_label=model_label,
        quantization_method=args.quantization_method,
        output_dir=args.output_dir,
        prompt_name=args.prompt_name,
        prompt_template=prompt_cfg["template"],
        max_new_tokens=prompt_cfg["max_new_tokens"],
    )
    model = SummarizationModel(config)

    datasets = load_datasets_streaming(sample=args.sample, seed=args.seed, names=pending)
    for dataset_name in pending:
        data = datasets[dataset_name]
        out_path = output_path_for(
            args.output_dir, model_label, args.prompt_name,
            args.quantization_method, dataset_name, args.sample,
        )
        print("=" * 80, flush=True)
        print(f"[{model_label} | {args.prompt_name} | {args.quantization_method}] "
              f"{dataset_name} -> {out_path}", flush=True)
        dataset_start = time.time()
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
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
                if idx == 1 or idx % 10 == 0:
                    target = args.sample if args.sample is not None else "?"
                    print(f"  sample {idx}/{target} | input_tokens={input_len} "
                          f"| generated_tokens={generated_len}", flush=True)
        elapsed_min = (time.time() - dataset_start) / 60
        print(f"  Done in {elapsed_min:.2f} min -> {out_path}", flush=True)


def run_baselines(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    datasets = load_datasets_streaming(sample=args.sample, seed=args.seed, names=args.datasets)

    # (display name, output prefix, callable taking a filtered datasets dict)
    specs = [
        ("Lead-1",   "Lead-1",   lambda d: run_lead(1, d, args.output_dir, args.sample)),
        ("Lead-3",   "Lead-3",   lambda d: run_lead(3, d, args.output_dir, args.sample)),
        ("TextRank", "TextRank", lambda d: run_textrank(d, args.output_dir, args.sample)),
        ("TF-IDF",   "TFIDF",    lambda d: run_tfidf(d, args.output_dir, args.sample)),
    ]

    for display, prefix, runner in specs:
        pending = {}
        for dataset_name in args.datasets:
            out_path = os.path.join(
                args.output_dir, baseline_filename(prefix, dataset_name, args.sample)
            )
            if args.skip_existing and os.path.exists(out_path):
                print(f"[skip] exists: {out_path}", flush=True)
            else:
                pending[dataset_name] = datasets[dataset_name]
        if not pending:
            print(f"--- {display}: nothing to do ---", flush=True)
            continue
        print(f"\n--- Running {display} baseline ---", flush=True)
        runner(pending)


def evaluate_all(output_dir: str, log_path: str, append: bool = False) -> None:
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.splitext(log_path)[0] + ".csv"
    if not append:
        # Start fresh so the results reflect exactly this evaluation run.
        for path in (log_path, csv_path):
            if os.path.exists(path):
                os.remove(path)

    evaluator = Evaluator()
    jsonl_files = sorted(glob.glob(os.path.join(output_dir, "*.jsonl")))
    if not jsonl_files:
        print("No .jsonl files found to evaluate.", flush=True)
        return

    print(f"\n{'=' * 80}", flush=True)
    print(f"Evaluating {len(jsonl_files)} output file(s) -> {log_path} (+ {csv_path})", flush=True)

    failed = []
    for file_path in jsonl_files:
        print(f"\n  [{os.path.basename(file_path)}]", flush=True)
        try:
            metrics = evaluator.run_and_log(file_path, log_path, csv_path)
        except Exception as e:
            # One bad/empty file must not abort scoring of all the others.
            print(f"    ERROR evaluating {os.path.basename(file_path)}: {e}", flush=True)
            failed.append(os.path.basename(file_path))
            continue
        for key, value in metrics.items():
            line = f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}"
            print(line, flush=True)

    print(f"\nAll metrics saved to {log_path} and {csv_path}", flush=True)
    if failed:
        print(f"WARNING: {len(failed)} file(s) failed and were skipped: {', '.join(failed)}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)  # seeds random / numpy / torch for reproducibility

    if args.task in ("summarize", "all"):
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("Warning: no GPU found, running on CPU (this will be slow)", flush=True)

    if args.task == "baselines":
        run_baselines(args)
    elif args.task == "summarize":
        run_summarize(args)
    elif args.task == "evaluate":
        evaluate_all(args.output_dir, args.log_path, append=args.append_eval)
    elif args.task == "all":
        run_baselines(args)
        run_summarize(args)
        evaluate_all(args.output_dir, args.log_path, append=args.append_eval)


if __name__ == "__main__":
    main()

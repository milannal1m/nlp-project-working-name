import argparse
import glob
import json
import os
import time

import torch

from dataset import extract_fields, load_datasets_streaming
from model import RunConfig, SummarizationModel
from baseline_lead import run_lead
from baseline_textrank import run_textrank
from baseline_tfidf import run_tfidf
from evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Run all summarization models and evaluate")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--quantization_method",
        type=str,
        choices=["None", "4bit", "8bit"],
        default="None",
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Articles per dataset. Omit to use the full test set.")
    parser.add_argument("--output_dir", type=str, default="./summaries")
    parser.add_argument("--log_path", type=str, default="evaluation.log")
    return parser.parse_args()


def run_llama(model: SummarizationModel, datasets: dict, output_dir: str, sample: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for dataset_name, data in datasets.items():
        output_path = os.path.join(
            output_dir,
            f"Llama_{model.config.quantization_method}_{dataset_name}_summaries.jsonl",
        )
        print("=" * 80, flush=True)
        print(f"[LLaMA] {dataset_name} (up to {sample} samples) -> {output_path}", flush=True)
        dataset_start = time.time()
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
                if idx == 1 or idx % 10 == 0:
                    print(
                        f"  sample {idx}/{sample} | input_tokens={input_len} | generated_tokens={generated_len}",
                        flush=True,
                    )
        elapsed_min = (time.time() - dataset_start) / 60
        print(f"  Done in {elapsed_min:.2f} min -> {output_path}", flush=True)


def evaluate_all(output_dir: str, log_path: str) -> None:
    evaluator = Evaluator()
    jsonl_files = sorted(glob.glob(os.path.join(output_dir, "*.jsonl")))
    if not jsonl_files:
        print("No .jsonl files found to evaluate.", flush=True)
        return

    print(f"\n{'=' * 80}", flush=True)
    print(f"Evaluating {len(jsonl_files)} output file(s) -> {log_path}", flush=True)

    for file_path in jsonl_files:
        print(f"\n  [{os.path.basename(file_path)}]", flush=True)
        metrics = evaluator.run_and_log(file_path, log_path)
        for key, value in metrics.items():
            line = f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}"
            print(line, flush=True)

    print(f"\nAll metrics saved to {log_path}", flush=True)


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Warning: no GPU found, running on CPU (this will be slow)", flush=True)

    label = f"sample={args.sample}" if args.sample is not None else "full test sets"
    print(f"\nLoading datasets ({label}, streaming)...", flush=True)
    datasets = load_datasets_streaming(sample=args.sample)

    print("\n--- Running Lead-1 and Lead-3 baselines ---", flush=True)
    for n in [1, 3]:
        run_lead(n, datasets, args.output_dir, args.sample)

    print("\n--- Running TextRank baseline ---", flush=True)
    run_textrank(datasets, args.output_dir, args.sample)

    print("\n--- Running TF-IDF baseline ---", flush=True)
    run_tfidf(datasets, args.output_dir, args.sample)

    print("\n--- Running LLaMA model ---", flush=True)
    config = RunConfig(
        model_name_or_path=args.model_name_or_path,
        quantization_method=args.quantization_method,
        output_dir=args.output_dir,
    )
    model = SummarizationModel(config)
    run_llama(model, datasets, args.output_dir, args.sample)

    evaluate_all(args.output_dir, args.log_path)


if __name__ == "__main__":
    main()

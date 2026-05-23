import argparse
import json
import os
import sys
import time

import torch

from dataset import extract_fields, load_all_datasets
from model import RunConfig, SummarizationModel


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Summarization with LLaMA")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--quantization_method",
        type=str,
        choices=["None", "4bit", "8bit"],
        default="None",
    )
    args = parser.parse_args()
    return RunConfig(
        model_name_or_path=args.model_name_or_path,
        quantization_method=args.quantization_method,
    )


def run(model: SummarizationModel, datasets: dict) -> None:
    os.makedirs(model.config.output_dir, exist_ok=True)

    for dataset_name, data in datasets.items():
        try:
            total = len(data)
        except TypeError:
            total = "unknown"

        output_path = os.path.join(
            model.config.output_dir,
            f"Llama_{model.config.quantization_method}_{dataset_name}_summaries.jsonl",
        )

        print("=" * 80, flush=True)
        print(f"Processing dataset: {dataset_name} ({total} samples) -> {output_path}", flush=True)

        dataset_start = time.time()

        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            for idx, item in enumerate(data, start=1):
                sample_start = time.time()
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
                    elapsed = time.time() - sample_start
                    print(
                        f"[{dataset_name}] sample {idx}/{total} | "
                        f"input_tokens={input_len} | generated_tokens={generated_len} | "
                        f"time={elapsed:.2f}s",
                        flush=True,
                    )

        elapsed_min = (time.time() - dataset_start) / 60
        print(f"Finished {dataset_name} in {elapsed_min:.2f} min -> {output_path}", flush=True)


def main() -> None:
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Warning: no GPU found, running on CPU (this will be slow)", flush=True)

    config = parse_args()
    model = SummarizationModel(config)
    datasets = load_all_datasets()
    run(model, datasets)


if __name__ == "__main__":
    main()

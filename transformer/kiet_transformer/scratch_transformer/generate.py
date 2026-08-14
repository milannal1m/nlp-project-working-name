import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset import extract_fields, load_datasets_streaming
from transformer.kiet_transformer.pipeline_config import OUTPUT_DIR, SAMPLE, output_filename, target_count

LABEL = "Transformer"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def checkpoint_path(dataset_name: str, checkpoint_dir: str = CHECKPOINT_DIR) -> str:
    return os.path.join(checkpoint_dir, f"transformer_{dataset_name}.pt")


def _count_records(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def generate_scratch(datasets: dict, output_dir: str, sample,
                     label: str = LABEL, checkpoint_dir: str = CHECKPOINT_DIR) -> None:
    import torch
    from transformer.kiet_transformer.scratch_transformer.summarizer import ScratchSummarizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Warning: no GPU found, running on CPU (slow)", flush=True)

    for dataset_name, data in datasets.items():
        ckpt = checkpoint_path(dataset_name, checkpoint_dir)
        if not os.path.exists(ckpt):
            print(f"[skip] {dataset_name}: no checkpoint at {ckpt} — train first", flush=True)
            continue

        target = target_count(dataset_name, sample)
        output_path = os.path.join(output_dir, output_filename(label, dataset_name))
        existing = _count_records(output_path)
        if existing >= target:
            print(f"[skip] {output_path} already has >= {target} samples", flush=True)
            continue

        mode = "a" if existing > 0 else "w"
        if existing > 0:
            print(f"[resume] {output_path} has {existing}/{target}; continuing from {existing + 1}", flush=True)

        model = ScratchSummarizer(ckpt, device=device)
        print("=" * 80, flush=True)
        print(f"[{label}] {dataset_name} (target {target}) -> {output_path}", flush=True)
        start = time.time()
        with open(output_path, mode, encoding="utf-8", newline="\n") as f:
            for idx, item in enumerate(data, start=1):
                if idx <= existing:
                    continue
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
                    print(f"  sample {idx}/{target} | in={input_len} | gen={generated_len}", flush=True)
        print(f"  Done in {(time.time() - start) / 60:.2f} min -> {output_path}", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate from-scratch Transformer summaries")
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = load_datasets_streaming(sample=args.sample)
    generate_scratch(datasets, args.output_dir, args.sample, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()

"""
TF-IDF baseline for news summarization.

Extraction is sumy's LSA summarizer: a truncated SVD over the document's TF-IDF
sentence-term matrix, which is why it lives under the `TFIDF` output prefix. That
prefix is baked into the summaries/ filenames and into tex_report.py's row keys,
so it stays as-is.

Usage:
    python src/baselines/tfidf.py --sample 20   # local testing
    python src/baselines/tfidf.py --sample 500  # full run on cluster
"""

import argparse
import json
import os
import sys
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Allow running both via `python src/main.py` and standalone `python src/baselines/tfidf.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import extract_fields, load_datasets_streaming
from naming import baseline_filename


def tfidf_summarize(text: str, n_sentences: int = 2) -> str:
    """Extract the top n_sentences by LSA over the TF-IDF matrix (see module docstring).

    n_sentences=2 matches the prompt given to the LLMs ("summarize in two sentences"),
    keeping the comparison fair.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    sentences = summarizer(parser.document, n_sentences)
    return " ".join(str(s) for s in sentences)


def run_tfidf(datasets: dict, output_dir: str, sample: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, data in datasets.items():
        output_path = os.path.join(
            output_dir,
            baseline_filename("TFIDF", dataset_name, sample),
        )

        print("=" * 80, flush=True)
        print(f"[TF-IDF] {dataset_name} (up to {sample} samples) -> {output_path}", flush=True)

        dataset_start = time.time()

        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            for idx, item in enumerate(data, start=1):
                news_text, ref_summary, qa_pairs = extract_fields(dataset_name, item)
                generated_text = tfidf_summarize(news_text, n_sentences=2)

                record = {
                    "news": news_text,
                    "reference_summary": ref_summary,
                    "generated_summary": generated_text,
                }
                if qa_pairs is not None:
                    record["qa_pairs"] = qa_pairs

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if idx == 1 or idx % 10 == 0:
                    print(f"  sample {idx}/{sample}", flush=True)

        elapsed = time.time() - dataset_start
        print(f"  Done in {elapsed:.1f}s -> {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF baseline summarization")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./summaries")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading datasets (streaming, sample={args.sample})...", flush=True)
    datasets = load_datasets_streaming(args.sample, seed=args.seed)
    print()

    run_tfidf(datasets, args.output_dir, args.sample)
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
"""
TextRank baseline for news summarization.

Usage:
    python baseline_textrank.py --sample 20   # local testing
    python baseline_textrank.py --sample 500  # full run on cluster
"""

import argparse
import json
import os
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

from dataset import extract_fields, load_datasets_streaming


def textrank_summarize(text: str, n_sentences: int = 2) -> str:
    """Summarize text by extracting the top n_sentences using TextRank.
    We use n_sentences=2 to match the prompt we give Llama ("summarize in two sentences"), keeping the comparison fair.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, n_sentences)
    return " ".join(str(s) for s in sentences)


def run_textrank(datasets: dict, output_dir: str, sample: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, data in datasets.items():
        output_path = os.path.join(
            output_dir,
            f"TextRank_{dataset_name}_summaries.jsonl",
        )

        print("=" * 80, flush=True)
        print(f"[TextRank] {dataset_name} (up to {sample} samples) -> {output_path}", flush=True)

        dataset_start = time.time()

        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            for idx, item in enumerate(data, start=1):
                news_text, ref_summary, qa_pairs = extract_fields(dataset_name, item)
                generated_text = textrank_summarize(news_text, n_sentences=2)

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
    parser = argparse.ArgumentParser(description="TextRank baseline summarization")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./summaries")
    args = parser.parse_args()

    print(f"Loading datasets (streaming, sample={args.sample})...", flush=True)
    datasets = load_datasets_streaming(args.sample)
    print()

    run_textrank(datasets, args.output_dir, args.sample)
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
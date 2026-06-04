import argparse
import json
import os
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

from dataset import extract_fields, load_datasets_streaming


def tfidf_summarize(text: str, n_sentences: int = 2) -> str:
    """Summarize text by extracting top n_sentences using TF-IDF scoring."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()  # sumy's TF-IDF based summarizer
    sentences = summarizer(parser.document, n_sentences)
    return " ".join(str(s) for s in sentences)


# Write 2-sentence TF-IDF/LSA summaries for each dataset as JSONL.
def run_tfidf(datasets: dict, output_dir: str, sample: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, data in datasets.items():
        output_path = os.path.join(
            output_dir,
            f"TFIDF_{dataset_name}_summaries.jsonl",
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


# CLI entry point: run the TF-IDF baseline over the datasets.
def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF baseline summarization")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./summaries")
    args = parser.parse_args()

    print(f"Loading datasets (streaming, sample={args.sample})...", flush=True)
    datasets = load_datasets_streaming(args.sample)
    print()

    run_tfidf(datasets, args.output_dir, args.sample)
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
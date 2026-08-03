import argparse

import json

import os

import re

import time

from dataset import extract_fields, load_datasets_streaming


def sent_tokenize(text: str) -> list[str]:

    """Split text into sentences using punctuation + capital-letter cues."""

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

    return [s.strip() for s in parts if s.strip()]


def lead_n(text: str, n: int) -> str:

    """Return the first n sentences of text as a single string."""

    return " ".join(sent_tokenize(text)[:n])


def run_lead(n: int, datasets: dict, output_dir: str, sample: int) -> None:

    os.makedirs(output_dir, exist_ok=True)

    baseline_name = f"Lead-{n}"


    for dataset_name, data in datasets.items():

        output_path = os.path.join(

            output_dir,

            f"{baseline_name}_{dataset_name}_summaries.jsonl",

        )


        print("=" * 80, flush=True)

        print(f"[{baseline_name}] {dataset_name} (up to {sample} samples) -> {output_path}", flush=True)


        dataset_start = time.time()


        with open(output_path, "w", encoding="utf-8", newline="\n") as f:

            for idx, item in enumerate(data, start=1):

                news_text, ref_summary, qa_pairs = extract_fields(dataset_name, item)

                generated_text = lead_n(news_text, n)


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

    parser = argparse.ArgumentParser(description="Lead-N baseline summarization")

    parser.add_argument(

        "--n",

        type=int,

        choices=[1, 3],

        default=None,

        help="Which Lead-N to run. Omit to run both Lead-1 and Lead-3.",

    )

    parser.add_argument(

        "--output_dir",

        type=str,

        default="./summaries",

        help="Where to write .jsonl output files",

    )

    parser.add_argument(

        "--sample",

        type=int,

        default=500,

        help=(

            "How many articles to process per dataset. "

            "Use 10-20 locally to save space and time. "

            "Use 500 on the cluster for the full benchmark."

        ),

    )

    args = parser.parse_args()


    ns = [args.n] if args.n is not None else [1, 3]


    print(f"Loading datasets (streaming, sample={args.sample})...", flush=True)

    datasets = load_datasets_streaming(args.sample)

    print()


    for n in ns:

        run_lead(n, datasets, args.output_dir, args.sample)


    print("\nAll done.", flush=True)


if __name__ == "__main__":

    main()

"""Show summaries that got a LERC score of 0 (or any low score), with their gold QA pairs.

Joins final_evaluation_results.jsonl (per-article scores from run_qa.py) with
master_evaluation_dataset.jsonl (the summaries and QA pairs) on article_id, and prints
examples so you can judge by hand whether a 0 means "the summary lost the information"
or "the metric refused to answer an answerable question".

Usage:
    python QA_Evaluation/scripts/show_zero_scores.py                       # 5 random zeros
    python QA_Evaluation/scripts/show_zero_scores.py -n 20
    python QA_Evaluation/scripts/show_zero_scores.py --config qwen2_1.5b_instruct_4bit_P1
    python QA_Evaluation/scripts/show_zero_scores.py --max-score 1.0       # also low non-zeros
    python QA_Evaluation/scripts/show_zero_scores.py --show-source
"""

import argparse
import json
import os
import random

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SUFFIX = "_lerc_score"


def first(value):
    """NewsQA fields are sometimes [text] instead of text."""
    return value[0] if isinstance(value, list) and value else value


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "final_evaluation_results.jsonl"))
    parser.add_argument("--master", default=os.path.join(RESULTS_DIR, "master_evaluation_dataset.jsonl"))
    parser.add_argument("-n", type=int, default=5, help="How many examples to print")
    parser.add_argument("--config", default=None, help="Only this summary column")
    parser.add_argument("--max-score", type=float, default=0.0,
                        help="Show cells with score <= this (default 0.0, i.e. only zeros)")
    parser.add_argument("--show-source", action="store_true", help="Also print the source article")
    parser.add_argument("--seed", type=int, default=0, help="Set for a reproducible sample")
    args = parser.parse_args()

    for path in (args.results, args.master):
        if not os.path.exists(path):
            raise SystemExit(f"{path} does not exist.")

    # Collect the (article, config) cells at or below the threshold.
    hits = []
    for line in open(args.results, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        aid = row.get("article_id")
        for key, value in row.items():
            if not key.endswith(SUFFIX) or value is None:
                continue
            config = key[: -len(SUFFIX)]
            if args.config and config != args.config:
                continue
            if float(value) <= args.max_score:
                hits.append((aid, config, float(value)))

    if not hits:
        raise SystemExit(f"No cells with score <= {args.max_score}"
                         f"{f' for config {args.config}' if args.config else ''}.")

    random.seed(args.seed)
    picked = random.sample(hits, min(args.n, len(hits)))
    wanted = {aid for aid, _, _ in picked}

    master = {}
    for line in open(args.master, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("article_id") in wanted:
            master[row["article_id"]] = row

    print(f"{len(hits)} cell(s) with LERC <= {args.max_score}; showing {len(picked)}")

    for i, (aid, config, score) in enumerate(picked, 1):
        row = master.get(aid)
        print("\n" + "=" * 78)
        print(f"[{i}/{len(picked)}]  {config}")
        print(f"         article_id: {aid}")
        print(f"         LERC score: {score}")
        print("=" * 78)

        if row is None:
            print("  (article not found in the master dataset)")
            continue

        summary = str(row.get(config, "<column missing>"))
        print(f"\n-- SUMMARY ({len(summary)} chars)\n{summary}")

        questions = row.get("human_questions", [])
        answers = row.get("human_answers", [])
        print(f"\n-- GOLD QA PAIRS ({len(questions)})")
        for n, (q, a) in enumerate(zip(questions, answers), 1):
            print(f"  {n}. Q: {first(q)}")
            print(f"     A: {first(a)}")

        if args.show_source:
            source = str(row.get("source_article", ""))
            print(f"\n-- SOURCE ARTICLE ({len(source)} chars)\n{source}")

    print()


if __name__ == "__main__":
    main()

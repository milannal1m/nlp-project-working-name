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
    python QA_Evaluation/scripts/show_zero_scores.py --verbatim-rate       # Calculate the verbatim match rate
"""

import argparse
import json
import os
import random
import csv

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SUFFIX = "_lerc_score"


def first(value):
    """NewsQA fields are sometimes [text] instead of text."""
    return value[0] if isinstance(value, list) and value else value


import csv # Ensure this is imported at the top of the file

def calculate_verbatim_rate(hits, master_data):
    """Calculates the Verbatim Match Rate for the 0.0 scores and exports to CSV."""
    global_zeros = len(hits)
    global_matches = 0
    config_stats = {}

    for aid, config, score in hits:
        if config not in config_stats:
            config_stats[config] = {"zeros": 0, "matches": 0}
            
        config_stats[config]["zeros"] += 1
        
        row = master_data.get(aid)
        if not row:
            continue
            
        summary = str(row.get(config, "")).lower()
        answers = row.get("human_answers", [])
        
        # Check if ANY of the human answers are verbatim inside the summary
        is_match = False
        for a in answers:
            ans_text = str(first(a)).lower().strip()
            if ans_text and ans_text in summary:
                is_match = True
                break
                
        if is_match:
            global_matches += 1
            config_stats[config]["matches"] += 1

    # --- Print to Terminal ---
    print("\n" + "=" * 50)
    print("=== GLOBAL VERBATIM MATCH RATE ===")
    print("=" * 50)
    print(f"Total scores analyzed (<= max-score): {global_zeros}")
    print(f"Times the exact answer was in summary: {global_matches}")
    if global_zeros > 0:
        print(f"Global Rate: {(global_matches / global_zeros) * 100:.2f}%\n")

    print("=" * 50)
    print("=== BREAKDOWN BY CONFIGURATION ===")
    print("=" * 50)
    
    # --- Save to CSV ---
    output_csv = os.path.join(RESULTS_DIR, "verbatim_analysis.csv")
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "zeros", "matches", "match_rate"])
        
        for config, stats in sorted(config_stats.items()):
            zeros = stats["zeros"]
            matches = stats["matches"]
            rate = (matches / zeros * 100) if zeros > 0 else 0
            
            # Print to screen
            print(f"{config}:")
            print(f"  - Scores found: {zeros}")
            print(f"  - Verbatim Matches: {matches}")
            print(f"  - Match Rate: {rate:.2f}%\n")
            
            # Write config to CSV
            writer.writerow([config, zeros, matches, f"{rate:.2f}"])
        
        # Add the Global Total row at the very bottom of the CSV
        global_rate = (global_matches / global_zeros * 100) if global_zeros > 0 else 0
        writer.writerow(["GLOBAL_TOTAL", global_zeros, global_matches, f"{global_rate:.2f}"])
            
    print(f"Success! Data exported to {output_csv}")


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
    parser.add_argument("--verbatim-rate", action="store_true", 
                        help="Calculate and print the verbatim match rate instead of showing examples")
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

    # Load the master dataset to memory if we need it for printing or verbatim checks
    master = {}
    for line in open(args.master, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        master[row.get("article_id")] = row

    if args.verbatim_rate:
        # Run the new statistical analysis
        calculate_verbatim_rate(hits, master)
    else:
        # Run the original teammate's manual review code
        random.seed(args.seed)
        picked = random.sample(hits, min(args.n, len(hits)))
        
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
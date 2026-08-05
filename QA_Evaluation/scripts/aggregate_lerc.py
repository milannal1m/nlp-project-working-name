"""Aggregate the per-article LERC scores into one row per config.

run_qa.py writes final_evaluation_results.jsonl: one row per article, with a
"<config>_lerc_score" column for every summary variation. This collapses that into
mean / std / n per config, so the 27 (or however many) variations become a ranking.

Articles the evaluator could not score appear as null and are counted separately
rather than silently treated as zeros -- a config with many nulls has a mean that
only describes the articles that worked.

Usage:
    python aggregate_lerc.py
    python aggregate_lerc.py --csv results/lerc_summary.csv
"""

import argparse
import csv
import json
import os
import statistics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SUFFIX = "_lerc_score"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "final_evaluation_results.jsonl"),
                        help="Per-article scores from run_qa.py")
    parser.add_argument("--csv", default=None, help="Also write the table as CSV")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        raise SystemExit(f"{args.results} does not exist. Run run_qa.py first.")

    scores, missing = {}, {}
    n_rows = 0
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_rows += 1
            for key, value in json.loads(line).items():
                if not key.endswith(SUFFIX):
                    continue
                config = key[: -len(SUFFIX)]
                scores.setdefault(config, [])
                missing.setdefault(config, 0)
                if value is None:
                    missing[config] += 1
                else:
                    scores[config].append(float(value))

    if not scores:
        raise SystemExit(f"No '*{SUFFIX}' columns found in {args.results}.")

    rows = []
    for config in scores:
        values = scores[config]
        rows.append({
            "config": config,
            "n": len(values),
            "missing": missing[config],
            "mean_lerc": statistics.mean(values) if values else None,
            "std_lerc": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_lerc": min(values) if values else None,
            "max_lerc": max(values) if values else None,
        })
    # Best first; configs with nothing scored sink to the bottom.
    rows.sort(key=lambda r: (r["mean_lerc"] is None, -(r["mean_lerc"] or 0)))

    width = max(len(r["config"]) for r in rows)
    print(f"{n_rows} article(s) in {os.path.basename(args.results)}\n")
    print(f"{'config':<{width}}  {'n':>6}  {'missing':>7}  {'mean':>7}  {'std':>7}  {'min':>6}  {'max':>6}")
    print("-" * (width + 48))
    for r in rows:
        if r["mean_lerc"] is None:
            print(f"{r['config']:<{width}}  {r['n']:>6}  {r['missing']:>7}  {'--':>7}  {'--':>7}  {'--':>6}  {'--':>6}")
            continue
        print(f"{r['config']:<{width}}  {r['n']:>6}  {r['missing']:>7}  "
              f"{r['mean_lerc']:>7.3f}  {r['std_lerc']:>7.3f}  "
              f"{r['min_lerc']:>6.2f}  {r['max_lerc']:>6.2f}")

    total_missing = sum(missing.values())
    if total_missing:
        print(f"\n[note] {total_missing} config/article cell(s) had no score (null) and are "
              f"excluded from the means.")

    if args.csv:
        out_dir = os.path.dirname(args.csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()

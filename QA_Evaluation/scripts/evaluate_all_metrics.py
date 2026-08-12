"""Score every QA-prep config with the reference-based metrics AND the LERC results,
into one CSV.

Per config (one Outputs/*.jsonl file) it computes BLEU, ROUGE-L, METEOR and BERTScore
against the human reference summary, then joins in lerc_mean / lerc_std from
final_evaluation_results.jsonl. One row per config, one CSV.

The reference summaries are not in the Outputs files -- data_preparation_pipeline.py
reads 'story', 'questions' and 'answers' from the gold file but never 'summary'. They
are joined back in here on article_id (md5 of the dateline-stripped story, the same
hash the prep pipeline builds).

The reference metrics come from src.evaluator.Evaluator, so the numbers are identical
to what `main.py --task evaluate` produces. That needs the nlp-project env (evaluate,
bert_score) and a GPU for BERTScore. --skip-metrics writes the LERC columns only and
runs anywhere.

Usage:
    python QA_Evaluation/scripts/evaluate_all_metrics.py                       # everything
    python QA_Evaluation/scripts/evaluate_all_metrics.py --skip-metrics        # LERC columns only
    python QA_Evaluation/scripts/evaluate_all_metrics.py --union               # no intersection
    python QA_Evaluation/scripts/evaluate_all_metrics.py --keep-inputs /tmp/ei # keep the temp files
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import statistics
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QA_DIR = os.path.dirname(SCRIPT_DIR)                 # QA_Evaluation/
REPO_ROOT = os.path.dirname(QA_DIR)                  # repo root, holds the top-level src/
# Must be the repo root exactly -- the src.evaluator / src.dataset imports below resolve
# the top-level src/ package, which QA_Evaluation/ would not provide.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LERC_SUFFIX = "_lerc_score"

CSV_FIELDS = [
    "config", "model", "quant", "prompt", "n_articles",
    "bleu", "rougeL", "meteor",
    "bertscore_f1_raw", "bertscore_f1_raw_std",
    "bertscore_f1_scaled", "bertscore_f1_scaled_std",
    "lerc_mean", "lerc_std", "lerc_n", "lerc_zeros",
    "error",
]


def article_id(text):
    """The hash data_preparation_pipeline.py assigns, over the cleaned story."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def split_config(config):
    """'llama_3.2_3b_instruct_4bit_P1' -> ('llama_3.2_3b_instruct', '4bit', 'P1')."""
    parts = config.rsplit("_", 2)
    return tuple(parts) if len(parts) == 3 else (config, "", "")


def load_references(gold_path):
    """{article_id: reference_summary} from the gold file."""
    from src.dataset import strip_dateline  # stdlib-only, safe in any env

    references = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            story, summary = row.get("story", ""), row.get("summary", "")
            if story and summary:
                references[article_id(strip_dateline(story))] = summary
    return references


def read_config_file(path):
    """{article_id: (story, generated_summary)}; blank summaries count as absent."""
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            aid, story = row.get("article_id"), row.get("story", "")
            summary = row.get("summary_text")
            if aid and story and summary and str(summary).strip():
                rows[aid] = (story, summary)
    return rows


def lerc_stats(results_path):
    """{config: {lerc_mean, lerc_std, lerc_n, lerc_zeros}} from run_qa.py's output.

    Nulls are counted, not treated as zeros -- they mean 'not scored', which is a
    different thing from 'scored zero'.
    """
    scores, zeros, nulls = {}, {}, {}
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            for key, value in json.loads(line).items():
                if not key.endswith(LERC_SUFFIX):
                    continue
                config = key[: -len(LERC_SUFFIX)]
                scores.setdefault(config, [])
                zeros.setdefault(config, 0)
                nulls.setdefault(config, 0)
                if value is None:
                    nulls[config] += 1
                    continue
                value = float(value)
                scores[config].append(value)
                if value == 0.0:
                    zeros[config] += 1

    out = {}
    for config, values in scores.items():
        out[config] = {
            "lerc_mean": statistics.mean(values) if values else None,
            "lerc_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "lerc_n": len(values),
            "lerc_zeros": zeros[config],
        }
    return out


def write_eval_input(path, article_ids, rows, references):
    """Write the three keys Evaluator.evaluate_metrics reads. Returns rows written."""
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for aid in article_ids:
            if aid not in rows or aid not in references:
                continue
            story, summary = rows[aid]
            f.write(json.dumps({
                "article_id": aid,
                "news": story,
                "reference_summary": references[aid],
                "generated_summary": summary,
            }) + "\n")
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", default=os.path.join(QA_DIR, "Outputs"))
    parser.add_argument("--gold", default=os.path.join(QA_DIR, "Datasets", "newsqasum_gold.jsonl"))
    parser.add_argument("--results", default=os.path.join(QA_DIR, "results", "final_evaluation_results.jsonl"),
                        help="Per-article LERC scores from run_qa.py")
    parser.add_argument("--csv", default=os.path.join(QA_DIR, "results", "qa_evaluation.csv"))
    parser.add_argument("--union", action="store_true",
                        help="Score every article a config covers. Default is the "
                             "intersection, so all configs are compared on the same articles.")
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip BLEU/ROUGE/METEOR/BERTScore; write the LERC columns only. "
                             "Runs without the evaluate/bert_score dependencies. Leaves the "
                             "reference-metric columns EMPTY, so pass --csv unless you mean "
                             "to overwrite a full CSV at the default path.")
    parser.add_argument("--keep-inputs", metavar="DIR", default=None,
                        help="Keep the converted per-config files instead of using a temp dir.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.outputs_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"No .jsonl files in {args.outputs_dir}")

    per_config = {}
    for path in files:
        config = os.path.basename(path).replace("temp_", "").replace(".jsonl", "")
        per_config[config] = read_config_file(path)
        print(f" -> {os.path.basename(path)}: {len(per_config[config])} articles", flush=True)

    id_sets = [set(rows) for rows in per_config.values()]
    keep = set().union(*id_sets) if args.union else set.intersection(*id_sets)
    print(f"\nArticle coverage ({'union' if args.union else 'intersection'}): "
          f"{len(keep)} of {max(len(s) for s in id_sets)}", flush=True)

    lerc = {}
    if os.path.exists(args.results):
        lerc = lerc_stats(args.results)
        print(f"LERC results loaded for {len(lerc)} config(s)")
    else:
        print(f"[warn] {args.results} not found -- LERC columns stay empty")

    references = {}
    evaluator = None
    if not args.skip_metrics:
        references = load_references(args.gold)
        print(f"Gold references loaded: {len(references)}")
        from src.evaluator import Evaluator  # heavy deps; only import when needed
        evaluator = Evaluator()

    work_dir = args.keep_inputs or tempfile.mkdtemp(prefix="qa_eval_inputs_")
    os.makedirs(work_dir, exist_ok=True)

    rows_out = []
    for config in sorted(per_config):
        row = {"config": config, "lerc_mean": "", "lerc_std": "", "lerc_n": "",
               "lerc_zeros": "", "error": ""}
        row["model"], row["quant"], row["prompt"] = split_config(config)
        row.update(lerc.get(config, {}))
        if config not in lerc and lerc:
            row["error"] = "no LERC column"

        if args.skip_metrics:
            row["n_articles"] = len(keep & set(per_config[config]))
            rows_out.append(row)
            continue

        eval_input = os.path.join(work_dir, f"{config}.jsonl")
        n = write_eval_input(eval_input, sorted(keep), per_config[config], references)
        row["n_articles"] = n
        if n == 0:
            row["error"] = (row["error"] + "; " if row["error"] else "") + "no scoreable rows"
            rows_out.append(row)
            continue

        print(f"\nScoring {config} ({n} articles)...", flush=True)
        try:
            row.update(evaluator.evaluate_metrics(eval_input))
        except Exception as e:                                    # keep going; note it
            row["error"] = (row["error"] + "; " if row["error"] else "") + f"{type(e).__name__}: {e}"
            print(f"  [!] failed: {e}", flush=True)
        rows_out.append(row)

    out_dir = os.path.dirname(args.csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {args.csv} ({len(rows_out)} rows)")
    if args.keep_inputs:
        print(f"Converted inputs kept in {work_dir}")
    failed = [r["config"] for r in rows_out if r["error"]]
    if failed:
        print(f"[warn] {len(failed)} config(s) with an error column: {', '.join(failed)}")


if __name__ == "__main__":
    main()

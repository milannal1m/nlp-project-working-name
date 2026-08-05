"""Pivot the per-variation summary files in Outputs/ into one master matrix.

Each Outputs/<model>_<quant>_<prompt>.jsonl holds one row per article, written by
data_preparation_pipeline.py. This flattens them into master_evaluation_dataset.jsonl:
one row per article, one column per variation, plus the four "foundation" fields the
QA evaluator needs.

The foundation fields are NOT optional and their names are load-bearing. qa_evaluator
treats every key that is not in FOUNDATION_KEYS as a candidate summary to score, so a
source article carried through under its original name ('story') would be silently
LERC-scored as if a model had produced it. Hence the rename to 'source_article'.

Usage:
    python summaries_merger.py                  # intersection of articles (default)
    python summaries_merger.py --union          # keep every article any variation covers
    python summaries_merger.py --outputs-dir DIR --output FILE
"""

import argparse
import glob
import json
import os

# Keys qa_evaluator.py reserves as metadata; everything else in a row is a summary.
FOUNDATION_KEYS = ("article_id", "source_article", "human_questions", "human_answers")


def foundation_from(row):
    """Map a data_preparation_pipeline row onto the names qa_evaluator expects."""
    return {
        "source_article": row.get("story", row.get("source_article", "")),
        "human_questions": row.get("human_questions", row.get("questions", [])),
        "human_answers": row.get("human_answers", row.get("answers", [])),
    }


def read_variation(file_path):
    """Return ({article_id: summary}, {article_id: foundation_dict}) for one file.

    Rows with a blank summary are treated as absent rather than as an empty summary,
    so a truncated or failed generation cannot enter the matrix as a scoreable ''.
    """
    summaries, foundations = {}, {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            article_id = row.get("article_id")
            summary = row.get("summary_text")
            if not article_id or not summary or not str(summary).strip():
                continue
            summaries[article_id] = summary
            if article_id not in foundations:
                foundations[article_id] = foundation_from(row)
    return summaries, foundations


def verify_merge(output_file, n_variations):
    """Report the shape of the written matrix and flag anything unscoreable."""
    print("\n--- Running Verification Check ---")
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                print("[WARNING] Master output file is empty — nothing will be scored.")
                return
            first_row = json.loads(first_line)

        keys = list(first_row.keys())
        expected = len(FOUNDATION_KEYS) + n_variations

        print(f"Total columns compiled in master row: {len(keys)}")
        print("Columns mapping preview:")
        for key in keys:
            val_preview = str(first_row[key])[:40].replace("\n", " ") + "..."
            print(f"   - {key}: {val_preview}")

        missing = [k for k in FOUNDATION_KEYS if not first_row.get(k)]
        if missing:
            print(f"\n[ERROR] Foundation fields missing or empty: {', '.join(missing)}.")
            print("        qa_evaluator will skip these articles and score nothing.")
        elif len(keys) == expected:
            print(f"\n[SUCCESS] Verification passed: {len(FOUNDATION_KEYS)} foundation "
                  f"fields + {n_variations} variations.")
        else:
            print(f"\n[WARNING] Expected {expected} columns "
                  f"({len(FOUNDATION_KEYS)} foundation + {n_variations} variations), "
                  f"found {len(keys)}. Coverage is ragged across variations.")
    except Exception as e:
        print(f"Verification failed to process file: {e}")


def main():
    qa_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # QA_Evaluation/

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", default=os.path.join(qa_dir, "Outputs"),
                        help="Directory of per-variation .jsonl files")
    parser.add_argument("--output", default=os.path.join(qa_dir, "results", "master_evaluation_dataset.jsonl"),
                        help="Path to write the master matrix to")
    parser.add_argument("--union", action="store_true",
                        help="Keep every article any variation covers. Default is the "
                             "intersection, so all variations are scored on the same "
                             "articles and their mean LERC stays comparable.")
    args = parser.parse_args()

    print(f"Targeting variations directory: {args.outputs_dir}")
    print(f"Targeting output master matrix: {args.output}\n")

    all_jsonl_files = sorted(glob.glob(os.path.join(args.outputs_dir, "*.jsonl")))
    variation_files = [f for f in all_jsonl_files
                       if os.path.basename(f) != os.path.basename(args.output)]

    print(f"Found {len(variation_files)} variation datasets to process.")
    if not variation_files:
        print("Error: 0 datasets found. Please check that your files are physically "
              "inside the directory shown above.")
        return

    per_variation = {}   # col_key -> {article_id: summary}
    foundations = {}     # article_id -> foundation dict (first file wins)

    for file_path in variation_files:
        filename = os.path.basename(file_path)
        col_key = filename.replace("temp_", "").replace(".jsonl", "")

        summaries, file_foundations = read_variation(file_path)
        per_variation[col_key] = summaries
        for article_id, found in file_foundations.items():
            if article_id not in foundations:
                foundations[article_id] = found

        print(f" -> {filename}: {len(summaries)} articles (column: {col_key})")

    # Which articles make it into the matrix.
    id_sets = [set(s) for s in per_variation.values()]
    if args.union:
        article_ids = set().union(*id_sets)
        mode = "union"
    else:
        article_ids = set.intersection(*id_sets)
        mode = "intersection"

    # An article is only scoreable if its source and QA pairs survived the merge.
    scoreable = [a for a in article_ids
                 if foundations.get(a, {}).get("source_article")
                 and foundations.get(a, {}).get("human_questions")
                 and foundations.get(a, {}).get("human_answers")]
    dropped = len(article_ids) - len(scoreable)

    widest = max(len(s) for s in id_sets)
    print(f"\nArticle coverage ({mode}): {len(article_ids)} of {widest} "
          f"in the widest variation.")
    if not args.union and len(article_ids) < widest:
        thinnest = min(per_variation, key=lambda k: len(per_variation[k]))
        print(f"  Capped by '{thinnest}' ({len(per_variation[thinnest])} articles). "
              f"Use --union to keep partial coverage instead.")
    if dropped:
        print(f"  [WARNING] Dropped {dropped} article(s) with no source text or QA pairs.")

    print(f"\nWriting master evaluation matrix to {args.output}")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for article_id in sorted(scoreable):
            row = {"article_id": article_id}
            row.update(foundations[article_id])
            for col_key, summaries in per_variation.items():
                if article_id in summaries:
                    row[col_key] = summaries[article_id]
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(scoreable)} articles.")
    print("\n--- Compilation Complete ---")
    verify_merge(args.output, len(variation_files))


if __name__ == "__main__":
    main()

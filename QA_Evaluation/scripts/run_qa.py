"""Score the merged summary matrix with QAFactEval (LERC).

Reads QA_Evaluation/results/master_evaluation_dataset.jsonl (produced by
summaries_merger.py) and writes QA_Evaluation/results/final_evaluation_results.jsonl.
Paths are anchored to the repository layout, not the working directory, so it behaves
the same whether it is launched from QA_Evaluation/ or from the repo root.

Usage:
    python run_qa.py                     # full dataset
    python run_qa.py --max-articles 5    # smoke test: confirms the merge and the
                                         # model weights are wired up, in minutes
"""

import argparse
import os

from qa_evaluator import QAFactEvaluator

QA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # QA_Evaluation/
RESULTS_DIR = os.path.join(QA_DIR, "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", default=os.path.join(RESULTS_DIR, "master_evaluation_dataset.jsonl"),
                        help="Merged matrix from summaries_merger.py")
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "final_evaluation_results.jsonl"),
                        help="Where to write the per-article scores")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Stop after N articles (smoke test). Default: whole dataset.")
    args = parser.parse_args()

    print("==================================================")
    print("  Starting QAFactEval Pipeline Execution")
    print("==================================================")
    print(f"Target Dataset: {args.master}")
    if args.max_articles is not None:
        print(f"TEST MODE: limiting to {args.max_articles} article(s).")

    if not os.path.exists(args.master):
        print(f"\nCRITICAL ERROR: {args.master} does not exist. "
              f"Run summaries_merger.py first.")
        return

    # 1. Initialize the architecture built in qa_evaluator.py
    print("\n[1/2] Initializing the Evaluator...")
    try:
        evaluator = QAFactEvaluator(master_file=args.master)
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to initialize evaluator. Check environment. {e}")
        return

    # 2. Trigger the evaluation
    print("\n[2/2] Executing Factual Consistency Scoring...")
    try:
        final_output_path = evaluator.run_qa_evaluation(output_file=args.output,
                                                        max_articles=args.max_articles)
    except Exception as e:
        print(f"\nCRITICAL ERROR: Pipeline failed during execution. {e}")
        return

    print("==================================================")
    print("PIPELINE COMPLETE.")
    print(f"Scores successfully saved to: {final_output_path}")
    print("==================================================")


if __name__ == "__main__":
    main()

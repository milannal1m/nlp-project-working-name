import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformer.kiet_transformer.pipeline_config import DATASETS, METRICS_DIR, OUTPUT_DIR, SAMPLE, output_filename
from transformer.kiet_transformer.run_evaluation import evaluate_target

LABEL = "Transformer"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate from-scratch Transformer summaries")
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--metrics_dir", type=str, default=METRICS_DIR)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)
    for dataset in args.datasets:
        filename = output_filename(LABEL, dataset)
        if not os.path.exists(os.path.join(OUTPUT_DIR, filename)):
            print(f"[skip] {dataset}: no summary file {filename} — generate first", flush=True)
            continue

        target = {"label": LABEL, "dataset": dataset, "filename": filename}
        result = evaluate_target(target, args.sample)

        out_path = os.path.join(args.metrics_dir, f"{LABEL}__{dataset}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

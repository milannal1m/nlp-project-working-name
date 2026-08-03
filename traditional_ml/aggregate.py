"""Collate traditional-ML metric JSONs into a CSV + Markdown report (+ charts).

Self-contained (no shared import). Also prints a head-to-head vs the published
Lead-3 bar so the score goal is easy to check. Those Lead-3 numbers are the
published baseline measured on the same full test splits by the main pipeline,
hard-coded here so this package stays standalone.

    python -m traditional_ml.aggregate
"""


import argparse

import csv

import glob

import json

import os


from . import config


METRICS = [("bleu", "BLEU"), ("rougeL", "ROUGE-L"), ("meteor", "METEOR"), ("bertscore_f1", "BERTScore-F1")]

LENGTH_COLS = ["avg_news_len", "avg_ref_len", "avg_gen_len", "std_gen_len", "avg_gen_sents", "avg_compression"]

CSV_COLUMNS = (["label", "dataset", "num_samples"] + [k for k, _ in METRICS]

               + ["bertscore_f1_std"] + LENGTH_COLS)

PRETTY = {"cnn_dailymail": "CNN/DailyMail", "xsum": "XSum"}


LEAD3 = {

    "cnn_dailymail": {"bleu": 0.1150, "rougeL": 0.2428, "meteor": 0.3854, "bertscore_f1": 0.8691},

    "xsum": {"bleu": 0.0077, "rougeL": 0.1155, "meteor": 0.2089, "bertscore_f1": 0.8546},

}

MODEL_ORDER = [m["label"] for m in config.MODELS]


def load_rows(metrics_dir: str) -> list[dict]:

    rows = []

    for path in sorted(glob.glob(os.path.join(metrics_dir, "*.json"))):

        with open(path, "r", encoding="utf-8") as f:

            rows.append(json.load(f))

    return rows


def _order_key(r: dict):

    label = r.get("label", "")

    idx = MODEL_ORDER.index(label) if label in MODEL_ORDER else len(MODEL_ORDER)

    return (idx, r.get("dataset", ""))


def write_csv(rows: list[dict], path: str) -> None:

    with open(path, "w", newline="", encoding="utf-8") as f:

        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")

        w.writeheader()

        for r in sorted(rows, key=_order_key):

            out = {c: (f"{v:.6f}" if isinstance(v := r.get(c), float) else ("" if v is None else v))

                   for c in CSV_COLUMNS}

            w.writerow(out)

    print(f"[ok] CSV -> {path}")


def _fmt(v):

    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def write_markdown(rows: list[dict], path: str) -> None:

    with open(path, "w", encoding="utf-8") as f:

        f.write("# Traditional-ML Summarization Results\n\n")

        f.write("Extractive sentence classifiers (LogReg / NB / XGBoost) with validation-tuned "

                "selection length. Compared against the published **Lead-3** bar.\n\n")

        for ds in config.DATASETS:

            ds_rows = sorted([r for r in rows if r.get("dataset") == ds], key=_order_key)

            if not ds_rows:

                continue

            f.write(f"## {PRETTY.get(ds, ds)}\n\n")

            header = " | ".join(n for _, n in METRICS)

            f.write(f"| Model | {header} | Gen Len |\n|---|{'---|' * (len(METRICS) + 1)}\n")

            f.write(f"| _Lead-3 (bar)_ | " + " | ".join(_fmt(LEAD3[ds].get(k)) for k, _ in METRICS)

                    + " | — |\n")

            for r in ds_rows:

                cells = " | ".join(_fmt(r.get(k)) for k, _ in METRICS)

                f.write(f"| {r.get('label')} | {cells} | {_fmt(r.get('avg_gen_len'))} |\n")

            f.write("\n")


            beats = [r["label"] for r in ds_rows

                     if isinstance(r.get("rougeL"), float) and r["rougeL"] > LEAD3[ds]["rougeL"]]

            f.write(f"**Beats Lead-3 ROUGE-L ({LEAD3[ds]['rougeL']:.4f}):** "

                    f"{', '.join(beats) if beats else 'none'}\n\n")

    print(f"[ok] MD  -> {path}")


def write_charts(rows: list[dict], charts_dir: str) -> None:

    try:

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        import numpy as np

    except ImportError as e:

        print(f"[warn] matplotlib/numpy missing, skipping charts: {e}")

        return

    os.makedirs(charts_dir, exist_ok=True)

    labels = [l for l in MODEL_ORDER if any(r.get("label") == l for r in rows)]

    if not labels:

        return

    x = np.arange(len(labels))

    width = 0.8 / max(len(config.DATASETS), 1)


    def val(label, ds, key):

        for r in rows:

            if r.get("label") == label and r.get("dataset") == ds:

                v = r.get(key)

                return v if isinstance(v, (int, float)) else np.nan

        return np.nan


    for key, name in METRICS:

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))

        for i, ds in enumerate(config.DATASETS):

            ax.bar(x + i * width, [val(l, ds, key) for l in labels], width, label=PRETTY.get(ds, ds))

            ax.axhline(LEAD3[ds].get(key, np.nan), ls="--", lw=1, alpha=0.5)

        ax.set_xticks(x + width * (len(config.DATASETS) - 1) / 2)

        ax.set_xticklabels(labels, rotation=30, ha="right")

        ax.set_ylabel(name); ax.set_title(f"{name} (dashed = Lead-3 bar)"); ax.legend()

        fig.tight_layout(); fig.savefig(os.path.join(charts_dir, f"{key}.png"), dpi=130); plt.close(fig)

    print(f"[ok] charts -> {charts_dir}/")


def main():

    p = argparse.ArgumentParser(description="Aggregate traditional-ML metrics")

    p.add_argument("--metrics_dir", type=str, default=config.METRICS_DIR)

    p.add_argument("--results_dir", type=str, default=config.RESULTS_DIR)

    p.add_argument("--charts_dir", type=str, default=config.CHARTS_DIR)

    args = p.parse_args()


    os.makedirs(args.results_dir, exist_ok=True)

    rows = load_rows(args.metrics_dir)

    if not rows:

        raise SystemExit(f"No metric JSONs in {args.metrics_dir}. Run traditional_ml.evaluate first.")

    print(f"Loaded {len(rows)} metric files.")

    write_csv(rows, os.path.join(args.results_dir, "results.csv"))

    write_markdown(rows, os.path.join(args.results_dir, "results.md"))

    write_charts(rows, args.charts_dir)


if __name__ == "__main__":

    main()

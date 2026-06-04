"""Render a single overview figure comparing every model across the headline metrics.

Reads ``results/results.csv`` (produced by ``aggregate.py``) and writes
``results/charts/comparison.png`` — a 2x3 grid of horizontal bar charts, one per
metric, with each model's score averaged over the two datasets and baselines vs
LLMs colour-coded. Run after aggregation:  python make_comparison_chart.py
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline_config import MODELS, RESULTS_DIR, CHARTS_DIR

# (csv column, display name) for each panel; Gen length is descriptive, not "better".
PANELS = [
    ("bleu", "BLEU"),
    ("rougeL", "ROUGE-L"),
    ("meteor", "METEOR"),
    ("bertscore_f1", "BERTScore-F1"),
    ("summac", "SummaC"),
    ("avg_gen_len", "Gen length (tokens)"),
]
KIND = {m.label: m.kind for m in MODELS}
ORDER = [m.label for m in MODELS]


# Collect each metric's values per model label across all dataset rows.
def load_values(csv_path: Path) -> dict:
    acc = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, _ in PANELS:
                v = row.get(key, "")
                if v not in ("", "nan", "None", None):
                    acc[row["label"]][key].append(float(v))
    return acc


# Draw one horizontal bar panel: models sorted by score, LLM vs baseline colours.
def draw_panel(ax, labels, values, name) -> None:
    pairs = sorted(zip(labels, values),
                   key=lambda p: (np.isnan(p[1]), -(0 if np.isnan(p[1]) else p[1])))
    ls = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]
    colors = ["#4C72B0" if KIND.get(l) == "llm" else "#DD8452" for l in ls]
    ax.barh(range(len(ls)), [0 if np.isnan(v) else v for v in vs], color=colors)
    ax.set_yticks(range(len(ls)))
    ax.set_yticklabels(ls, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(name, fontsize=12, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for i, v in enumerate(vs):
        if not np.isnan(v):
            ax.text(v, i, f" {v:.0f}" if v >= 10 else f" {v:.3f}", va="center", fontsize=8)


# Entry point: build the 2x3 comparison figure and save it to the charts dir.
def main() -> None:
    csv_path = Path(RESULTS_DIR) / "results.csv"
    acc = load_values(csv_path)
    labels = [l for l in ORDER if l in acc]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, name) in zip(axes.flat, PANELS):
        values = [float(np.mean(acc[l][key])) if acc[l][key] else float("nan") for l in labels]
        draw_panel(ax, labels, values, name)

    handles = [plt.Rectangle((0, 0), 1, 1, color="#4C72B0"),
               plt.Rectangle((0, 0), 1, 1, color="#DD8452")]
    fig.legend(handles, ["LLM", "Baseline"], loc="upper right", fontsize=11)
    fig.suptitle("Model comparison — mean over CNN/DailyMail + XSum (higher is better; "
                 "Gen length is descriptive)", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(CHARTS_DIR, exist_ok=True)
    out = Path(CHARTS_DIR) / "comparison.png"
    fig.savefig(out, dpi=130)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()

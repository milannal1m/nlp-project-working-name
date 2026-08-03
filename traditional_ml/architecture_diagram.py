"""Render the traditional-ML pipeline architecture to a PNG.

Two panels on one canvas:
  1. Method anatomy — how a summary is produced: the supervised training path
     (oracle labels -> features -> hyperparameter search -> fit -> validation
     calibration) and the inference path (score -> select top-k), bridged by the
     trained model.
  2. Code pipeline  — module + data flow across the self-contained package:
     train -> generate -> evaluate -> aggregate.

    python -m traditional_ml.architecture_diagram          # -> traditional_ml/architecture.png
    python -m traditional_ml.architecture_diagram --out foo.png
"""


import argparse

import os


import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch


BLUE = "#4C72B0"

ORANGE = "#DD8452"

GREEN = "#55A868"

PURPLE = "#8172B3"

GRAY = "#B0B0B0"

NEUTRAL = "#ECECEC"

INK = "#222222"

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")


def _tint(hex_color: str, alpha: float) -> tuple:

    """Blend a hex colour toward white by ``1 - alpha`` (a soft fill)."""

    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))

    return tuple(c * alpha + (1 - alpha) for c in (r, g, b))


def box(ax, cx, cy, w, h, text, face, edge=INK, fontsize=9, weight="normal"):

    ax.add_patch(FancyBboxPatch(

        (cx - w / 2, cy - h / 2), w, h,

        boxstyle="round,pad=0.15,rounding_size=0.7",

        linewidth=1.4, edgecolor=edge, facecolor=face, zorder=2,

    ))

    ax.text(cx, cy, text, ha="center", va="center",

            fontsize=fontsize, weight=weight, color=INK, zorder=3, linespacing=1.35)


def arrow(ax, p1, p2, color=INK, lw=1.7, style="-|>", ls="-"):

    ax.annotate("", xy=p2, xytext=p1, zorder=1,

                arrowprops=dict(arrowstyle=style, color=color, lw=lw,

                                linestyle=ls, shrinkA=3, shrinkB=3))


def draw_method(ax):

    """Panel 1 — the supervised training path and the inference path."""

    ax.text(25, 95, "①  Method anatomy — supervised extractive summarization",

            ha="center", fontsize=12.5, weight="bold", color=INK)

    ax.text(13, 89, "Training  (full train split)", ha="center", fontsize=10,

            weight="bold", color="#555555")

    ax.text(38, 89, "Inference  (test article)", ha="center", fontsize=10,

            weight="bold", color="#555555")


    tx = 13

    train = [

        (82, 21, 7.5, "Train split (full)\nCNN/DM 287k · XSum 204k", NEUTRAL, "#9A9A9A", "normal"),

        (70, 21, 7.5, "Sentence split + greedy\nROUGE-1+2 oracle → y ∈ {0,1}", _tint(GREEN, 0.5), GREEN, "bold"),

        (56.5, 22, 9.5, "Features / sentence\n• dense ~14 (position, length,\n  TF-IDF centrality, overlap)\n• hashed BoW (2¹⁸)", _tint(BLUE, 0.42), BLUE, "normal"),

        (43, 21, 7.5, "Hyperparameter search\nGrid / Randomized · 3-fold CV", _tint(PURPLE, 0.42), PURPLE, "normal"),

        (31, 21, 7.5, "Fit 3 classifiers\nLogReg · NB · XGBoost  (full split)", _tint(BLUE, 0.62), BLUE, "bold"),

        (18.5, 22, 8, "Validation calibration\ntune k + trigram blocking\n(balanced R-L+METEOR+BLEU)", _tint(ORANGE, 0.5), ORANGE, "normal"),

    ]

    for cy, w, h, text, face, edge, weight in train:

        box(ax, tx, cy, w, h, text, face, edge=edge, fontsize=8.2, weight=weight)

    for (y1, *_), (y2, *_) in zip(train, train[1:]):

        arrow(ax, (tx, y1 - 4.0), (tx, y2 + 4.0), color="#555555")


    box(ax, 25.5, 8, 24, 7, "model.joblib  +  meta.json\n(best_params, best_k, use_blocking)",

        _tint(GREEN, 0.28), edge=GREEN, fontsize=8.2, weight="bold")

    arrow(ax, (tx, 14.6), (20, 9.5), color=GREEN)


    ix = 38

    infer = [

        (82, 21, 7.5, "Test article", NEUTRAL, "#9A9A9A", "normal"),

        (69, 22, 8.5, "Sentence split → features\n(identical to training)", _tint(BLUE, 0.32), BLUE, "normal"),

        (55, 21, 7.5, "Sentence scores\nmodel.predict_proba(·)[:, 1]", _tint(GRAY, 0.45), "#7A7A7A", "normal"),

        (41, 22, 8.5, "Select top-k + trigram block\nselect_sentences() · doc order", _tint(ORANGE, 0.5), ORANGE, "bold"),

        (27, 21, 7.5, "Extractive summary\n(k sentences, verbatim)", _tint(GREEN, 0.5), GREEN, "bold"),

    ]

    for cy, w, h, text, face, edge, weight in infer:

        box(ax, ix, cy, w, h, text, face, edge=edge, fontsize=8.2, weight=weight)

    for (y1, *_), (y2, *_) in zip(infer, infer[1:]):

        arrow(ax, (ix, y1 - 4.0), (ix, y2 + 4.0), color="#555555")


    arrow(ax, (31, 9), (ix, 51), color=GREEN, lw=1.5, style="-|>", ls=(0, (4, 2)))

    ax.text(34.5, 30, "trained\nmodel", ha="center", fontsize=7.5, style="italic", color=GREEN)


def draw_pipeline(ax):

    """Panel 2 — module + data flow across the self-contained package."""

    ax.text(76, 95, "②  Code pipeline  (self-contained package)",

            ha="center", fontsize=12.5, weight="bold", color=INK)


    cx = 71

    stages = [

        (85, "Datasets: CNN/DailyMail · XSum\nstreamed via data.py (own loader)", GREEN),

        (71, "train.py\nextract-cache → search → fit → calibrate", GREEN),

        (57, "generate.py\nscore → tuned top-k + trigram blocking", GREEN),

        (43, "evaluate.py\nBLEU · ROUGE-L · METEOR · BERTScore", GREEN),

        (29, "aggregate.py\nglob metrics → CSV · MD · charts", GREEN),

        (15, "results/results.csv · results.md\ncharts/  (vs. Lead-3 bar)", NEUTRAL),

    ]

    for y, text, key in stages:

        face = _tint(GREEN, 0.45) if key == GREEN else NEUTRAL

        edge = GREEN if key == GREEN else "#9A9A9A"

        box(ax, cx, y, 30, 8.5, text, face, edge=edge, fontsize=8.3,

            weight="bold" if key == GREEN else "normal")

    for (y1, *_), (y2, *_) in zip(stages, stages[1:]):

        arrow(ax, (cx, y1 - 4.6), (cx, y2 + 4.6))


    chips = [

        (78, "data/features/\nchunk_*.jsonl.gz"),

        (64, "models/<ds>/<name>/\nmodel.joblib + meta.json"),

        (50, "summaries/\nML-*_<ds>.jsonl"),

        (36, "results/metrics/\nML-*__<ds>.json"),

    ]

    for y, text in chips:

        box(ax, 92.5, y, 13, 6.6, text, NEUTRAL, edge="#9A9A9A", fontsize=7.1)

        arrow(ax, (cx + 15, y), (85.7, y), color="#9A9A9A", lw=1.2, style="-|>", ls=(0, (3, 2)))


    ax.text(71, 6,

            "Imports nothing from the shared repo (own data.py + evaluate.py) — merges to main as a pure addition.",

            ha="center", fontsize=7.3, style="italic", color="#666666", wrap=True)


def draw_legend(fig):

    handles = [

        plt.Rectangle((0, 0), 1, 1, facecolor=_tint(GREEN, 0.5), edgecolor=GREEN),

        plt.Rectangle((0, 0), 1, 1, facecolor=_tint(BLUE, 0.5), edgecolor=BLUE),

        plt.Rectangle((0, 0), 1, 1, facecolor=_tint(PURPLE, 0.45), edgecolor=PURPLE),

        plt.Rectangle((0, 0), 1, 1, facecolor=_tint(ORANGE, 0.5), edgecolor=ORANGE),

        plt.Rectangle((0, 0), 1, 1, facecolor=_tint(GRAY, 0.45), edgecolor="#7A7A7A"),

        plt.Rectangle((0, 0), 1, 1, facecolor=NEUTRAL, edgecolor="#9A9A9A"),

    ]

    labels = ["Local module / supervision", "Features", "Hyperparameter search",

              "Calibration / selection", "Scoring", "Data artifact"]

    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,

               frameon=False, bbox_to_anchor=(0.5, 0.005))


def main():

    parser = argparse.ArgumentParser(description="Render the traditional-ML architecture diagram")

    parser.add_argument("--out", default=DEFAULT_OUT, help="Output PNG path")

    parser.add_argument("--dpi", type=int, default=150)

    args = parser.parse_args()


    fig, (ax_method, ax_pipe) = plt.subplots(1, 2, figsize=(16, 9))

    for ax in (ax_method, ax_pipe):

        ax.set_ylim(0, 100)

        ax.axis("off")

    ax_method.set_xlim(0, 50)

    ax_pipe.set_xlim(50, 100)


    draw_method(ax_method)

    draw_pipeline(ax_pipe)

    draw_legend(fig)


    fig.suptitle("Traditional-ML extractive summarization — method & pipeline",

                 fontsize=15, weight="bold", y=0.97)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.07, wspace=0.02)


    fig.savefig(args.out, dpi=args.dpi)

    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":

    main()

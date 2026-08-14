import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
GRAY = "#B0B0B0"
NEUTRAL = "#ECECEC"
INK = "#222222"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")


def _tint(hex_color: str, alpha: float) -> tuple:
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


def draw_model(ax):
    ax.text(24, 92, "①  Model anatomy — encoder–decoder  (model.py)",
            ha="center", fontsize=12.5, weight="bold", color=INK)

    enc_x, dec_x = 15, 35

    box(ax, enc_x, 12, 17, 7, "src ids\n<bos> article… <eos>\n(≤ max_src_len)", NEUTRAL, fontsize=8)
    box(ax, enc_x, 24, 17, 6.5, "Token + Positional\nEmbedding", _tint(BLUE, 0.30), edge=BLUE, fontsize=8.5)
    box(ax, enc_x, 45, 18, 24,
        "6× Encoder block\n\n• Multi-Head\n  Self-Attention\n• Add & Norm\n• Feed-Forward (ReLU)\n• Add & Norm",
        _tint(BLUE, 0.55), edge=BLUE, fontsize=8.5, weight="bold")
    arrow(ax, (enc_x, 15.5), (enc_x, 20.7), color=BLUE)
    arrow(ax, (enc_x, 27.3), (enc_x, 33), color=BLUE)

    box(ax, dec_x, 12, 17, 7, "trg ids (shifted)\n<bos> summary…\n(teacher forcing)", NEUTRAL, fontsize=8)
    box(ax, dec_x, 24, 17, 6.5, "Token + Positional\nEmbedding", _tint(ORANGE, 0.30), edge=ORANGE, fontsize=8.5)
    box(ax, dec_x, 45, 18, 24,
        "6× Decoder block\n\n• Masked Self-Attn\n• Add & Norm\n• Cross-Attn ← enc\n• Feed-Forward (ReLU)\n• Add & Norm",
        _tint(ORANGE, 0.55), edge=ORANGE, fontsize=8.5, weight="bold")
    box(ax, dec_x, 66, 17, 6.5, "Linear → vocab logits\n(fc_out)", _tint(ORANGE, 0.30), edge=ORANGE, fontsize=8.5)
    box(ax, dec_x, 78, 18, 7,
        "CE loss vs. trg[1:]  (train)\ngreedy argmax loop (infer)", NEUTRAL, fontsize=8)
    arrow(ax, (dec_x, 15.5), (dec_x, 20.7), color=ORANGE)
    arrow(ax, (dec_x, 27.3), (dec_x, 33), color=ORANGE)
    arrow(ax, (dec_x, 57), (dec_x, 62.7), color=ORANGE)
    arrow(ax, (dec_x, 69.3), (dec_x, 74.5), color=ORANGE)

    arrow(ax, (enc_x + 9, 52), (dec_x - 9, 46), color="#555555", lw=1.6, style="-|>", ls=(0, (4, 2)))
    ax.text(25, 55, "enc_out", ha="center", fontsize=8, style="italic", color="#555555")

    ax.text(25, 5.2,
            "Deviations kept verbatim from the lecture source: learned positional embeddings, "
            "attention ÷ √embed_size, causal-only target mask.",
            ha="center", fontsize=7.3, style="italic", color="#666666", wrap=True)


def draw_pipeline(ax):
    ax.text(76, 92, "②  Code pipeline  (self-contained package)",
            ha="center", fontsize=12.5, weight="bold", color=INK)

    cx = 72
    stages = [
        (85, "Datasets: CNN/DailyMail · XSum\nstreamed via dataset.py (shared)", GRAY, "shared"),
        (71, "train.py  →  data.py · model.py\nteacher-forced seq2seq training", GREEN, "local"),
        (57, "generate.py  →  summarizer.py\ngreedy autoregressive decode", GREEN, "local"),
        (43, "evaluate.py  →  run_evaluation (shared)\nBLEU · ROUGE-L · METEOR · BERTScore · SummaC", GREEN, "local"),
        (29, "aggregate.py  (shared, read-only)\nglob metrics → tables", GRAY, "shared"),
        (15, "results.md  ·  results.csv\n(Transformer row appears here)", NEUTRAL, "artifact"),
    ]
    fills = {GREEN: _tint(GREEN, 0.45), GRAY: _tint(GRAY, 0.55), NEUTRAL: NEUTRAL}
    edges = {GREEN: GREEN, GRAY: "#7A7A7A", NEUTRAL: "#9A9A9A"}
    for y, text, key, _ in stages:
        box(ax, cx, y, 30, 8.5, text, fills[key], edge=edges[key], fontsize=8.3,
            weight="bold" if key == GREEN else "normal")
    for (y1, *_), (y2, *_) in zip(stages, stages[1:]):
        arrow(ax, (cx, y1 - 4.6), (cx, y2 + 4.6))

    chips = [
        (64, "checkpoints/\ntransformer_<ds>.pt"),
        (50, "summaries/\nTransformer_<ds>.jsonl"),
        (36, "results/metrics/\nTransformer__<ds>.json"),
    ]
    for y, text in chips:
        box(ax, 92.5, y, 12.5, 6.5, text, NEUTRAL, edge="#9A9A9A", fontsize=7.2)
        arrow(ax, (cx + 15, y), (86.3, y), color="#9A9A9A", lw=1.2, style="-|>", ls=(0, (3, 2)))


def draw_legend(fig):
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_tint(BLUE, 0.55), edgecolor=BLUE),
        plt.Rectangle((0, 0), 1, 1, facecolor=_tint(ORANGE, 0.55), edgecolor=ORANGE),
        plt.Rectangle((0, 0), 1, 1, facecolor=_tint(GREEN, 0.45), edgecolor=GREEN),
        plt.Rectangle((0, 0), 1, 1, facecolor=_tint(GRAY, 0.55), edgecolor="#7A7A7A"),
        plt.Rectangle((0, 0), 1, 1, facecolor=NEUTRAL, edgecolor="#9A9A9A"),
    ]
    labels = ["Encoder", "Decoder", "Local module", "Reused shared file (read-only)", "Data artifact"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.005))


def main():
    parser = argparse.ArgumentParser(description="Render the Transformer architecture diagram")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    fig, (ax_model, ax_pipe) = plt.subplots(1, 2, figsize=(16, 9))
    for ax in (ax_model, ax_pipe):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")
    ax_model.set_xlim(0, 50)
    ax_pipe.set_xlim(50, 100)

    draw_model(ax_model)
    draw_pipeline(ax_pipe)
    draw_legend(fig)

    fig.suptitle("From-scratch Transformer for summarization — architecture & pipeline",
                 fontsize=15, weight="bold", y=0.97)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.07, wspace=0.02)

    fig.savefig(args.out, dpi=args.dpi)
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()

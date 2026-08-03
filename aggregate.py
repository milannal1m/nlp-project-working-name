import csv

import glob

import json

import os

from pathlib import Path


from pipeline_config import MODELS, DATASETS, RESULTS_DIR, METRICS_DIR, CHARTS_DIR, SAMPLE


METRICS = [

    ("bleu", "BLEU", True),

    ("rougeL", "ROUGE-L", True),

    ("meteor", "METEOR", True),

    ("bertscore_f1", "BERTScore-F1", True),

    ("summac", "SummaC", True),

    ("qa_eval", "QAFactEval", True),

]

STD_OF = {"bertscore_f1": "bertscore_f1_std", "summac": "summac_std", "qa_eval": "qa_eval_std"}


LENGTH_COLS = [

    "avg_news_len", "avg_ref_len", "avg_gen_len", "std_gen_len",

    "avg_gen_sents", "avg_compression",

]


CSV_COLUMNS = (

    ["label", "dataset", "num_samples"]

    + [m for m, _, _ in METRICS]

    + ["bertscore_f1_std", "summac_std", "qa_eval_std"]

    + LENGTH_COLS

)


MODEL_ORDER = [m.label for m in MODELS]

PRETTY_DATASET = {"cnn_dailymail": "CNN/DailyMail", "xsum": "XSum"}


def load_results() -> list[dict]:

    rows = []

    for path in sorted(glob.glob(os.path.join(METRICS_DIR, "*.json"))):

        with open(path, "r", encoding="utf-8") as f:

            rows.append(json.load(f))

    return rows


def _order_key(row: dict) -> tuple:

    label = row.get("label", "")

    idx = MODEL_ORDER.index(label) if label in MODEL_ORDER else len(MODEL_ORDER)

    return (idx, row.get("dataset", ""))


def write_csv(rows: list[dict], path: Path) -> None:

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")

        writer.writeheader()

        for row in sorted(rows, key=_order_key):

            out = {}

            for col in CSV_COLUMNS:

                val = row.get(col)

                out[col] = f"{val:.6f}" if isinstance(val, float) else ("" if val is None else val)

            writer.writerow(out)

    print(f"[ok] CSV  -> {path}")


def _fmt(val, std=None) -> str:

    if val is None:

        return "—"

    if std is not None:

        return f"{val:.4f} ± {std:.4f}"

    return f"{val:.4f}"


def write_markdown(rows: list[dict], path: Path) -> None:

    by_dataset = {ds: [r for r in rows if r.get("dataset") == ds] for ds in DATASETS}

    labels = sorted({r.get("label") for r in rows}, key=lambda l: MODEL_ORDER.index(l) if l in MODEL_ORDER else 99)


    with open(path, "w", encoding="utf-8") as f:

        f.write("# Summarization Experiment Results\n\n")

        f.write(f"**Models**: {len(labels)} &nbsp;|&nbsp; **Datasets**: "

                f"{', '.join(PRETTY_DATASET.get(d, d) for d in DATASETS)} &nbsp;|&nbsp; "

                f"**Samples/dataset**: {SAMPLE}\n\n")

        f.write("**Metrics**: BLEU, ROUGE-L, METEOR, BERTScore-F1, SummaC, QAFactEval "

                "(ROUGE-1 / ROUGE-2 excluded). Higher is better for all.\n\n")

        f.write("---\n\n")


        metric_header = " | ".join(name for _, name, _ in METRICS)

        sep = "|".join(["------:"] * len(METRICS))


        for ds in DATASETS:

            ds_rows = sorted(by_dataset.get(ds, []), key=_order_key)

            if not ds_rows:

                continue

            f.write(f"## {PRETTY_DATASET.get(ds, ds)}\n\n")

            f.write(f"| Model | {metric_header} | Gen Len | Compression |\n")

            f.write(f"|-------|{sep}|--------:|------------:|\n")

            for r in ds_rows:

                cells = []

                for key, _, _ in METRICS:

                    cells.append(_fmt(r.get(key), r.get(STD_OF.get(key)) if STD_OF.get(key) else None))

                gen_len = r.get("avg_gen_len")

                comp = r.get("avg_compression")

                gen_len_s = f"{gen_len:.1f}" if gen_len is not None else "—"

                comp_s = f"{comp:.4f}" if comp is not None else "—"

                f.write(f"| {r.get('label')} | {' | '.join(cells)} | {gen_len_s} | {comp_s} |\n")

            f.write("\n")


            f.write("**Best per metric:** ")

            bests = []

            for key, name, hib in METRICS:

                scored = [r for r in ds_rows if isinstance(r.get(key), (int, float))]

                if not scored:

                    continue

                best = max(scored, key=lambda r: r[key])

                bests.append(f"{name}: **{best['label']}** ({best[key]:.4f})")

            f.write("; ".join(bests) + "\n\n---\n\n")


        errs = [(r.get("label"), r.get("dataset"), r.get("errors")) for r in rows if r.get("errors")]

        if errs:

            f.write("## Notes — metrics that did not run\n\n")

            for label, ds, err in errs:

                for group, msg in err.items():

                    f.write(f"- `{label}` / {PRETTY_DATASET.get(ds, ds)} — **{group}**: {msg}\n")

            f.write("\n")


        f.write("## Charts\n\n")

        for key, name, _ in METRICS:

            f.write(f"![{name}](charts/{key}.png)\n\n")

        for ds in DATASETS:

            f.write(f"![Heatmap {PRETTY_DATASET.get(ds, ds)}](charts/heatmap_{ds}.png)\n\n")


    print(f"[ok] MD   -> {path}")


def write_charts(rows: list[dict], charts_dir: Path) -> None:

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

    x = np.arange(len(labels))

    width = 0.8 / max(len(DATASETS), 1)


    def value(label, ds, key):

        for r in rows:

            if r.get("label") == label and r.get("dataset") == ds:

                v = r.get(key)

                return v if isinstance(v, (int, float)) else np.nan

        return np.nan


    for key, name, _ in METRICS:

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))

        for i, ds in enumerate(DATASETS):

            vals = [value(l, ds, key) for l in labels]

            ax.bar(x + i * width, vals, width, label=PRETTY_DATASET.get(ds, ds))

        ax.set_xticks(x + width * (len(DATASETS) - 1) / 2)

        ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_ylabel(name)

        ax.set_title(f"{name} by model")

        ax.legend()

        ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()

        fig.savefig(charts_dir / f"{key}.png", dpi=130)

        plt.close(fig)


    for ds in DATASETS:

        matrix = np.array([[value(l, ds, key) for key, _, _ in METRICS] for l in labels], dtype=float)

        if np.all(np.isnan(matrix)):

            continue


        col_min = np.nanmin(matrix, axis=0)

        col_max = np.nanmax(matrix, axis=0)

        denom = np.where(col_max - col_min == 0, 1, col_max - col_min)

        norm = (matrix - col_min) / denom


        fig, ax = plt.subplots(figsize=(max(6, len(METRICS)), max(4, len(labels) * 0.5)))

        im = ax.imshow(norm, aspect="auto", cmap="viridis")

        ax.set_xticks(np.arange(len(METRICS)))

        ax.set_xticklabels([n for _, n, _ in METRICS], rotation=30, ha="right")

        ax.set_yticks(np.arange(len(labels)))

        ax.set_yticklabels(labels)

        ax.set_title(f"{PRETTY_DATASET.get(ds, ds)} — normalised metric heatmap")

        for r_i in range(len(labels)):

            for c_i in range(len(METRICS)):

                if not np.isnan(matrix[r_i, c_i]):

                    ax.text(c_i, r_i, f"{matrix[r_i, c_i]:.3f}", ha="center", va="center",

                            color="white" if norm[r_i, c_i] < 0.5 else "black", fontsize=8)

        fig.colorbar(im, ax=ax, label="normalised (per column)")

        fig.tight_layout()

        fig.savefig(charts_dir / f"heatmap_{ds}.png", dpi=130)

        plt.close(fig)


    print(f"[ok] charts -> {charts_dir}/")


def main() -> None:

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = load_results()

    if not rows:

        raise SystemExit(f"No metric JSONs found in {METRICS_DIR}/. Run evaluation first.")

    print(f"Loaded {len(rows)} metric files.")

    write_csv(rows, Path(RESULTS_DIR) / "results.csv")

    write_markdown(rows, Path(RESULTS_DIR) / "results.md")

    write_charts(rows, Path(CHARTS_DIR))

    print("Done.")


if __name__ == "__main__":

    main()

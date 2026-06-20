"""Unsupervised cluster analysis of the summarization results.

Groups the 10 models x 2 datasets = 20 evaluation configurations into
data-driven clusters and cross-validates them against a conceptual taxonomy
(model family / extraction style). Runs three scopes:

    * combined        - all 20 (model, dataset) points
    * cnn_dailymail   - 10 models on CNN/DailyMail
    * xsum            - 10 models on XSum

For each scope it standardises the metric matrix, then:
    * Ward hierarchical linkage  -> dendrogram PNG
    * K-Means (k chosen by silhouette over k=2..6) -> cluster ids
    * PCA(2)                      -> annotated scatter PNG

It also computes the "proof" statistics the write-up cites
(quantization-invariance, intra/inter-family distance, Phi-3 prompt-leak rate)
and writes:

    results/clusters.csv         - per-config cluster id + conceptual group
    results/cluster_stats.json   - the proof statistics
    results/charts/cluster_*.png - dendrogram + PCA figures

Reads results/results.csv (produced by aggregate.py). Quality/behaviour
features only; SummaC is held out of the primary clustering (it is unreliable)
and analysed separately as a robustness check.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pipeline_config import DATASETS, MODELS, RESULTS_DIR, CHARTS_DIR, output_filename

# --------------------------------------------------------------------------
# Feature sets & taxonomy
# --------------------------------------------------------------------------
# Primary clustering features (z-score standardised). SummaC is deliberately
# excluded — it is bimodal/unreliable — and only added for the robustness check.
QUALITY = ["bleu", "rougeL", "meteor", "bertscore_f1"]
BEHAVIOURAL = ["avg_gen_len", "avg_gen_sents", "avg_compression"]
PRIMARY = QUALITY + BEHAVIOURAL
NUMERIC_COLS = PRIMARY + ["summac"]

MODEL_ORDER = [m.label for m in MODELS]
PRETTY_DATASET = {"cnn_dailymail": "CNN/DailyMail", "xsum": "XSum"}

# Conceptual taxonomy: model identity -> group (for cross-validation).
CONCEPTUAL_GROUP = {
    "Lead-1": "Positional-extractive",
    "Lead-3": "Positional-extractive",
    "TextRank": "Salience-extractive",
    "TFIDF": "Salience-extractive",
    "Llama_None": "Abstractive-LLM",
    "Llama_4bit": "Abstractive-LLM",
    "Llama_8bit": "Abstractive-LLM",
    "Phi-3_None": "Prompt-degraded-LLM",
    "Phi-3_4bit": "Prompt-degraded-LLM",
    "Phi-3_8bit": "Prompt-degraded-LLM",
}

# Quantization families: same base model at three precisions.
FAMILIES = {
    "Llama": ["Llama_None", "Llama_4bit", "Llama_8bit"],
    "Phi-3": ["Phi-3_None", "Phi-3_4bit", "Phi-3_8bit"],
}

# Markers that betray a leaked instruction template in a generated summary.
LEAK_MARKERS = ["## Your task", "Document:", "Based on the provided document",
                "## Instruction", "## Response", "## Summary"]

RANDOM_STATE = 0


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
# Read results/results.csv into a list of dicts with numeric fields as floats.
def load_rows(results_dir: str) -> list[dict]:
    path = Path(results_dir) / "results.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run aggregate.py first.")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for c in NUMERIC_COLS:
                r[c] = float(r[c]) if r.get(c) not in (None, "") else np.nan
            rows.append(r)
    return rows


# Order rows by the config's declared model order, then dataset.
def _order_key(r: dict) -> tuple:
    label = r.get("label", "")
    idx = MODEL_ORDER.index(label) if label in MODEL_ORDER else len(MODEL_ORDER)
    return (idx, r.get("dataset", ""))


# Build the (rows, point-labels, feature-matrix) for one scope.
# scope is "combined" (all 20) or a dataset name (10 models).
def scope_matrix(rows: list[dict], scope: str, features: list[str]):
    if scope == "combined":
        sel = sorted(rows, key=_order_key)
        names = [f"{r['label']} / {('CNN' if r['dataset']=='cnn_dailymail' else 'XSum')}" for r in sel]
    else:
        sel = sorted([r for r in rows if r["dataset"] == scope], key=_order_key)
        names = [r["label"] for r in sel]
    matrix = np.array([[r[c] for c in features] for r in sel], dtype=float)
    return sel, names, matrix


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------
# K-Means with k chosen by best silhouette over k=2..min(6, n-1).
# Returns (labels, chosen_k, {k: silhouette}).
def kmeans_best_k(z: np.ndarray) -> tuple:
    n = z.shape[0]
    sils = {}
    best_k, best_labels, best_s = 2, None, -1.0
    for k in range(2, min(6, n - 1) + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(z)
        s = float(silhouette_score(z, km.labels_))
        sils[k] = round(s, 4)
        if s > best_s:
            best_k, best_labels, best_s = k, km.labels_, s
    return best_labels, best_k, sils


# Cross-tab: for each data-driven cluster, the conceptual groups it contains.
def crosstab(names: list[str], labels: np.ndarray, point_label_to_model) -> dict:
    table: dict[str, dict[str, int]] = {}
    for name, cl in zip(names, labels):
        group = CONCEPTUAL_GROUP.get(point_label_to_model(name), "?")
        table.setdefault(f"cluster_{int(cl)}", {}).setdefault(group, 0)
        table[f"cluster_{int(cl)}"][group] += 1
    return table


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def save_dendrogram(z, names, scope, charts_dir, plt):
    link = linkage(z, method="ward")
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.6), 5))
    dendrogram(link, labels=names, ax=ax, leaf_rotation=90, color_threshold=0.7 * link[:, 2].max())
    title = "all 20 configs" if scope == "combined" else PRETTY_DATASET.get(scope, scope)
    ax.set_title(f"Ward hierarchical clustering — {title}")
    ax.set_ylabel("Ward linkage distance (standardised features)")
    fig.tight_layout()
    out = Path(charts_dir) / f"cluster_dendrogram_{scope}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return link


def save_pca(z, names, labels, scope, charts_dir, plt):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    xy = pca.fit_transform(z)
    ev = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10", s=90, edgecolors="black", linewidths=0.5)
    for (x, y), name in zip(xy, names):
        ax.annotate(name, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    title = "all 20 configs" if scope == "combined" else PRETTY_DATASET.get(scope, scope)
    ax.set_title(f"PCA of standardised metrics (colour = K-Means cluster) — {title}")
    ax.set_xlabel(f"PC1 ({ev[0]:.0f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0f}% var)")
    ax.grid(linestyle="--", alpha=0.3)
    fig.colorbar(sc, ax=ax, label="K-Means cluster id")
    fig.tight_layout()
    fig.savefig(Path(charts_dir) / f"cluster_pca_{scope}.png", dpi=130)
    plt.close(fig)
    return ev.tolist()


# --------------------------------------------------------------------------
# Proof statistics
# --------------------------------------------------------------------------
# Spread (max-min) across the 3 precision variants of each family, per dataset.
def quantization_spread(rows: list[dict]) -> dict:
    metrics = ["bleu", "rougeL", "meteor", "bertscore_f1", "summac"]
    out: dict = {}
    for fam, members in FAMILIES.items():
        out[fam] = {}
        for ds in DATASETS:
            vals = {m: [] for m in metrics}
            for label in members:
                r = next((x for x in rows if x["label"] == label and x["dataset"] == ds), None)
                if r:
                    for m in metrics:
                        vals[m].append(r[m])
            out[fam][ds] = {m: round(float(max(v) - min(v)), 6) for m, v in vals.items() if v}
    return out


# Mean intra-family vs inter-family distance in standardised PRIMARY space, per dataset.
def family_distance(rows: list[dict]) -> dict:
    out: dict = {}
    for ds in DATASETS:
        sel, names, matrix = scope_matrix(rows, ds, PRIMARY)
        z = StandardScaler().fit_transform(matrix)
        idx = {name: i for i, name in enumerate(names)}
        out[ds] = {}
        for fam, members in FAMILIES.items():
            fam_idx = [idx[m] for m in members if m in idx]
            other_idx = [i for i in range(len(names)) if i not in fam_idx]
            intra = [np.linalg.norm(z[a] - z[b]) for a in fam_idx for b in fam_idx if a < b]
            inter = [np.linalg.norm(z[a] - z[b]) for a in fam_idx for b in other_idx]
            out[ds][fam] = {
                "mean_intra_family_dist": round(float(np.mean(intra)), 4),
                "mean_inter_family_dist": round(float(np.mean(inter)), 4),
                "ratio_inter_over_intra": round(float(np.mean(inter) / np.mean(intra)), 2),
            }
    return out


# Fraction of generated summaries containing a leaked-template marker, per config.
# Scanned over results/eval_inputs/<label>_<dataset>_summaries.jsonl.
def leak_rates(results_dir: str) -> dict:
    inputs_dir = Path(results_dir) / "eval_inputs"
    out: dict = {}
    for label in ["Phi-3_None", "Phi-3_4bit", "Phi-3_8bit", "Llama_None", "Llama_4bit", "Llama_8bit"]:
        out[label] = {}
        for ds in DATASETS:
            path = inputs_dir / output_filename(label, ds)
            if not path.exists():
                continue
            n, leaked, lengths = 0, 0, []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    gen = json.loads(line).get("generated_summary", "") or ""
                    n += 1
                    lengths.append(len(gen.split()))
                    if any(mk in gen for mk in LEAK_MARKERS):
                        leaked += 1
            if n:
                out[label][ds] = {
                    "n_scanned": n,
                    "leak_rate": round(leaked / n, 4),
                    "avg_words": round(float(np.mean(lengths)), 1),
                }
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(f"matplotlib required for figures: {e}")

    rows = load_rows(RESULTS_DIR)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    print(f"Loaded {len(rows)} (model, dataset) rows from {RESULTS_DIR}/results.csv\n")

    scopes = ["combined", "cnn_dailymail", "xsum"]
    cluster_assignments: list[dict] = []  # for clusters.csv
    stats: dict = {"scopes": {}}

    # For combined scope, map a point label like "Llama_8bit / XSum" back to the model.
    def to_model(point_name: str) -> str:
        return point_name.split(" / ")[0]

    for scope in scopes:
        # --- primary clustering (no SummaC) ---
        sel, names, matrix = scope_matrix(rows, scope, PRIMARY)
        z = StandardScaler().fit_transform(matrix)
        labels, k, sils = kmeans_best_k(z)
        link = save_dendrogram(z, names, scope, CHARTS_DIR, plt)
        ward = fcluster(link, t=k, criterion="maxclust")
        ev = save_pca(z, names, labels, scope, CHARTS_DIR, plt)
        ct = crosstab(names, labels, to_model)

        # --- robustness check: same clustering WITH SummaC added ---
        _, _, matrix_s = scope_matrix(rows, scope, PRIMARY + ["summac"])
        zs = StandardScaler().fit_transform(matrix_s)
        labels_s, k_s, _ = kmeans_best_k(zs)
        # agreement: fraction of point-pairs that stay together / apart in both labelings
        same_primary = labels[:, None] == labels[None, :]
        same_summac = labels_s[:, None] == labels_s[None, :]
        n = len(names)
        pair_agree = (same_primary == same_summac).sum() - n  # exclude diagonal
        rand_index = round(pair_agree / (n * (n - 1)), 3)

        stats["scopes"][scope] = {
            "n_points": n,
            "kmeans_k": k,
            "silhouette_by_k": sils,
            "pca_explained_var_pct": [round(e, 1) for e in ev],
            "crosstab_cluster_x_conceptual": ct,
            "summac_robustness": {
                "k_with_summac": k_s,
                "rand_index_vs_primary": rand_index,
            },
        }

        print(f"== {scope} ==  n={n}  k*={k}  silhouette={sils}  PCA var={['%.0f%%' % e for e in ev]}")
        clusters_print: dict[int, list[str]] = {}
        for name, cl, wd in zip(names, labels, ward):
            clusters_print.setdefault(int(cl), []).append(name)
            cluster_assignments.append({
                "scope": scope,
                "label": to_model(name) if scope == "combined" else name,
                "dataset": (name.split(" / ")[1] if scope == "combined" else scope),
                "conceptual_group": CONCEPTUAL_GROUP.get(to_model(name), "?"),
                "kmeans_cluster": int(cl),
                "ward_cluster": int(wd),
            })
        for cl in sorted(clusters_print):
            print(f"   cluster {cl}: {', '.join(clusters_print[cl])}")
        print()

    # --- proof statistics ---
    stats["quantization_spread"] = quantization_spread(rows)
    stats["family_distance"] = family_distance(rows)
    stats["prompt_leak_rates"] = leak_rates(RESULTS_DIR)

    # --- write artifacts ---
    clusters_csv = Path(RESULTS_DIR) / "clusters.csv"
    with open(clusters_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "label", "dataset", "conceptual_group",
                                          "kmeans_cluster", "ward_cluster"])
        w.writeheader()
        w.writerows(cluster_assignments)
    print(f"[ok] cluster assignments -> {clusters_csv}")

    stats_json = Path(RESULTS_DIR) / "cluster_stats.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[ok] proof statistics   -> {stats_json}")
    print(f"[ok] figures            -> {CHARTS_DIR}/cluster_*.png")

    # headline proof numbers to console
    print("\n--- proof highlights ---")
    for fam in FAMILIES:
        for ds in DATASETS:
            sp = stats["quantization_spread"][fam][ds]
            print(f"{fam} {ds}: BERTScore spread {sp['bertscore_f1']:.4f}, ROUGE-L spread {sp['rougeL']:.4f}")
    for label in ["Phi-3_None", "Llama_None"]:
        for ds in DATASETS:
            lr = stats["prompt_leak_rates"].get(label, {}).get(ds)
            if lr:
                print(f"{label} {ds}: leak_rate {lr['leak_rate']:.2%} (n={lr['n_scanned']}), avg {lr['avg_words']} words")


if __name__ == "__main__":
    main()

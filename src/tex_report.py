"""Generate the LaTeX tables and figures under ``results/tex/`` from the CSV.

Everything printed here is derived *only* from ``results/evaluation.csv`` (the
output of ``src/evaluator.py``). The layout, sizing and captions are meant to
match the hand-built templates that live in ``results/tex/`` verbatim, so the
same file can be regenerated whenever the numbers change.

Run it manually:

    python src/tex_report.py                       # reads results/evaluation.csv
    python src/tex_report.py --csv path/to.csv --out-dir results/tex

It writes five files:

  * ``PerPromptTables.tex``            – one metrics table per prompt (P1/P2/P3)
  * ``PerPrecisionTables.tex``         – one metrics table per precision
  * ``GraphsPerMetrics.tex``           – pgfplots line charts (ROUGE-L, BERTScore)
  * ``PromptSensitivityPerPrecision.tex`` – prompt spread per precision
  * ``QuantizationGapPerPrompt.tex``   – 16-bit-vs-quantized gap per prompt
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field

from naming import summary_filename


# --- Experiment grid (order here is the order rows/columns appear in the tex) ---
# NB: MODELS holds the CSV/filename tokens (e.g. "Llama_P1_16bit_..."), while
# MODEL_LABEL maps each to the name printed in the tables/figures. The full
# official names (Llama-3.2-3B-Instruct, Phi-3-mini-4k-instruct) and their HF
# ids belong in the paper's setup section; these are the compact labels.
MODELS = ["Llama", "Phi"]
MODEL_LABEL = {"Llama": "Llama-3.2-3B", "Phi": "Phi-3-mini"}
PROMPTS = ["P1", "P2", "P3"]
QUANTS = ["16bit", "8bit", "4bit"]

# CSV dataset token (as it appears in the filename) -> (sample tag, printed label)
DATASETS = [
    ("cnn_dailymail", "full", "CNN/DM"),
    ("xsum", "full", "XSum"),
    ("xu_cnndm", "500", "Xu-CNN/DM"),
    ("xu_xsum", "500", "Xu-XSum"),
]

# Pretty precision labels ("16-bit" style), shared by tables, graphs and derived tables.
PRECISION_LABEL = {"16bit": "16-bit", "8bit": "8-bit", "4bit": "4-bit"}

# CSV column names for the metrics we print.
COL_ROUGEL = "rougeL"
COL_BERT = "bertscore_f1_raw"
COL_BERT_SCALED = "bertscore_f1_scaled"


# --- Captions (edit here; one place for all tables/figures) ----------------
# The <...> placeholders are filled per table before writing:
#   <prompt>    -> P1 / P2 / P3
#   <precision> -> 16-bit / 8-bit / 4-bit
#   <metric>    -> ROUGE-L / BERTScore
CAPTIONS = {
    "per_prompt": (
        r"Evaluation metrics for prompt <prompt>. "
        r"BERTScore$_{\mathrm{sc}}$ is the baseline-rescaled F1. "
        r"Best value per model and dataset across precisions is in \textbf{bold}."
    ),
    "per_precision": (
        r"Evaluation metrics at <precision> precision. "
        r"BERTScore$_{\mathrm{sc}}$ is the baseline-rescaled F1. "
        r"Best value per model and dataset across prompts is in \textbf{bold}."
    ),
    "graph": (
        r"<metric> across precision (16-bit, 8-bit, 4-bit) for each prompt. "
        r"Columns are models, rows are datasets. Vertical distance between "
        r"lines is prompt sensitivity; each line's slope from 16-bit to 4-bit "
        r"is that prompt's degradation under quantization."
    ),
    "prompt_sensitivity": (
        r"Prompt sensitivity per precision. Each cell is the spread "
        r"(max$-$min in points, $\times 100$) across the three prompts at that "
        r"precision. Larger values mean the score depends more on prompt wording; "
        r"the most prompt-sensitive precision per row is in \textbf{bold}."
    ),
    "quantization_gap": (
        r"Quantization gap by prompt. Each cell is $\Delta = \text{16-bit} "
        r"- \text{quantized (8-bit or 4-bit)}$ in points ($\times 100$); positive means quality "
        r"drops under quantization. For each model, dataset, and metric, the prompt "
        r"with the smallest gap magnitude is in \textbf{bold} (per precision). This "
        r"isolates whether more structured prompting (P1$\to$P3) narrows the "
        r"full-precision--quantized gap."
    ),
}


def _fmt3(v: float) -> str:
    return f"{v:.3f}"


def _fmt4(v: float) -> str:
    return f"{v:.4f}"


def _fmt_signed2(v: float) -> str:
    return f"{v:+.2f}"


def _bold(s: str) -> str:
    return r"\textbf{" + s + "}"


@dataclass
class TexReport:
    """Reads the evaluation CSV and emits the LaTeX files under ``out_dir``."""

    csv_path: str
    out_dir: str
    rows: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load()

    # --- data access -----------------------------------------------------
    def _load(self) -> None:
        with open(self.csv_path, newline="") as f:
            for row in csv.DictReader(f):
                self.rows[row["file"]] = row

    def value(self, model: str, prompt: str, quant: str, ds_token: str,
              sample: str, col: str) -> float:
        """One metric cell for a (model, prompt, quant, dataset) summary run."""
        fname = summary_filename(model, prompt, quant, ds_token, sample)
        if fname not in self.rows:
            raise KeyError(f"missing row in {self.csv_path}: {fname}")
        raw = self.rows[fname][col]
        if raw is None or raw == "":
            raise ValueError(f"empty {col} for {fname}")
        return float(raw)

    def _write(self, name: str, lines: list[str], trailing_newline: bool = False) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, name)
        with open(path, "w") as f:
            f.write("\n".join(lines) + ("\n" if trailing_newline else ""))
        return path

    # --- public entry point ----------------------------------------------
    def generate_all(self) -> list[str]:
        return [
            self.per_prompt_tables(),
            self.per_precision_tables(),
            self.graphs_per_metrics(),
            self.prompt_sensitivity_per_precision(),
            self.quantization_gap_per_prompt(),
        ]

    # =====================================================================
    #  Shared builder for the two "main" metric tables. Both have identical
    #  layout: Model -> (3 inner groups) -> 4 datasets, with ROUGE-L /
    #  BERTScore / BERTScore_sc columns and bold marking the best value per
    #  (model, dataset) across the inner groups.
    # =====================================================================
    def _metric_table(self, *, second_col: str, inner_values: list[str],
                       inner_labels: list[str], metrics_for, caption: str,
                       label: str, table_env: str) -> list[str]:
        out = [
            f"\\begin{{table}}[{table_env}]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{lllccc}",
            r"\toprule",
            (r"\multirow{2}{*}{\textbf{Model}} & "
             f"\\multirow{{2}}{{*}}{{\\textbf{{{second_col}}}}} & "
             r"\multirow{2}{*}{\textbf{Dataset}} & \textbf{Lexical} & "
             r"\multicolumn{2}{c}{\textbf{Semantic \& Factual}} \\"),
            r"\cmidrule(lr){4-4}\cmidrule(lr){5-6}",
            (r" & & & \textbf{ROUGE-L} & \textbf{BERTScore} & "
             r"\textbf{BERTScore$_{\mathrm{sc}}$} \\"),
            r"\midrule",
        ]

        for mi, model in enumerate(MODELS):
            if mi > 0:
                out.append(r"\midrule")

            # Collect rounded values so we can bold the per-(dataset) maxima.
            # cells[inner][ds] = (rl, bs, bssc) rounded to 3 dp
            cells = {}
            for inner in inner_values:
                cells[inner] = {}
                for ds_token, sample, _ in DATASETS:
                    rl, bs, bssc = metrics_for(model, inner, ds_token, sample)
                    cells[inner][ds_token] = (round(rl, 3), round(bs, 3), round(bssc, 3))
            best = {}  # ds_token -> (max_rl, max_bs, max_bssc)
            for ds_token, _, _ in DATASETS:
                cols = list(zip(*(cells[i][ds_token] for i in inner_values)))
                best[ds_token] = tuple(max(c) for c in cols)

            for gi, (inner, inner_label) in enumerate(zip(inner_values, inner_labels)):
                if gi > 0:
                    out.append(r"\cmidrule(lr){2-6}")
                for di, (ds_token, _, ds_label) in enumerate(DATASETS):
                    vals = cells[inner][ds_token]
                    txt = []
                    for v, mx in zip(vals, best[ds_token]):
                        s = _fmt3(v)
                        txt.append(_bold(s) if v == mx else s)
                    body = f"{ds_label} & {txt[0]} & {txt[1]} & {txt[2]} \\\\"
                    if gi == 0 and di == 0:
                        out.append(f"\\multirow{{12}}{{*}}{{{MODEL_LABEL[model]}}} & "
                                   f"\\multirow{{4}}{{*}}{{{inner_label}}} & {body}")
                    elif di == 0:
                        out.append(f" & \\multirow{{4}}{{*}}{{{inner_label}}} & {body}")
                    else:
                        out.append(f" &  & {body}")

        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return out

    def per_prompt_tables(self) -> str:
        blocks: list[str] = []
        for pi, prompt in enumerate(PROMPTS):
            if pi > 0:
                blocks.append("")
            caption = CAPTIONS["per_prompt"].replace("<prompt>", prompt)

            def metrics_for(model, quant, ds_token, sample, _p=prompt):
                return (self.value(model, _p, quant, ds_token, sample, COL_ROUGEL),
                        self.value(model, _p, quant, ds_token, sample, COL_BERT),
                        self.value(model, _p, quant, ds_token, sample, COL_BERT_SCALED))

            blocks += self._metric_table(
                second_col="Precision", inner_values=QUANTS,
                inner_labels=[PRECISION_LABEL[q] for q in QUANTS],
                metrics_for=metrics_for, caption=caption,
                label=f"tab:eval-{prompt}", table_env="tbp")
        # NB: the on-disk template names are inverted vs. their content — the
        # per-prompt tables live in PerPrecisionTables.tex (kept as-is to match).
        return self._write("PerPrecisionTables.tex", blocks)

    def per_precision_tables(self) -> str:
        blocks: list[str] = []
        for qi, quant in enumerate(QUANTS):
            if qi > 0:
                blocks.append("")
            caption = CAPTIONS["per_precision"].replace("<precision>", PRECISION_LABEL[quant])

            def metrics_for(model, prompt, ds_token, sample, _q=quant):
                return (self.value(model, prompt, _q, ds_token, sample, COL_ROUGEL),
                        self.value(model, prompt, _q, ds_token, sample, COL_BERT),
                        self.value(model, prompt, _q, ds_token, sample, COL_BERT_SCALED))

            blocks += self._metric_table(
                second_col="Prompt", inner_values=PROMPTS, inner_labels=PROMPTS,
                metrics_for=metrics_for, caption=caption,
                label=f"tab:eval-{quant}", table_env="H")
        # NB: on-disk names are inverted vs. content (see per_prompt_tables).
        return self._write("PerPromptTables.tex", blocks)

    # =====================================================================
    #  Graphs: 2x4 groupplot per metric (columns=models, rows=datasets),
    #  one line per prompt across 16-bit/8-bit/4-bit.
    # =====================================================================
    def _graph_figure(self, *, col: str, caption: str, label: str) -> list[str]:
        mark = {"P1": "*", "P2": "square*", "P3": "triangle*"}
        style = {"P1": "pOne", "P2": "pTwo", "P3": "pThree"}
        out = [
            r"\begin{figure}[tbp]",
            r"\centering",
            r"\begin{tikzpicture}[baseline, every node/.style={font=\small}]",
            r"\draw[pOne,line width=0.9pt,mark=*,mark size=1.7pt] plot coordinates{(0,0)(0.5,0)}; \node[right] at (0.55,0){P1};",
            r"\draw[pTwo,line width=0.9pt,mark=square*,mark size=1.7pt] plot coordinates{(1.5,0)(2.0,0)}; \node[right] at (2.05,0){P2};",
            r"\draw[pThree,line width=0.9pt,mark=triangle*,mark size=1.9pt] plot coordinates{(3.0,0)(3.5,0)}; \node[right] at (3.55,0){P3};",
            r"\end{tikzpicture}\\[2pt]",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tikzpicture}",
            r"\begin{groupplot}[",
            r"  group style={group size=2 by 4, horizontal sep=0.95cm,",
            r"    vertical sep=0.85cm, xticklabels at=edge bottom},",
            r"  width=3.6cm, height=2.7cm, scale only axis,",
            r"  xmin=0.8, xmax=3.2, xtick={1,2,3}, xticklabels={16-bit,8-bit,4-bit},",
            r"  tick label style={font=\scriptsize}, title style={font=\small},",
            r"  ylabel style={font=\small, align=center},",
            r"  yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=3},",
            r"  every axis plot/.append style={line width=0.9pt, mark size=1.7pt},",
            r"  grid=both, grid style={gray!25, line width=0.3pt},",
            r"]",
        ]
        for ri, (ds_token, sample, ds_label) in enumerate(DATASETS):
            out.append(f"% Row {ri + 1}: {ds_label}")
            for ci, model in enumerate(MODELS):
                opts = []
                if ri == 0:
                    opts.append(f"title={{{MODEL_LABEL[model]}}}")
                if ci == 0:
                    opts.append(f"ylabel={{{ds_label}}}")
                out.append(f"\\nextgroupplot[{', '.join(opts)}]")
                for prompt in PROMPTS:
                    coords = " ".join(
                        f"({i + 1},{_fmt4(self.value(model, prompt, q, ds_token, sample, col))})"
                        for i, q in enumerate(QUANTS))
                    out.append(f"\\addplot[{style[prompt]},mark={mark[prompt]}] "
                               f"coordinates {{{coords}}};")
        out += [
            r"\end{groupplot}",
            r"\end{tikzpicture}%",
            r"}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{figure}",
        ]
        return out

    def graphs_per_metrics(self) -> str:
        rouge = self._graph_figure(
            col=COL_ROUGEL,
            caption=CAPTIONS["graph"].replace("<metric>", "ROUGE-L"),
            label="fig:rq-rougeL")
        bert = self._graph_figure(
            col=COL_BERT,
            caption=CAPTIONS["graph"].replace("<metric>", "BERTScore"),
            label="fig:rq-bertscore")
        return self._write("GraphsPerMetrics.tex", rouge + [""] + bert)

    # =====================================================================
    #  Prompt sensitivity: spread (max-min)*100 across prompts, per precision.
    # =====================================================================
    def prompt_sensitivity_per_precision(self) -> str:
        metrics = [("ROUGE-L", COL_ROUGEL), ("BERTScore", COL_BERT)]
        out = [
            r"\begin{table}[tbp]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
            f"\\caption{{{CAPTIONS['prompt_sensitivity']}}}",
            r"\label{tab:rq-spread}",
            r"\begin{tabular}{lllccc}",
            r"\toprule",
            (r"\textbf{Model} & \textbf{Dataset} & \textbf{Metric} & \textbf{16-bit} & "
             r"\textbf{8-bit} & \textbf{4-bit} \\"),
            r"\midrule",
        ]
        for mi, model in enumerate(MODELS):
            if mi > 0:
                out.append(r"\midrule")
            for di, (ds_token, sample, ds_label) in enumerate(DATASETS):
                if di > 0:
                    out.append(r"\cmidrule(lr){2-6}")
                for mj, (metric_label, col) in enumerate(metrics):
                    spreads = []
                    for q in QUANTS:
                        vals = [self.value(model, p, q, ds_token, sample, col) for p in PROMPTS]
                        spreads.append(round((max(vals) - min(vals)) * 100, 2))
                    mx = max(spreads)
                    txt = [_bold(f"{s:.2f}") if s == mx else f"{s:.2f}" for s in spreads]
                    cells = f"{metric_label} & {txt[0]} & {txt[1]} & {txt[2]} \\\\"
                    if di == 0 and mj == 0:
                        out.append(f"\\multirow{{8}}{{*}}{{{MODEL_LABEL[model]}}} & "
                                   f"\\multirow{{2}}{{*}}{{{ds_label}}} & {cells}")
                    elif mj == 0:
                        out.append(f" & \\multirow{{2}}{{*}}{{{ds_label}}} & {cells}")
                    else:
                        out.append(f" &  & {cells}")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return self._write("PromptSensitivityPerPrecision.tex", out)

    # =====================================================================
    #  Quantization gap: Delta = (16-bit - quantized)*100 per prompt.
    # =====================================================================
    def quantization_gap_per_prompt(self) -> str:
        metrics = [("ROUGE-L", COL_ROUGEL), ("BERTScore", COL_BERT)]
        out = [
            r"\begin{table*}[!tp]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
            f"\\caption{{{CAPTIONS['quantization_gap']}}}",
            r"\label{tab:rq-gap}",
            r"\begin{tabular}{lllccccccc}",
            r"\toprule",
            (r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Dataset}} & "
             r"\multirow{2}{*}{\textbf{Metric}} & \multicolumn{3}{c}{$\Delta$ "
             r"\textbf{8-bit} (pts)} & \multicolumn{3}{c}{$\Delta$ \textbf{4-bit} (pts)} \\"),
            r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
            (r" & & & \textbf{P1} & \textbf{P2} & \textbf{P3} & \textbf{P1} & "
             r"\textbf{P2} & \textbf{P3} \\"),
            r"\midrule",
        ]
        for mi, model in enumerate(MODELS):
            if mi > 0:
                out.append(r"\midrule")
            for di, (ds_token, sample, ds_label) in enumerate(DATASETS):
                if di > 0:
                    out.append(r"\cmidrule(lr){2-9}")
                for mj, (metric_label, col) in enumerate(metrics):
                    group_txt = []
                    for q in ("8bit", "4bit"):
                        gaps = []
                        for p in PROMPTS:
                            fp = self.value(model, p, "16bit", ds_token, sample, col)
                            qv = self.value(model, p, q, ds_token, sample, col)
                            gaps.append((fp - qv) * 100)
                        # Smallest gap magnitude wins, judged on the unrounded value
                        # (so cells that both print as e.g. +0.25 still order correctly).
                        best_i = min(range(len(gaps)), key=lambda i: abs(gaps[i]))
                        for i, g in enumerate(gaps):
                            s = _fmt_signed2(round(g, 2))
                            group_txt.append(_bold(s) if i == best_i else s)
                    cells = f"{metric_label} & " + " & ".join(group_txt) + r" \\"
                    if di == 0 and mj == 0:
                        out.append(f"\\multirow{{8}}{{*}}{{{MODEL_LABEL[model]}}} & "
                                   f"\\multirow{{2}}{{*}}{{{ds_label}}} & {cells}")
                    elif mj == 0:
                        out.append(f" & \\multirow{{2}}{{*}}{{{ds_label}}} & {cells}")
                    else:
                        out.append(f" &  & {cells}")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
        return self._write("QuantizationGapPerPrompt.tex", out, trailing_newline=True)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(here, "..", "results", "evaluation.csv")
    default_out = os.path.join(here, "..", "results", "tex")

    ap = argparse.ArgumentParser(description="Generate LaTeX tables/figures from evaluation.csv")
    ap.add_argument("--csv", default=default_csv, help="path to evaluation.csv")
    ap.add_argument("--out-dir", default=default_out, help="output dir for .tex files")
    args = ap.parse_args()

    report = TexReport(csv_path=os.path.abspath(args.csv), out_dir=os.path.abspath(args.out_dir))
    for path in report.generate_all():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

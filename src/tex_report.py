"""Generate the LaTeX tables and figures under ``results/tex/`` from an evaluation CSV.

Two experiments are rendered by the *same* builders, selected with ``--grid``:

  * ``summary`` (default) -- the main experiment, from ``results/evaluation.csv``.
  * ``qa`` -- the QA_Evaluation experiment, from ``results/qa_evaluation.csv``,
    which additionally carries the LERC columns.

Both write into ``results/tex/``; the QA files are prefixed ``QA_`` and their
``\\label`` keys carry a ``-qa`` suffix, so the two sets never collide and can be
\\input into the same document.

A `Grid` holds everything that differs between the two (axes, printed labels, how
a CSV row is addressed, the full-precision baseline, whether LERC exists), so the
table code exists exactly once.

Run it manually:

    python src/tex_report.py                       # summary -> results/tex/*.tex
    python src/tex_report.py --grid qa             # QA      -> results/tex/QA_*.tex
    python src/tex_report.py --csv path/to.csv --out-dir somewhere

It writes three files per grid:

  * ``PerPrecisionCondensed.tex`` – the full precision x prompt grid, one table
  * ``GraphsPerMetrics.tex``      – pgfplots line charts per metric
  * ``RQAnalysisCombined.tex``    – prompt sensitivity + quantization gap
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from naming import summary_filename


# --- Experiment grid (order here is the order rows/columns appear in the tex) ---
# NB: MODELS holds the CSV/filename tokens (e.g. "Llama_P1_16bit_..."), while
# MODEL_LABEL maps each to the name printed in the tables/figures. The full
# official names (Llama-3.2-3B-Instruct, Phi-3-mini-4k-instruct) and their HF
# ids belong in the paper's setup section; these are the compact labels.
MODELS = ["Llama", "Phi", "Qwen2"]
MODEL_LABEL = {"Llama": "Llama-3.2-3B", "Phi": "Phi-3-mini", "Qwen2": "Qwen2-1.5B"}
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
# The CSV also carries bertscore_f1_scaled(_std), the baseline-rescaled F1. It is
# deliberately NOT printed: it is an exact affine transform of the raw F1
# (raw = scaled*(1-b) + b with a single constant b = 0.831226 across every row), so
# it orders all configurations identically and would duplicate the raw column at the
# cost of three table columns.
COL_ROUGEL = "rougeL"
COL_BERT = "bertscore_f1_raw"
COL_BERT_STD = "bertscore_f1_raw_std"
COL_LERC = "lerc_mean"
COL_LERC_STD = "lerc_std"


# --- Captions (edit here; one place for all tables/figures) ----------------
# The <...> placeholders are filled by TexReport._caption() before writing, which
# also appends the grid's caption_note (e.g. the QA grid's 500-article disclaimer):
#   <metric>   -> ROUGE-L / BERTScore              (graph only)
#   <dataset>  -> the grid's single dataset label   (graph_stacked only)
#   <datasets> -> all of the grid's dataset labels, comma-joined
CAPTIONS = {
    "graph": (
        r"<metric> for each quantization precision and prompt. Columns are models, "
        r"rows are datasets. The vertical spread between lines is prompt sensitivity; "
        r"each line's slope from 16-bit to 4-bit is that prompt's degradation under "
        r"quantization."
    ),
    "graph_stacked": (
        r"Evaluation metrics on <dataset> for each quantization precision and prompt. "
        r"Columns are models, rows are metrics. The vertical spread between lines is "
        r"prompt sensitivity; each line's slope from 16-bit to 4-bit is that prompt's "
        r"degradation under quantization."
    ),
    "per_precision_condensed": (
        r"Evaluation metrics on <datasets> for every precision$\times$prompt "
        r"configuration. Columns are grouped by precision, prompt is the inner row "
        r"grouping. BERTScore and LERC are mean$\pm$standard deviation across "
        r"articles; ROUGE-L is a single aggregate value with no per-article deviation "
        r"recorded. Within each row group, the best mean per metric is in "
        r"\textbf{bold}."
    ),
    "rq_combined": (
        r"Prompt sensitivity and the quantization gap, which share the same rows. "
        r"\emph{Prompt sensitivity}: the spread (max$-$min across the three prompts, "
        r"in points $\times 100$) at each precision; larger values mean the score "
        r"depends more on prompt wording, and the most sensitive precision per row is "
        r"in \textbf{bold}. \emph{$\Delta$ 8-bit / 4-bit}: the gap $\Delta = "
        r"\text{16-bit} - \text{quantized}$ (points $\times 100$), where positive "
        r"means quality drops; within each precision the prompt with the smallest gap "
        r"magnitude is in \textbf{bold}, isolating whether more structured prompting "
        r"(P1$\to$P3) narrows the gap."
    ),
}


# --- Experiment grids ------------------------------------------------------
# The table/figure builders are identical for every experiment; only the axes and
# the way a CSV row is addressed differ. A Grid bundles exactly that, so the QA
# experiment reuses the generators instead of duplicating them.


@dataclass(frozen=True)
class Grid:
    """The axes of one experiment plus how to locate its rows in the CSV."""

    models: list[str]
    model_label: dict[str, str]
    prompts: list[str]
    quants: list[str]
    quant_label: dict[str, str]
    datasets: list[tuple[str, str, str]]
    key_column: str                       # CSV column holding the row key
    row_key: Callable[[str, str, str, str, str], str]
    baseline_quant: str                   # full-precision reference for the gap tables
    label_suffix: str = ""                # keeps \label keys unique across grids
    out_subdir: str = "tex"
    file_prefix: str = ""                  # prepended to every written filename
    has_lerc: bool = False                 # whether lerc_mean/lerc_std columns exist
    caption_note: str = ""                 # appended (with its own leading space) to every caption


SUMMARY_GRID = Grid(
    models=MODELS,
    model_label=MODEL_LABEL,
    prompts=PROMPTS,
    quants=QUANTS,
    quant_label=PRECISION_LABEL,
    datasets=DATASETS,
    key_column="file",
    row_key=lambda model, prompt, quant, ds_token, sample: summary_filename(
        model, prompt, quant, ds_token, sample),
    baseline_quant="16bit",
)

# QA experiment (QA_Evaluation/): one dataset, and the row key is the config token
# rather than a summary filename. Quant "None" means no bitsandbytes quantization,
# i.e. the model is loaded in fp16 -- identical to "16bit" in src/model.py -- so it
# is labelled 16-bit and serves as the baseline for the quantization gap.
QA_GRID = Grid(
    models=["llama_3.2_3b_instruct", "phi_3_mini_4k_instruct", "qwen2_1.5b_instruct"],
    model_label={"llama_3.2_3b_instruct": "Llama-3.2-3B",
                 "phi_3_mini_4k_instruct": "Phi-3-mini",
                 "qwen2_1.5b_instruct": "Qwen2-1.5B"},
    prompts=PROMPTS,
    quants=["None", "8bit", "4bit"],
    quant_label={"None": "16-bit", "8bit": "8-bit", "4bit": "4-bit"},
    datasets=[("newsqasum", "500", "NewsQASum")],
    key_column="config",
    row_key=lambda model, prompt, quant, ds_token, sample: f"{model}_{quant}_{prompt}",
    baseline_quant="None",
    label_suffix="-qa",
    file_prefix="QA_",          # the prefix, not a separate dir, keeps the two apart
    has_lerc=True,
    caption_note=r" Computed on the first 500 of 10388 NewsQASum articles.",
)

GRIDS = {"summary": SUMMARY_GRID, "qa": QA_GRID}


def _fmt3(v: float) -> str:
    return f"{v:.3f}"


def _fmt4(v: float) -> str:
    return f"{v:.4f}"


def _fmt_signed2(v: float) -> str:
    return f"{v:+.2f}"


def _bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def _fmt_cell(mean: float, std: float | None) -> str:
    """'0.235' alone, or '0.235$\\pm$0.012' with the deviation set smaller.

    Deliberately one line. Stacking the std under the mean is narrower, but the
    table is set with \\resizebox, which preserves the aspect ratio -- so trading
    width for height buys nothing and makes the float ~1.5x taller (measured),
    which is what pushes it off the page.
    """
    s = _fmt3(mean)
    if std is None:
        return s
    return f"{s}{{\\scriptsize$\\pm${_fmt3(std)}}}"


@dataclass(frozen=True)
class MetricSpec:
    """One metric column: its header text, mean column, and optional std column."""

    header: str      # already-wrapped LaTeX, e.g. r"\textbf{ROUGE-L}"
    col: str
    std_col: str | None = None


@dataclass
class TexReport:
    """Reads the evaluation CSV and emits the LaTeX files under ``out_dir``."""

    csv_path: str
    out_dir: str
    grid: Grid = SUMMARY_GRID
    rows: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._load()

    # --- data access -----------------------------------------------------
    def _load(self) -> None:
        with open(self.csv_path, newline="") as f:
            for row in csv.DictReader(f):
                self.rows[row[self.grid.key_column]] = row

    def value(self, model: str, prompt: str, quant: str, ds_token: str,
              sample: str, col: str) -> float:
        """One metric cell for a (model, prompt, quant, dataset) run."""
        key = self.grid.row_key(model, prompt, quant, ds_token, sample)
        if key not in self.rows:
            raise KeyError(f"missing row in {self.csv_path}: {key}")
        raw = self.rows[key][col]
        if raw is None or raw == "":
            raise ValueError(f"empty {col} for {key}")
        return float(raw)

    def label(self, name: str) -> str:
        """A \\label key, suffixed so two grids can coexist in one document."""
        return f"{name}{self.grid.label_suffix}"

    def _dataset_list(self) -> str:
        """The grid's dataset labels as prose: 'A', 'A and B', 'A, B and C'."""
        labels = [ds_label for _, _, ds_label in self.grid.datasets]
        if len(labels) == 1:
            return labels[0]
        return ", ".join(labels[:-1]) + " and " + labels[-1]

    def _caption(self, key: str, **replacements: str) -> str:
        """CAPTIONS[key] with its <placeholder>s filled, plus the grid's caption_note.

        Replacements are given by placeholder name without angle brackets, e.g.
        ``self._caption("graph", metric="ROUGE-L")`` fills ``<metric>``. ``<datasets>``
        is filled from the grid itself, so callers never pass it.
        """
        text = CAPTIONS[key].replace("<datasets>", self._dataset_list())
        for placeholder, value in replacements.items():
            text = text.replace(f"<{placeholder}>", value)
        return text + self.grid.caption_note

    def _metric_specs(self) -> list[MetricSpec]:
        """ROUGE-L / BERTScore, plus LERC when the grid has it.

        ROUGE-L carries no std -- the CSV has no rougeL_std column, so it prints as
        a bare mean while the others print as mean over $\\pm$std.
        """
        specs = [
            MetricSpec(r"\textbf{ROUGE-L}", COL_ROUGEL),
            MetricSpec(r"\textbf{BERTScore}", COL_BERT, COL_BERT_STD),
        ]
        if self.grid.has_lerc:
            specs.append(MetricSpec(r"\textbf{LERC}", COL_LERC, COL_LERC_STD))
        return specs

    def _rq_metrics(self) -> list[tuple[str, str]]:
        """(label, column) rows of the RQ analysis table; LERC only where it exists."""
        metrics = [("ROUGE-L", COL_ROUGEL), ("BERTScore", COL_BERT)]
        if self.grid.has_lerc:
            metrics.append(("LERC", COL_LERC))
        return metrics

    def _fetch_specs(self, model: str, prompt: str, quant: str, ds_token: str,
                     sample: str, specs: list[MetricSpec]) -> list[tuple[float, float | None]]:
        """One (mean, std) pair per spec for a single (model, prompt, quant, dataset) cell."""
        return [
            (self.value(model, prompt, quant, ds_token, sample, spec.col),
             self.value(model, prompt, quant, ds_token, sample, spec.std_col) if spec.std_col else None)
            for spec in specs
        ]

    def _write(self, name: str, lines: list[str], trailing_newline: bool = False) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{self.grid.file_prefix}{name}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + ("\n" if trailing_newline else ""))
        return path

    # --- public entry points ----------------------------------------------
    def generate_grid_files(self) -> list[str]:
        """The files that describe this grid on its own."""
        return [
            self.per_precision_condensed(),
            self.graphs_per_metrics(),
        ]

    def generate_all(self, rq_extra: Sequence[TexReport] = ()) -> list[str]:
        """This grid's own files plus the RQ table, optionally merged with others."""
        return self.generate_grid_files() + [self.rq_analysis_combined(rq_extra)]

    # =====================================================================
    #  Condensed metrics table: the whole precision x prompt grid as one wide
    #  table*. Rows: Model -> prompt (-> dataset, when the grid has more than
    #  one); columns: precision x metric. Bold = best mean per (row group,
    #  metric) across the full grid. Cells are the mean stacked over $\pm$std
    #  where a std column exists (BERTScore, and LERC when the grid has it) --
    #  ROUGE-L stays a plain mean, there is no rougeL_std in the CSV.
    # =====================================================================
    def _condensed_metric_table(self, *, inner_header: str, inner_values: list[str],
                                inner_labels: list[str], group_values: list[str],
                                group_labels: list[str], specs: list[MetricSpec],
                                fetch, caption: str, label: str) -> list[str]:
        # A single-dataset grid repeats the same dataset name on every row, which
        # costs a column and buys nothing -- the name is in the caption instead.
        show_ds = len(self.grid.datasets) > 1
        n_label_cols = 3 if show_ds else 2
        n, n_grp = len(specs), len(group_values)
        first_data_col = n_label_cols + 1
        last_col = n_label_cols + n * n_grp      # index of the rightmost data column
        model_span = len(inner_values) * len(self.grid.datasets)
        ds_span = len(self.grid.datasets)
        colspec = "l" * n_label_cols + "c" * (n * n_grp)

        head_groups = "".join(
            f" & \\multicolumn{{{n}}}{{c}}{{\\textbf{{{gl}}}}}" for gl in group_labels)
        group_cmids = "".join(
            f"\\cmidrule(lr){{{first_data_col + n * i}-{n_label_cols + n * (i + 1)}}}"
            for i in range(n_grp))
        sub = " & ".join(spec.header for spec in specs)
        sub_row = " & " * n_label_cols + " & ".join([sub] * n_grp) + r" \\"
        ds_header = r" & \multirow{2}{*}{\textbf{Dataset}}" if show_ds else ""

        out = [
            r"\begin{table*}[b]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            # \resizebox stretches the (naturally narrow) tabular to the full
            # two-column text width, scaling the text up with it.
            # Shrink to the text width only if the table is wider than it; a table
            # that already fits keeps its true \footnotesize instead of being blown
            # up. \width is the box's natural width, which \resizebox exposes here.
            r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
            f"\\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            (r"\multirow{2}{*}{\textbf{Model}} & "
             f"\\multirow{{2}}{{*}}{{\\textbf{{{inner_header}}}}}" + ds_header + head_groups + r" \\"),
            group_cmids,
            sub_row,
            r"\midrule",
        ]

        for mi, model in enumerate(self.grid.models):
            if mi > 0:
                out.append(r"\midrule")

            # cells[group][inner][ds] = list of (mean, std) per spec, mean rounded to 3dp
            cells = {}
            for g in group_values:
                cells[g] = {}
                for inner in inner_values:
                    cells[g][inner] = {}
                    for ds_token, sample, _ in self.grid.datasets:
                        cells[g][inner][ds_token] = [
                            (round(mean, 3), std) for mean, std in fetch(model, g, inner, ds_token, sample)
                        ]
            # best per (dataset, metric) across the whole group x inner grid
            best = {}
            for ds_token, _, _ in self.grid.datasets:
                means = [[m for m, _ in cells[g][i][ds_token]] for g in group_values for i in inner_values]
                best[ds_token] = tuple(max(col) for col in zip(*means))

            for gi, (inner, inner_label) in enumerate(zip(inner_values, inner_labels)):
                if gi > 0:
                    out.append(f"\\cmidrule(lr){{2-{last_col}}}")
                for di, (ds_token, _, ds_label) in enumerate(self.grid.datasets):
                    parts = []
                    for g in group_values:
                        for (mean, std), mx in zip(cells[g][inner][ds_token], best[ds_token]):
                            s = _fmt_cell(mean, std)
                            parts.append(_bold(s) if mean == mx else s)
                    ds_cell = f"{ds_label} & " if show_ds else ""
                    numbers = " & ".join(parts)
                    if gi == 0 and di == 0:
                        out.append(f"\\multirow{{{model_span}}}{{*}}{{{self.grid.model_label[model]}}} & "
                                   f"\\multirow{{{ds_span}}}{{*}}{{{inner_label}}} & "
                                   f"{ds_cell}{numbers} \\\\")
                    elif di == 0:
                        out.append(f" & \\multirow{{{ds_span}}}{{*}}{{{inner_label}}} & "
                                   f"{ds_cell}{numbers} \\\\")
                    else:
                        out.append(f" &  & {ds_cell}{numbers} \\\\")

        out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
        return out

    def per_precision_condensed(self) -> str:
        """One wide table*, precision as the column grouping, prompt as inner rows."""
        specs = self._metric_specs()

        def fetch(model, quant, prompt, ds_token, sample, _specs=specs):
            return self._fetch_specs(model, prompt, quant, ds_token, sample, _specs)

        out = self._condensed_metric_table(
            inner_header="Prompt", inner_values=self.grid.prompts, inner_labels=self.grid.prompts,
            group_values=self.grid.quants, group_labels=[self.grid.quant_label[q] for q in self.grid.quants],
            specs=specs, fetch=fetch, caption=self._caption("per_precision_condensed"),
            label=self.label("tab:eval-per-precision-condensed"))
        return self._write("PerPrecisionCondensed.tex", out)

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
            r"\begin{tikzpicture}[baseline, every node/.style={font=\large}]",
            r"\draw[pOne,line width=0.9pt,mark=*,mark size=1.7pt] plot coordinates{(0,0)(0.5,0)}; \node[right] at (0.55,0){P1};",
            r"\draw[pTwo,line width=0.9pt,mark=square*,mark size=1.7pt] plot coordinates{(1.5,0)(2.0,0)}; \node[right] at (2.05,0){P2};",
            r"\draw[pThree,line width=0.9pt,mark=triangle*,mark size=1.9pt] plot coordinates{(3.0,0)(3.5,0)}; \node[right] at (3.55,0){P3};",
            r"\end{tikzpicture}\\[2pt]",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tikzpicture}",
            r"\begin{groupplot}[",
            f"  group style={{group size={len(self.grid.models)} by 4, horizontal sep=0.95cm,",
            r"    vertical sep=0.85cm, xticklabels at=edge bottom},",
            r"  width=3.6cm, height=2.7cm, scale only axis,",
            r"  xmin=0.8, xmax=3.2, xtick={1,2,3}, xticklabels={16-bit,8-bit,4-bit},",
            r"  tick label style={font=\normalsize}, title style={font=\large},",
            r"  ylabel style={font=\large, align=center},",
            r"  yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=3},",
            r"  every axis plot/.append style={line width=0.9pt, mark size=1.7pt},",
            r"  grid=both, grid style={gray!25, line width=0.3pt},",
            r"]",
        ]
        for ri, (ds_token, sample, ds_label) in enumerate(self.grid.datasets):
            out.append(f"% Row {ri + 1}: {ds_label}")
            for ci, model in enumerate(self.grid.models):
                opts = []
                if ri == 0:
                    opts.append(f"title={{{self.grid.model_label[model]}}}")
                if ci == 0:
                    opts.append(f"ylabel={{{ds_label}}}")
                out.append(f"\\nextgroupplot[{', '.join(opts)}]")
                for prompt in self.grid.prompts:
                    coords = " ".join(
                        f"({i + 1},{_fmt4(self.value(model, prompt, q, ds_token, sample, col))})"
                        for i, q in enumerate(self.grid.quants))
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
        # A single dataset leaves nothing to fill the dataset-rows layout below
        # with -- one row per dataset would be one row, full stop. Metrics take
        # the row slot instead, stacked into one figure (see _metrics_graph_figure).
        if len(self.grid.datasets) == 1:
            _, _, ds_label = self.grid.datasets[0]
            fig = self._metrics_graph_figure(
                metrics=self._rq_metrics(),
                caption=self._caption("graph_stacked", dataset=ds_label),
                label=self.label("fig:rq-metrics"))
            return self._write("GraphsPerMetrics.tex", fig)

        rouge = self._graph_figure(
            col=COL_ROUGEL,
            caption=self._caption("graph", metric="ROUGE-L"),
            label=self.label("fig:rq-rougeL"))
        bert = self._graph_figure(
            col=COL_BERT,
            caption=self._caption("graph", metric="BERTScore"),
            label=self.label("fig:rq-bertscore"))
        return self._write("GraphsPerMetrics.tex", rouge + [""] + bert)

    # =====================================================================
    #  Single-dataset variant of the graph above: there is no dataset axis to
    #  put in the rows, so the metrics themselves take that slot instead, and
    #  the (one) dataset name is printed once above the whole grid rather than
    #  repeated per row. Otherwise identical to _graph_figure: same legend,
    #  same per-column model title, x-axis is still precision.
    # =====================================================================
    def _metrics_graph_figure(self, *, metrics: list[tuple[str, str]],
                              caption: str, label: str) -> list[str]:
        assert len(self.grid.datasets) == 1, "use _graph_figure for multi-dataset grids"
        ds_token, sample, ds_label = self.grid.datasets[0]
        mark = {"P1": "*", "P2": "square*", "P3": "triangle*"}
        style = {"P1": "pOne", "P2": "pTwo", "P3": "pThree"}
        out = [
            r"\begin{figure}[tbp]",
            r"\centering",
            # Dataset name first, as a plain subtitle above the prompt legend.
            f"{{\\large {ds_label}}}\\\\[3pt]",
            r"\begin{tikzpicture}[baseline, every node/.style={font=\large}]",
            r"\draw[pOne,line width=0.9pt,mark=*,mark size=1.7pt] plot coordinates{(0,0)(0.5,0)}; \node[right] at (0.55,0){P1};",
            r"\draw[pTwo,line width=0.9pt,mark=square*,mark size=1.7pt] plot coordinates{(1.5,0)(2.0,0)}; \node[right] at (2.05,0){P2};",
            r"\draw[pThree,line width=0.9pt,mark=triangle*,mark size=1.9pt] plot coordinates{(3.0,0)(3.5,0)}; \node[right] at (3.55,0){P3};",
            r"\end{tikzpicture}\\[2pt]",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tikzpicture}",
            r"\begin{groupplot}[",
            f"  group style={{group size={len(self.grid.models)} by {len(metrics)}, horizontal sep=0.95cm,",
            r"    vertical sep=0.85cm, xticklabels at=edge bottom},",
            r"  width=3.6cm, height=2.7cm, scale only axis,",
            r"  xmin=0.8, xmax=3.2, xtick={1,2,3}, xticklabels={16-bit,8-bit,4-bit},",
            r"  tick label style={font=\normalsize}, title style={font=\large},",
            r"  ylabel style={font=\large, align=center},",
            r"  yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=3},",
            r"  every axis plot/.append style={line width=0.9pt, mark size=1.7pt},",
            r"  grid=both, grid style={gray!25, line width=0.3pt},",
            r"]",
        ]
        for ri, (metric_label, col) in enumerate(metrics):
            out.append(f"% Row {ri + 1}: {metric_label}")
            for ci, model in enumerate(self.grid.models):
                opts = []
                if ri == 0:
                    opts.append(f"title={{{self.grid.model_label[model]}}}")
                if ci == 0:
                    opts.append(f"ylabel={{{metric_label}}}")
                out.append(f"\\nextgroupplot[{', '.join(opts)}]")
                for prompt in self.grid.prompts:
                    coords = " ".join(
                        f"({i + 1},{_fmt4(self.value(model, prompt, q, ds_token, sample, col))})"
                        for i, q in enumerate(self.grid.quants))
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

    # =====================================================================
    #  Combined RQ table: prompt sensitivity (spread across prompts) AND the
    #  quantization gap share the same (model, dataset, metric) row skeleton, so
    #  they sit side by side as three column groups: sensitivity | D 8-bit |
    #  D 4-bit. Each block keeps its own bold rule (see caption).
    #
    #  It also MERGES grids: pass other TexReports as `extra` and their datasets
    #  are appended under the same model, so the main experiment and the QA
    #  experiment land in one table. That works because the two grids agree on
    #  everything the columns depend on -- same prompts, and precision tokens that
    #  differ ("16bit" vs "None") but carry the same printed label. Rows are keyed
    #  by the PRINTED model label, which is what lets the differing CSV model
    #  tokens (Llama vs llama_3.2_3b_instruct) line up.
    # =====================================================================
    def _rq_blocks_for(self, model_label: str) -> list[tuple[str, list[str]]]:
        """[(dataset label, [one '&'-joined row per metric]), ...] for one model."""
        model = next((m for m in self.grid.models
                      if self.grid.model_label[m] == model_label), None)
        if model is None:          # this grid does not cover that model
            return []
        blocks = []
        for ds_token, sample, ds_label in self.grid.datasets:
            rows = []
            for metric_label, col in self._rq_metrics():
                # Left block: prompt-sensitivity spread across prompts, per precision.
                sens = []
                for q in self.grid.quants:
                    vals = [self.value(model, p, q, ds_token, sample, col) for p in self.grid.prompts]
                    sens.append(round((max(vals) - min(vals)) * 100, 2))
                smax = max(sens)
                sens_txt = [_bold(f"{s:.2f}") if s == smax else f"{s:.2f}" for s in sens]
                # Right blocks: quantization gap Delta = full precision - quantized.
                gap_txt = []
                for q in ("8bit", "4bit"):
                    gaps = [(self.value(model, p, self.grid.baseline_quant, ds_token, sample, col)
                             - self.value(model, p, q, ds_token, sample, col)) * 100
                            for p in self.grid.prompts]
                    best_i = min(range(len(gaps)), key=lambda i: abs(gaps[i]))
                    for i, g in enumerate(gaps):
                        s = _fmt_signed2(round(g, 2))
                        gap_txt.append(_bold(s) if i == best_i else s)
                rows.append(f"{metric_label} & " + " & ".join(sens_txt + gap_txt))
            blocks.append((ds_label, rows))
        return blocks

    def rq_analysis_combined(self, extra: "Sequence[TexReport]" = ()) -> str:
        sources = [self, *extra]
        caption = self._caption("rq_combined")
        for src in extra:
            if src.grid.has_lerc:
                caption += (r" LERC is defined only for the QA experiment and therefore "
                            r"appears in the NewsQASum rows only.")
            if src.grid.caption_note:
                caption += src.grid.caption_note
        out = [
            r"\begin{table*}[b]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
            f"\\caption{{{caption}}}",
            f"\\label{{{self.label('tab:rq-combined')}}}",
            # booktabs adds vertical glue around \toprule/\midrule/\cmidrule that a
            # `|' rule can't cross, which breaks the vertical separators at every
            # horizontal rule. Zero it (scoped to this float) so the verticals run
            # full height.
            r"\setlength{\aboverulesep}{0pt}\setlength{\belowrulesep}{0pt}",
            # Shrink to the text width only if the table is wider than it; a table
            # that already fits keeps its true \footnotesize instead of being blown
            # up. \width is the box's natural width, which \resizebox exposes here.
            r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
            r"\begin{tabular}{lll|ccc|cccccc}",
            r"\toprule",
            (r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Dataset}} & "
             r"\multirow{2}{*}{\textbf{Metric}} & "
             r"\multicolumn{3}{c|}{\textbf{Prompt sensitivity}} & "
             r"\multicolumn{3}{c}{$\Delta$ \textbf{8-bit} (pts)} & "
             r"\multicolumn{3}{c}{$\Delta$ \textbf{4-bit} (pts)} \\"),
            r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
            (r" & & & \textbf{16-bit} & \textbf{8-bit} & \textbf{4-bit} & "
             r"\textbf{P1} & \textbf{P2} & \textbf{P3} & "
             r"\textbf{P1} & \textbf{P2} & \textbf{P3} \\"),
            r"\midrule",
        ]
        for mi, model in enumerate(self.grid.models):
            model_label = self.grid.model_label[model]
            if mi > 0:
                out.append(r"\midrule")
            # Every source's datasets for this model, in source order.
            blocks = [b for src in sources for b in src._rq_blocks_for(model_label)]
            model_span = sum(len(rows) for _, rows in blocks)
            first = True
            for ds_label, rows in blocks:
                if not first:
                    out.append(r"\cmidrule(lr){2-12}")
                for ri, row in enumerate(rows):
                    if first and ri == 0:
                        out.append(f"\\multirow{{{model_span}}}{{*}}{{{model_label}}} & "
                                   f"\\multirow{{{len(rows)}}}{{*}}{{{ds_label}}} & {row} \\\\")
                    elif ri == 0:
                        out.append(f" & \\multirow{{{len(rows)}}}{{*}}{{{ds_label}}} & {row} \\\\")
                    else:
                        out.append(f" &  & {row} \\\\")
                first = False
        out += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
        return self._write("RQAnalysisCombined.tex", out, trailing_newline=True)


# Default CSV per grid; the output dir follows the grid's own out_subdir.
DEFAULT_CSV = {"summary": "evaluation.csv", "qa": "qa_evaluation.csv"}


PRIMARY_GRID = "summary"   # owns the merged RQ table when several grids are rendered


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    results = os.path.join(here, "..", "results")

    ap = argparse.ArgumentParser(description="Generate LaTeX tables/figures from the evaluation CSVs")
    ap.add_argument("--grid", choices=[*sorted(GRIDS), "all"], default="all",
                    help="which experiment to render; 'all' (default) also merges the "
                         "RQ analysis of every grid into one table")
    ap.add_argument("--csv", default=None,
                    help="path to the CSV; only valid with a single --grid")
    ap.add_argument("--out-dir", default=None,
                    help="output dir for .tex files (default: results/tex)")
    args = ap.parse_args()

    names = sorted(GRIDS) if args.grid == "all" else [args.grid]
    if args.csv and len(names) > 1:
        ap.error("--csv needs a single --grid")

    reports: dict[str, TexReport] = {}
    for name in names:
        csv_path = args.csv or os.path.join(results, DEFAULT_CSV[name])
        if not os.path.exists(csv_path):
            print(f"skipping {name!r}: {csv_path} not found")
            continue
        out_dir = args.out_dir or os.path.join(results, GRIDS[name].out_subdir)
        reports[name] = TexReport(csv_path=os.path.abspath(csv_path),
                                  out_dir=os.path.abspath(out_dir), grid=GRIDS[name])
    if not reports:
        ap.error("no CSV found for any requested grid")

    # One RQ table for everything: the primary grid writes it and the others feed
    # their rows in, so a single float covers all datasets instead of one per grid.
    primary = reports.get(PRIMARY_GRID) or next(iter(reports.values()))
    extra = [r for r in reports.values() if r is not primary]

    written = primary.generate_all(rq_extra=extra)
    for report in extra:
        written += report.generate_grid_files()
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

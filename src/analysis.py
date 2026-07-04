"""Post-hoc analysis of generated summaries.

Standalone helper (not part of the summarize/evaluate pipeline). Reads the
`.jsonl` files in the summaries dir and writes Markdown reports to results/.

Run from the repo root:
    python src/analysis.py                       # all summaries/*.jsonl
    python src/analysis.py --output_dir summaries --results_dir results

Currently provides three analyses:
  1. token_limit    — how many summaries hit their prompt's max_new_tokens cap
  2. summary_marker — how many summaries lack a "Summary:"-style marker
  3. sanity_check   — a Markdown spot-check of the same N summaries across all files
"""

import argparse
import glob
import json
import os
import random
import re

from prompts import PROMPT_CONFIGS
from dataset import DATASET_CONFIGS
from job_time import _parse_name  # sum_{model}_{quant}_{prompt}_{dataset}_{jobid}.out -> config

# Expected grid for the job-status report. Prompts/datasets (incl. the xu_* ones)
# come from the registries; models/quants are small fixed lists so the report
# needs no torch. Keep _STATUS_MODELS in sync with MODEL_CONFIGS in model.py.
_STATUS_MODELS = ["Llama", "Phi"]
_STATUS_QUANTS = ["16bit", "8bit", "4bit"]

# A summary "has a marker" if this pattern matches — the same 'Summary:' marker the
# evaluator strips (tolerant of markdown bold and a lead-in on the line).
SUMMARY_MARKERS = re.compile(r"(?:^|\n)[^\n]*?\bsummary\s*\*{0,2}\s*:\s*\*{0,2}\s*", re.IGNORECASE)


def _read_summaries(file_path):
    """Return the list of generated_summary strings in a .jsonl file."""
    out = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line).get("generated_summary", ""))
    return out


def _parse_model_prompt(basename, model_keys):
    """From a summary filename, return (model_label, prompt_name) or (None, None).

    LLM files look like `{model}_{prompt}_{quant}_{dataset}_{sample}_summaries.jsonl`;
    baseline files (Lead-1/TextRank/...) have no model/prompt -> (None, None).
    """
    parts = basename.split("_")
    if len(parts) >= 2 and parts[0] in model_keys and parts[1] in PROMPT_CONFIGS:
        return parts[0], parts[1]
    return None, None


def _write_md(out_path, title, intro, header, rows, total_row=None):
    """Write a Markdown doc with a title, intro line, and a table."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines = [f"# {title}", "", intro, ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    if total_row is not None:
        lines.append("| " + " | ".join(str(c) for c in total_row) + " |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}", flush=True)


def analyze_token_limit(jsonl_files, out_path, tolerance=0):
    """Count summaries that reached their prompt's max_new_tokens cap.

    A generation truncated by length ends up with ~max_new_tokens tokens, so we
    re-tokenize each summary with the model's OWN tokenizer and flag those with
    >= (max_new_tokens - tolerance) tokens. Baseline files have no cap and are
    skipped. Needs `transformers` (loads each model's tokenizer, not its weights).
    """
    from transformers import AutoTokenizer  # lazy: heavy import, only for this analysis
    from model import MODEL_CONFIGS

    tokenizers = {}  # model_label -> tokenizer (cached)
    rows, tot_n, tot_over = [], 0, 0
    for file_path in sorted(jsonl_files):
        base = os.path.basename(file_path)
        model, prompt = _parse_model_prompt(base, set(MODEL_CONFIGS))
        if model is None:
            rows.append((base, "—", "—", "—", "—", "—", "baseline (no cap)"))
            continue

        if model not in tokenizers:
            tokenizers[model] = AutoTokenizer.from_pretrained(MODEL_CONFIGS[model])
        tok = tokenizers[model]
        cap = PROMPT_CONFIGS[prompt]["max_new_tokens"]

        summaries = _read_summaries(file_path)
        over = sum(
            1 for s in summaries
            if len(tok.encode(s, add_special_tokens=False)) >= cap - tolerance
        )
        n = len(summaries)
        pct = f"{100 * over / n:.1f}%" if n else "—"
        rows.append((base, model, prompt, cap, n, over, pct))
        tot_n += n
        tot_over += over

    tot_pct = f"{100 * tot_over / tot_n:.1f}%" if tot_n else "—"
    _write_md(
        out_path,
        "Token-limit analysis",
        "Generated summaries whose re-tokenized length reached the prompt's "
        "`max_new_tokens` cap (i.e. generation was cut off by length, not a stop "
        "token). Baseline files have no cap and are skipped.",
        ["file", "model", "prompt", "cap", "n", "at_cap", "at_cap %"],
        rows,
        total_row=("**TOTAL**", "", "", "", tot_n, tot_over, tot_pct),
    )


def analyze_summary_marker(jsonl_files, out_path):
    """Count summaries that do NOT contain a "Summary:"-style marker.

    "Has a marker" = contains any SUMMARY_MARKERS substring (case-insensitive).
    Reports both counts so you can read it either way.
    """
    rows, tot_n, tot_no = [], 0, 0
    for file_path in sorted(jsonl_files):
        summaries = _read_summaries(file_path)
        no_marker = sum(1 for s in summaries if not SUMMARY_MARKERS.search(s))
        n = len(summaries)
        pct = f"{100 * no_marker / n:.1f}%" if n else "—"
        rows.append((os.path.basename(file_path), n, n - no_marker, no_marker, pct))
        tot_n += n
        tot_no += no_marker

    tot_pct = f"{100 * tot_no / tot_n:.1f}%" if tot_n else "—"
    _write_md(
        out_path,
        "Summary-marker analysis",
        "Generated summaries that do NOT contain a `Summary:` marker "
        f"(regex `{SUMMARY_MARKERS.pattern}`).",
        ["file", "n", "with_marker", "no_marker", "no_marker %"],
        rows,
        total_row=("**TOTAL**", tot_n, tot_n - tot_no, tot_no, tot_pct),
    )


def analyze_sanity_check(jsonl_files, out_path, n=5, seed=42):
    """Markdown spot-check of generated summaries.

    For each .jsonl, writes the filename as a `##` heading and each sampled
    generated summary under a `###` heading. The SAME `n` record indices (chosen
    once with `seed`) are used for every file, so the same articles can be compared
    across all systems side by side.
    """
    files = sorted(jsonl_files)
    indices = None
    lines = ["# Sanity check — generated summaries", ""]
    for file_path in files:
        summaries = _read_summaries(file_path)
        if indices is None and summaries:  # pick shared indices once, from first non-empty file
            k = min(n, len(summaries))
            indices = sorted(random.Random(seed).sample(range(len(summaries)), k))
        lines.append(f"## {os.path.basename(file_path)}")
        lines.append("")
        for i in (indices or []):
            if i < len(summaries):
                lines.append(f"### Summary {i}")
                lines.append("")
                lines.append(summaries[i].strip())
                lines.append("")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}", flush=True)


def _job_id(basename):
    """Trailing job id from `sum_..._{jobid}.out`, or -1 if not numeric."""
    stem = basename[:-4] if basename.endswith(".out") else basename
    last = stem.split("_")[-1]
    return int(last) if last.isdigit() else -1


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _classify_status(out_text, err_text):
    """(status, detail) for one job from its .out/.err contents."""
    if "Done in" in out_text:
        return "finished", ""
    both = out_text + "\n" + err_text
    if re.search(r"due to time limit", both, re.I):
        return "stopped_early", "time limit"
    if re.search(r"traceback \(most recent call last\)|cuda out of memory|out of memory", both, re.I):
        return "stopped_early", "error"
    # SLURM epilogue / cancellation appears only once a job has ENDED (a running
    # job hasn't written it yet), so its presence without "Done in" = stopped early.
    if re.search(r"cancelled|slurmstepd: error|job wall-clock time|state:\s*(failed|timeout|cancelled|out_of_memory)",
                 both, re.I):
        return "stopped_early", "ended without finishing"
    return "running", ""


def _latest_baselines_jobid(logs_dir):
    """Job id of the most recent baselines job (`baselines_{jobid}.out`), or -1.

    Baselines is submitted first in a run, so it's the lower bound for that run's
    summarization job ids — anything with a higher id belongs to the current run.
    """
    ids = [_job_id(os.path.basename(p))
           for p in glob.glob(os.path.join(logs_dir, "**", "baselines_*.out"), recursive=True)]
    return max(ids) if ids else -1


def analyze_job_status(logs_dir, expected_configs, out_path):
    """Report each config's status, scoped to the CURRENT run.

    The current run = jobs submitted after the latest baselines job (baselines is
    submitted first, so its id is the lower bound). A config counts only if its
    newest log has a job id > that cutoff; a config whose only logs are older (from
    a previous run) is reported as 'not_started' — as is one with no log at all.
    """
    cutoff = _latest_baselines_jobid(logs_dir)

    latest = {}  # config -> (jobid, out_path): overall newest log per config
    for path in glob.glob(os.path.join(logs_dir, "**", "sum_*.out"), recursive=True):
        base = os.path.basename(path)
        cfg = _parse_name(base)
        if cfg is None:
            continue
        jid = _job_id(base)
        if cfg not in latest or jid > latest[cfg][0]:
            latest[cfg] = (jid, path)

    rows, counts = [], {}
    for cfg in set(expected_configs) | set(latest):
        entry = latest.get(cfg)
        if entry and entry[0] > cutoff:  # a log from the current run exists
            jid, path = entry
            status, detail = _classify_status(_read_text(path), _read_text(path[:-4] + ".err"))
        elif entry:  # only older logs -> treat as not started this run
            status, detail, jid = "not_started", f"older run ({entry[0]})", "—"
        else:  # no log at all
            status, detail, jid = "not_started", "", "—"
        rows.append((*cfg, status, detail, jid))
        counts[status] = counts.get(status, 0) + 1

    order = {"running": 0, "stopped_early": 1, "not_started": 2, "finished": 3}
    rows.sort(key=lambda r: (order.get(r[4], 9), r[:4]))

    cutoff_note = f"jobs after baselines job {cutoff}" if cutoff >= 0 else "all logs (no baselines log found)"
    summary = ", ".join(f"{counts[k]} {k}" for k in
                        ("running", "stopped_early", "not_started", "finished") if counts.get(k))
    _write_md(
        out_path,
        "Job status",
        f"Status of each config for the current run ({cutoff_note}) from `{logs_dir}/`. "
        f"**{summary or 'no jobs'}.**",
        ["model", "quant", "prompt", "dataset", "status", "detail", "job id"],
        rows,
    )


def main():
    parser = argparse.ArgumentParser(description="Analyse generated summaries.")
    parser.add_argument("--output_dir", default="summaries",
                        help="Directory of *_summaries.jsonl files.")
    parser.add_argument("--results_dir", default="results",
                        help="Where to write the Markdown reports.")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="Count as 'at cap' if token count >= cap - tolerance.")
    parser.add_argument("--sanity_n", type=int, default=5,
                        help="How many summaries per file in the sanity-check report.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the (shared) sanity-check sample indices.")
    parser.add_argument("--status-only", action="store_true",
                        help="Only write the job-status report (logs only).")
    args = parser.parse_args()

    # Job status depends only on logs — run it first so it works even before any
    # summaries exist (e.g. while jobs are still running). The expected grid is every
    # model x quant x prompt x dataset (all datasets, including the xu_* ones).
    expected = [(m, q, p, d) for m in _STATUS_MODELS for q in _STATUS_QUANTS
                for p in PROMPT_CONFIGS for d in DATASET_CONFIGS]
    analyze_job_status("logs", expected,
                       os.path.join(args.results_dir, "job_status.md"))
    if args.status_only:
        return

    files = sorted(glob.glob(os.path.join(args.output_dir, "*.jsonl")))
    if not files:
        print(f"No .jsonl files found in {args.output_dir} — skipping summary analyses.", flush=True)
        return

    analyze_token_limit(files, os.path.join(args.results_dir, "token_limit_analysis.md"),
                        tolerance=args.tolerance)
    analyze_summary_marker(files, os.path.join(args.results_dir, "summary_marker_analysis.md"))
    analyze_sanity_check(files, os.path.join(args.results_dir, "sanity_check.md"),
                        n=args.sanity_n, seed=args.seed)


if __name__ == "__main__":
    main()

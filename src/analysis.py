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
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.output_dir, "*.jsonl")))
    if not files:
        print(f"No .jsonl files found in {args.output_dir}.", flush=True)
        return

    analyze_token_limit(files, os.path.join(args.results_dir, "token_limit_analysis.md"),
                        tolerance=args.tolerance)
    analyze_summary_marker(files, os.path.join(args.results_dir, "summary_marker_analysis.md"))
    analyze_sanity_check(files, os.path.join(args.results_dir, "sanity_check.md"),
                        n=args.sanity_n, seed=args.seed)


if __name__ == "__main__":
    main()

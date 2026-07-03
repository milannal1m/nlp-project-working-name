"""Post-hoc analysis of generated summaries.

Standalone helper (not part of the summarize/evaluate pipeline). Reads the
`.jsonl` files in the summaries dir and writes Markdown reports to results/.

Run from the repo root:
    python src/analysis.py                       # all summaries/*.jsonl
    python src/analysis.py --output_dir summaries --results_dir results

Currently provides two analyses:
  1. token_limit   — how many summaries hit their prompt's max_new_tokens cap
  2. summary_marker — how many summaries lack a "Summary:"-style marker
"""

import argparse
import glob
import json
import os

from prompts import PROMPT_CONFIGS

# A summary "has a marker" if it contains any of these (case-insensitive).
# Edit this list to match whatever scaffolding you want to look for.
SUMMARY_MARKERS = [
    "summary:",
    "here is a summary",
    "here's a summary",
    "in summary",
    "to summarize",
    "two-sentence summary",
]


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
        no_marker = sum(
            1 for s in summaries
            if not any(m in s.lower() for m in SUMMARY_MARKERS)
        )
        n = len(summaries)
        pct = f"{100 * no_marker / n:.1f}%" if n else "—"
        rows.append((os.path.basename(file_path), n, n - no_marker, no_marker, pct))
        tot_n += n
        tot_no += no_marker

    tot_pct = f"{100 * tot_no / tot_n:.1f}%" if tot_n else "—"
    _write_md(
        out_path,
        "Summary-marker analysis",
        "Generated summaries that do NOT contain a summary marker "
        f"(any of: {', '.join(repr(m) for m in SUMMARY_MARKERS)}).",
        ["file", "n", "with_marker", "no_marker", "no_marker %"],
        rows,
        total_row=("**TOTAL**", tot_n, tot_n - tot_no, tot_no, tot_pct),
    )


def main():
    parser = argparse.ArgumentParser(description="Analyse generated summaries.")
    parser.add_argument("--output_dir", default="summaries",
                        help="Directory of *_summaries.jsonl files.")
    parser.add_argument("--results_dir", default="results",
                        help="Where to write the Markdown reports.")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="Count as 'at cap' if token count >= cap - tolerance.")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.output_dir, "*.jsonl")))
    if not files:
        print(f"No .jsonl files found in {args.output_dir}.", flush=True)
        return

    analyze_token_limit(files, os.path.join(args.results_dir, "token_limit_analysis.md"),
                        tolerance=args.tolerance)
    analyze_summary_marker(files, os.path.join(args.results_dir, "summary_marker_analysis.md"))


if __name__ == "__main__":
    main()

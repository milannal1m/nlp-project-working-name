"""Estimate SLURM --time for a summarization job from historical log durations.

Standalone, pure-stdlib helper. `run_experiment.sh` calls the CLI once per job to
set a per-config `--time` instead of a flat 36h; `--report` writes an average-
duration table for inspection.

How it works: SLURM writes one log per job named
`sum_{model}_{quant}_{prompt}_{dataset}_{jobid}.out`, and every completed run prints
`Done in X.XX min` plus `sample N/...` progress lines (from src/main.py). We recover
(done_min, n_articles) per config, turn it into a per-article rate, and scale by the
requested article count (+ a safety margin).

CLI:
    # one HH:MM:SS for a config (falls back if unmeasured)
    python src/job_time.py --model Llama --quant 8bit --prompt P3 --dataset cnn_dailymail
    python src/job_time.py --model Llama --quant 4bit --prompt P1 --dataset xu_cnndm --sample 500

    # average-duration report
    python src/job_time.py --report results/job_time_analysis.md
"""

import argparse
import glob
import math
import os
import re

_DONE_RE = re.compile(r"Done in ([0-9.]+) min")
_SAMPLE_RE = re.compile(r"sample (\d+)/")
# SLURM epilogue line, e.g. "Job Wall-clock time: 1-12:00:03" (D-HH:MM:SS or HH:MM:SS).
_WALL_RE = re.compile(r"Job Wall-clock time:\s*([0-9:-]+)")


def _wallclock_to_minutes(s):
    days, _, hms = s.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return (int(days) if days else 0) * 1440 + h * 60 + m + sec / 60.0


def _parse_name(basename):
    """`sum_{model}_{quant}_{prompt}_{dataset}_{jobid}.out` -> (model, quant, prompt, dataset)."""
    stem = basename[:-4] if basename.endswith(".out") else basename
    parts = stem.split("_")
    if len(parts) < 6 or parts[0] != "sum":
        return None
    # parts = ['sum', model, quant, prompt, <dataset...>, jobid]
    model, quant, prompt = parts[1], parts[2], parts[3]
    dataset = "_".join(parts[4:-1])  # dataset may contain underscores (cnn_dailymail, xu_cnndm)
    return model, quant, prompt, dataset


def parse_logs(logs_dir="logs"):
    """Yield one record per summarization log with a recoverable per-article rate.

    Two kinds contribute a rate:
      - completed runs: `Done in X min` over `n` articles.
      - timed-out runs (CANCELLED ... DUE TO TIME LIMIT): the SLURM `Job Wall-clock
        time:` over the `n` articles it managed before being killed. That rate is a
        good (slightly conservative) estimate for exactly the cells that don't fit.
    """
    records = []
    for path in glob.glob(os.path.join(logs_dir, "**", "sum_*.out"), recursive=True):
        cfg = _parse_name(os.path.basename(path))
        if cfg is None:
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        samples = [int(m) for m in _SAMPLE_RE.findall(text)]
        n = max(samples) if samples else None
        if n is None:
            continue

        done = _DONE_RE.search(text)
        if done:
            minutes, complete = float(done.group(1)), True
        else:
            wall = _WALL_RE.search(text)
            if not wall:
                continue  # neither finished nor a recorded wall-clock -> unusable
            minutes, complete = _wallclock_to_minutes(wall.group(1)), False

        records.append({
            "config": cfg, "minutes": minutes, "n_articles": n,
            "rate": minutes / n, "complete": complete,
        })
    return records


def _full_size(records, dataset):
    """Best estimate of a dataset's article count: the largest n seen for it."""
    ns = [r["n_articles"] for r in records if r["config"][3] == dataset]
    return max(ns) if ns else None


def _weighted_rate(records, pred):
    """Article-weighted per-article rate = total minutes / total articles.

    Weighting by article count means short runs (e.g. a 5-sample smoke test, whose
    rate is dominated by fixed startup/warmup overhead) contribute almost nothing,
    while a full ~11k-article run dominates — so mixing the two no longer inflates
    the estimate.
    """
    matched = [r for r in records if pred(r)]
    tot_n = sum(r["n_articles"] for r in matched)
    return (sum(r["minutes"] for r in matched) / tot_n) if tot_n else None


def estimate_minutes(records, model, quant, prompt, dataset, sample):
    """Predicted minutes for a config = per-article rate x article count, or None.

    Rate is the article-weighted rate over the exact (model, quant, prompt, dataset)
    logs, falling back to the same (model, quant, prompt) across datasets (so xu_*
    reuses cnn/xsum, and a cell measured only on one dataset covers the other).
    Article count is the requested `sample`, or the dataset's full size for full runs.
    """
    rate = _weighted_rate(records, lambda r: r["config"] == (model, quant, prompt, dataset))
    if rate is None:
        rate = _weighted_rate(records, lambda r: r["config"][:3] == (model, quant, prompt))
    if rate is None:
        return None

    n = sample if sample is not None else _full_size(records, dataset)
    if n is None:
        return None
    return rate * n


def _hms_to_minutes(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 60 + m + math.ceil(s / 60)


def to_hms(minutes, margin=1.25, floor_min=30, max_hms="48:00:00"):
    """Apply margin, round up to whole minutes, clamp, format as HH:MM:SS."""
    total = math.ceil(minutes * margin)
    total = max(floor_min, min(total, _hms_to_minutes(max_hms)))
    return f"{total // 60:02d}:{total % 60:02d}:00"


def write_report(records, out_path):
    """Markdown table of per-config rate and estimated full-run time."""
    agg = {}
    for r in records:
        agg.setdefault(r["config"], []).append(r)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines = ["# Job-time analysis", "",
             "Per-article rate and estimated full-run time per "
             "`(model, quant, prompt, dataset)`, parsed from `logs/**/sum_*.out`. "
             "Timed-out runs contribute a rate from their SLURM wall-clock; they are "
             "flagged `timeout` in the status column.", "",
             "| model | quant | prompt | dataset | runs | status | min/article | est full (h) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for cfg in sorted(agg):
        rs = agg[cfg]
        rate = _weighted_rate(records, lambda r, c=cfg: r["config"] == c)
        full = _full_size(records, cfg[3])
        est_h = f"{rate * full / 60:.1f}" if full else "—"
        status = "ok" if all(r["complete"] for r in rs) else "timeout"
        lines.append(f"| {cfg[0]} | {cfg[1]} | {cfg[2]} | {cfg[3]} | {len(rs)} | "
                     f"{status} | {rate:.4f} | {est_h} |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({len(agg)} configs)", flush=True)


def main():
    p = argparse.ArgumentParser(description="Estimate SLURM --time from historical logs.")
    p.add_argument("--logs_dir", default="logs")
    p.add_argument("--report", metavar="PATH", help="Write an average-duration Markdown report and exit.")
    p.add_argument("-m", "--model")
    p.add_argument("-q", "--quant")
    p.add_argument("-p", "--prompt")
    p.add_argument("-d", "--dataset")
    p.add_argument("-s", "--sample", type=int, default=None)
    p.add_argument("--margin", type=float, default=1.25)
    p.add_argument("--floor_min", type=int, default=30)
    p.add_argument("--max", dest="max_hms", default="48:00:00")
    p.add_argument("--fallback", default="36:00:00")
    args = p.parse_args()

    records = parse_logs(args.logs_dir)

    if args.report:
        write_report(records, args.report)
        return

    if not all([args.model, args.quant, args.prompt, args.dataset]):
        p.error("--model, --quant, --prompt, --dataset are required (or use --report)")

    minutes = estimate_minutes(records, args.model, args.quant, args.prompt,
                               args.dataset, args.sample)
    if minutes is None:
        print(args.fallback)  # unmeasured config -> generous default
    else:
        print(to_hms(minutes, margin=args.margin, floor_min=args.floor_min, max_hms=args.max_hms))


if __name__ == "__main__":
    main()

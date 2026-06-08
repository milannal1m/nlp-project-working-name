# Whole-Dataset Results (full test split)

Evaluation on the **complete HuggingFace test split** (`SAMPLE=None`): CNN/DailyMail = 11,490, XSum = 11,334
samples per model. Generated on bwUniCluster — 2026-06-06, completed 2026-06-08.

## Scope of this snapshot

**Complete: all 20 of 20** model×dataset combinations finished the full split and are
included here. The four combinations that had previously hit the 24 h GPU wall-clock
limit were re-run to completion (with generation resume support) and added:

- **Llama_8bit / XSum** — 11,334 samples
- **Phi-3_4bit / XSum** — 11,334 samples
- **Phi-3_8bit / CNN/DailyMail** — 11,490 samples
- **Phi-3_8bit / XSum** — 11,334 samples

SummaC was also recomputed for all combinations with a more robust scorer (chunked
batches with per-example fallback, empty summaries skipped).

## Contents
- `results.md` — per-dataset comparison tables, best-per-metric, failure notes, charts.
- `results.csv` — same numbers, machine-readable.
- `metrics/` — one JSON per (model, dataset) with full metric + length stats.
- `charts/` — bar charts per metric + per-dataset heatmaps.

## Metric notes
- **QAFactEval** is empty everywhere (`qafacteval` not installed on the cluster).
- **SummaC** is now computed for all combinations, including the extractive baselines
  on XSum that previously failed (fixed by the chunked/per-example scorer).

# Whole-Dataset Results (full test split)

Evaluation on the **complete HuggingFace test split** (`SAMPLE=None`): CNN/DailyMail = 11,490, XSum = 11,334
samples per model. Generated on bwUniCluster — 2026-06-06.

## Scope of this snapshot

**16 of 20** model×dataset combinations finished the full
split and are included here. The remaining **4** were still generating
when their GPU jobs hit the 24 h wall-clock limit; they are **excluded** from this
snapshot and are being **re-run** (with generation resume support) to complete the
full split, after which they will be added.

### Pending re-run (not in this snapshot)
- **Llama_8bit / XSum** — had 1,487 / 11,334 samples at snapshot time
- **Phi-3_4bit / XSum** — had 2,749 / 11,334 samples at snapshot time
- **Phi-3_8bit / CNN/DailyMail** — had 8,669 / 11,490 samples at snapshot time
- **Phi-3_8bit / XSum** — had 500 / 11,334 samples at snapshot time

Re-run SLURM pipeline (gen → eval → aggregate):
`GEN1=4944335  GEN2=4944336  GEN3=4944337  EVAL=4944338  AGG=4944339`

## Contents
- `results.md` — per-dataset comparison tables, best-per-metric, failure notes, charts.
- `results.csv` — same numbers, machine-readable.
- `metrics/` — one JSON per (model, dataset) with full metric + length stats.
- `charts/` — bar charts per metric + per-dataset heatmaps.

## Metric notes
- **QAFactEval** is empty everywhere (`qafacteval` not installed on the cluster).
- **SummaC** is missing for some extractive baselines on XSum (input-truncation error).

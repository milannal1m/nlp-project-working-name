# Parallel Evaluation Pipeline (bwUniCluster 3.0)

Runs **all models** over **all datasets in parallel** on SLURM, evaluates them
with the full metric suite, and produces a CSV, a Markdown report, and
comparison charts.

## Model matrix (10 configs)

| Type | Models |
|------|--------|
| Baselines (CPU) | Lead-1, Lead-3, TextRank, TF-IDF |
| LLMs (GPU) | Llama-3.2-3B-Instruct × {None, 4bit, 8bit}, Phi-3-mini-4k-instruct × {None, 4bit, 8bit} |

Datasets: **CNN/DailyMail**, **XSum** — first **500** articles each
(`shuffle(seed=42)`, so reused full-test-set files and freshly generated files
score the same slice). Edit `pipeline_config.py` to change the matrix / sample.

## Metrics

BLEU, ROUGE-L, METEOR, BERTScore-F1, SummaC, QAFactEval.
**ROUGE-1 and ROUGE-2 are intentionally excluded.** QAFactEval is best-effort:
if it is not installed, that column is recorded as `null` and the rest proceed.

## How it runs

Four dependency-chained SLURM stages (`slurm/submit_all.sh`):

```
gen_baselines (CPU array 0-3) ┐
                              ├─ afterany ─► evaluate (GPU array 0-19) ─► aggregate (CPU)
gen_llms     (GPU array 4-9)  ┘
```

- **Generation is idempotent** — any summary file that already has ≥ 500 records
  is skipped, so existing committed summaries are reused, not regenerated.
- `afterany` (not `afterok`) keeps the pipeline robust: one failed task does not
  block the rest; the final report shows missing metrics as `—`.

## Usage

```bash
# 1. One-time environment setup (login node)
bash slurm/setup_env.sh

# 2. (optional) inspect the matrix
python pipeline_config.py

# 3. Submit the whole pipeline
bash slurm/submit_all.sh

# 4. Monitor
watch -n 10 squeue --me
tail -f logs/sum-*_*.out
```

## Outputs

```
results/results.csv          one row per (model, dataset), all metrics
results/results.md           comparison tables + best-per-metric + charts
results/charts/<metric>.png  grouped bar chart per metric (CNN vs XSum)
results/charts/heatmap_*.png normalised model×metric heatmap per dataset
results/metrics/*.json       raw per-file metric output
```

## Adjusting for the cluster

- **Partitions**: GPU jobs use `gpu_a100_il`; CPU jobs use `cpu`. If your
  allocation differs, edit the `#SBATCH --partition=` lines (check with
  `sinfo`).
- **Concurrency**: all array tasks are submitted at once; SLURM runs as many
  simultaneously as your QOS / GPU quota allows. The rest queue and start as
  slots free up — the pipeline still completes, just not literally all at once
  if the cluster is busy.
- **Re-running aggregation only**: `sbatch slurm/aggregate.sbatch` (or
  `python aggregate.py` on a login node) rebuilds the report/charts from the
  existing `results/metrics/*.json`.

## Container (optional)

`Apptainer.def` builds an image with all dependencies for full reproducibility.
The conda route above is the tested default; the container is provided for
portability. See the header of `Apptainer.def` for build/run instructions.

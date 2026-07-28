# Run the Transformer ablations locally

The wrapper runs E0–E5 sequentially on `xsum` and `cnn_dailymail`, then evaluates
each variant once. It never starts two Python jobs at the same time, so each
training/generation process exits and releases its CUDA memory before the next
one starts.

## 1. Activate the environment

Open PowerShell in the repository root and activate the Conda environment that
contains the project dependencies:

```powershell
conda activate NLP
```

Replace `NLP` if your environment has a different name. The preflight check will
stop before training if `python` is not on `PATH` or PyTorch cannot see CUDA.

The existing E0–E5 loaders require these local parquet layouts for both training
and test generation:

```text
xsum/data/train-*.parquet
xsum/data/test-*.parquet
cnn_dailymail/3.0.0/train-*.parquet
cnn_dailymail/3.0.0/test-*.parquet
```

Internet access is still used when Hugging Face evaluation assets are not
already cached.

## 2. Run the pipeline

```powershell
& .\Transformer\run_all_local.ps1
```

To preview the preflight and commands without running or writing anything:

```powershell
& .\Transformer\run_all_local.ps1 -WhatIf
```

Completed, non-empty summaries and current evaluation outputs are skipped.
Force a complete rerun with:

```powershell
& .\Transformer\run_all_local.ps1 -Force
```

If Windows execution policy blocks the script, use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Transformer\run_all_local.ps1
```

Per-run console logs are also saved under `Transformer/logs_local/`. Metric
outputs are written under `Transformer/results/E0/` through
`Transformer/results/E5/`.

## 3. Change the sample sizes

Edit the configuration block at the top of `run_all_local.ps1`:

```powershell
$TRAIN_SAMPLE = 100
$SAMPLE = 100
```

`TRAIN_SAMPLE` controls training examples; `SAMPLE` controls generated and
evaluated test examples. E0–E5 hard-code the allowed values to:

```text
1, 20, 50, 100, 500
```

The requested value `200` is therefore not usable without modifying E0–E5,
which this wrapper deliberately does not do. The wrapper defaults to `100`.
After a successful small run, use `500` for both values to scale up. Changing a
sample size changes the output filename, so the new run does not overwrite a
different-size result.

## 4. Retry one failed variant manually

For example, rerun E4 for both datasets and then refresh only E4's evaluation:

```powershell
python .\Transformer\E4.py --task all --dataset xsum --train_sample 100 --sample 100 --seed 42 --device cuda --overwrite
python .\Transformer\E4.py --task all --dataset cnn_dailymail --train_sample 100 --sample 100 --seed 42 --device cuda --overwrite
python .\Transformer\E4.py --task evaluate --device cuda --output_dir .\Transformer\summaries\E4 --log_path .\Transformer\results\E4\evaluation.log
```

Use the same sample values as the wrapper configuration. E4 does not require
extra arguments: its defaults are `--max_source_length 256` and
`--attention_window 64`.

Alternatively, temporarily set `$VARIANTS = @("E4")` in the wrapper and run it
with `-Force`.

## 5. If evaluation runs out of CUDA memory

Lower `$SAMPLE` first, for example from `100` to `50`, then `20`. Lowering
`$TRAIN_SAMPLE` helps training memory/time but does not directly reduce
evaluation memory.

BERTScore uses `roberta-large` on CUDA with an internal batch size of `64`.
Neither that batch size nor the evaluation device is exposed by the E0–E5 CLI;
the `--batch_size` option controls training only. A sample below 64 therefore
also reduces BERTScore's effective batch size.

Evaluation scores every `.jsonl` in a variant's summary directory. If a larger
old output is still present, temporarily move it out of
`Transformer/summaries/E#/` before retrying; otherwise it will still be loaded
and scored. As a slower last resort, run evaluation separately with CUDA hidden
so BERTScore falls back to CPU:

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
python .\Transformer\E4.py --task evaluate --output_dir .\Transformer\summaries\E4 --log_path .\Transformer\results\E4\evaluation.log
Remove-Item Env:\CUDA_VISIBLE_DEVICES
```

## 6. Run the seed sweep

`run_all_local.ps1` produces a single run at seed `42`. To report a cross-seed
standard deviation you need the same variants at additional seeds, which is what
the companion wrapper does:

```powershell
& .\Transformer\run_seeds_local.ps1
```

It shares the preflight, sequential one-job-at-a-time execution, skip logic and
`-WhatIf` / `-Force` switches with `run_all_local.ps1`. Its configuration block
is deliberately narrower than the full ablation:

```powershell
$SEEDS = @(0, 1)
$TRAIN_SAMPLE = 500
$SAMPLE = 500
$VARIANTS = @("E0", "E1", "E5")
$DATASETS = @("xsum", "cnn_dailymail")
```

That is 2 seeds x 3 variants x 2 datasets = 12 generation runs plus 6
evaluations. Together with the existing seed-42 outputs it gives three seeds per
(variant, dataset) cell. E2-E4 are not swept; if you need them, add them to
`$VARIANTS`.

### Per-seed output layout

Each seed gets its own summary and result roots, and the sweep keeps its console
logs separate from `logs_local/`:

```text
Transformer/summaries_seed0/E{0,1,5}/Transformer_{variant}_{dataset}_500_summaries.jsonl
Transformer/results_seed0/E{0,1,5}/evaluation.{csv,log}
Transformer/summaries_seed1/...
Transformer/results_seed1/...
Transformer/logs_local_seeds/{variant}_{dataset}_500_seed{N}_all.log
Transformer/logs_local_seeds/{variant}_seed{N}_all_evaluate.log
```

The seed-42 run in `summaries/` and `results/` is never written to, so the two
wrappers can be run in either order.

### Why the per-seed directories are required

E0-E5 name their output `Transformer_{variant}_{dataset}_{sample}_summaries.jsonl`.
That filename encodes the variant, dataset and sample size but **not** the seed,
so two seeds writing into the same directory would collide on the same path and
the second would either be refused or overwrite the first. Passing a distinct
`--output_dir` per seed is the only thing separating them. The same applies to
evaluation: `--task evaluate` scores *every* `.jsonl` in the directory it is
pointed at, so mixing seeds in one directory would also merge them into a single
`evaluation.csv`.

For the same reason, do not point the sweep at `Transformer/summaries/`.

### What is not isolated

Checkpoints and vocabulary artifacts are **not** per-seed. Every run writes to
the shared paths:

```text
Transformer/checkpoints/Transformer_{variant}_{dataset}_{sample}.pt
Transformer/artifacts/Transformer_{variant}_{dataset}_{sample}_{src_vocab,trg_vocab,metadata}.json
```

Running the sweep therefore replaces the seed-42 checkpoints and artifacts for
the swept variants with those of the last seed that ran. The summaries and
metrics of the seed-42 run are unaffected, but its trained weights are not
recoverable afterwards without a rerun. Untouched variants (E2-E4) keep their
original checkpoints.

### Reading the metrics

Each `results_seed{N}/E#/evaluation.csv` has one row per scored `.jsonl` with
the columns:

```text
file,bleu,rougeL,meteor,bertscore_f1,bertscore_f1_std,qa_eval,qa_eval_std,error
```

Note that `bertscore_f1_std` is the spread of BERTScore **across the 500 test
samples within one run**. It is not the cross-seed standard deviation; that has
to be computed afterwards over the three per-seed `bertscore_f1` values. The
`qa_eval` columns stay empty and `error` reads `qafacteval not installed` unless
that optional dependency is present.

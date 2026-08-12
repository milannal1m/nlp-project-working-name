# Transformer Summarization Baseline & Ablation Study

A from-scratch **encoder–decoder Transformer** abstractive-summarization baseline
for the news-summarization benchmark, plus a **six-variant architecture/optimizer
ablation** (`E0`–`E5`). Every variant trains a small Transformer on a dataset's
official **train** split and generates summaries on its official **test** split,
producing JSONL in the exact same schema as the rest of the benchmark.

Everything runs **locally on a single machine** (CUDA GPU or CPU). All outputs
stay **under `Transformer/`** and never mix with the benchmark's root-level
`summaries/` or `results/` directories.

```
Transformer/
├── E0.py … E5.py           # the six ablation variants (model + training + decoding + CLI)
├── run_all_local.ps1       # wrapper: run variants sequentially, then evaluate
├── run_seeds_local.ps1     # wrapper: repeat E0/E1/E5 at extra seeds for cross-seed std
├── README.md               # this file
├── RUN_LOCAL.md            # operational runbook (preflight, retries, OOM, seed sweep)
├── checkpoints/            # trained model checkpoints (.pt)          [auto-created]
├── artifacts/              # src/trg vocab JSON + training metadata   [auto-created]
├── summaries/E{0..5}/      # generated summary JSONL (seed 42)        [auto-created]
├── results/E{0..5}/        # evaluation.log / evaluation.csv (seed 42)[auto-created]
├── summaries_seed{0,1}/    # per-seed summaries from the seed sweep   [auto-created]
├── results_seed{0,1}/      # per-seed metrics from the seed sweep     [auto-created]
├── logs_local/             # console logs of the seed-42 runs         [auto-created]
└── logs_local_seeds/       # console logs of the seed sweep           [auto-created]
```

---

## The six variants

All variants share the same model skeleton (`SelfAttention`, `TransformerBlock`,
`Encoder`, `DecoderBlock`, `Decoder`, `Transformer`), the same hyperparameters,
the same data pipeline, and greedy decoding. They differ only in the single axis
listed below, so any metric change is attributable to that one change.

| Variant | Change vs. `E0` | Details |
|---------|-----------------|---------|
| **E0**  | — (baseline)    | Post-LN residual blocks, ReLU feed-forward, plain `Adam`, full (global) encoder self-attention |
| **E1**  | **Pre-LN**      | Normalize sublayer inputs, add the raw residual, plus a final `LayerNorm` after the encoder stack and before the output projection |
| **E2**  | **GELU**        | Feed-forward activation `nn.ReLU` → `nn.GELU` |
| **E3**  | **AdamW + cosine LR** | `AdamW` with `weight_decay=0.01` on ≥2-D tensors only (biases/LayerNorm excluded), and `CosineAnnealingLR` stepped per *optimizer* step |
| **E4**  | **Local attention** | Encoder self-attention restricted to a band `abs(i-j) <= --attention_window` (default `64`). Cross-attention keeps the plain padding mask |
| **E5**  | **E1 + E2 + E3**  | All three "modernization" changes combined (Pre-LN + GELU + AdamW/cosine). Attention stays global |

`E4` adds one extra CLI flag, `--attention_window` (default `64`); every other
variant exposes an identical CLI.

---

## Requirements

Use the project's Python environment (the one that has `torch`, `datasets`, and —
for evaluation — `evaluate`, `bert-score`, `nltk`). Install once from the repo root:

```powershell
python -m pip install -r requirements.txt
```

- Runs on **GPU (CUDA) or CPU**. The wrappers' preflight *requires* CUDA; a
  direct `python Transformer/E0.py …` call falls back to CPU automatically.
- The reference runs below were produced on a single **NVIDIA RTX 5060 Laptop GPU**.
- The `E0`–`E5` loaders read **local parquet files** for both training and test
  generation. These must exist under the repository root:

  ```text
  xsum/data/train-*.parquet
  xsum/data/test-*.parquet
  cnn_dailymail/3.0.0/train-*.parquet
  cnn_dailymail/3.0.0/test-*.parquet
  ```

- Network access is only needed the first time evaluation downloads its
  HuggingFace assets (`roberta-large` for BERTScore, NLTK data). After that,
  everything is offline.

Run all commands **from the repository root** so the scripts can import the
benchmark's `src/dataset.py` and `src/naming.py`.

---

## Supported datasets

| `--dataset`     | Local path                  | Article field | Summary field |
|-----------------|-----------------------------|---------------|---------------|
| `cnn_dailymail` | `cnn_dailymail/3.0.0/`      | `article`     | `highlights`  |
| `xsum`          | `xsum/data/`                | `document`    | `summary`     |

A **separate** model is trained per (variant, dataset). Datasets are never combined.

---

## Split protocol (important)

- **Training** uses that dataset's official **`train`** split only, sampled with
  `shuffle(seed).select(range(min(sample, len(ds))))`.
- **Summary generation** uses that dataset's official **`test`** split only, on
  the **same sampled articles** as every other benchmark model, via
  `src/dataset.py`'s `shuffle(seed).take(sample)` logic.
- No validation or test example is ever used during training, and the vocabulary
  is built from the training portion only.

---

## Tasks

| `--task`    | What it does                                                             |
|-------------|--------------------------------------------------------------------------|
| `train`     | Train on the train split and save a checkpoint (+ vocab + metadata).     |
| `summarize` | Load an existing checkpoint and generate summaries on the test split.    |
| `all`       | `train` then `summarize` (default).                                      |
| `evaluate`  | Run the original benchmark evaluator over a summaries directory.         |

---

## Quick start

### Run everything with the wrapper

The wrapper runs the configured variants sequentially on both datasets, then
evaluates each variant once. It never starts two Python jobs at the same time,
so each process exits and releases its CUDA memory before the next one starts.

```powershell
conda activate NLP
& .\Transformer\run_all_local.ps1
```

Useful switches:

```powershell
& .\Transformer\run_all_local.ps1 -WhatIf   # preflight + print commands, write nothing
& .\Transformer\run_all_local.ps1 -Force    # re-run even completed outputs
```

Completed, non-empty summaries and current evaluation outputs are skipped by
default. If Windows execution policy blocks the script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Transformer\run_all_local.ps1
```

The configuration block at the top of the script controls the run:

```powershell
$SEEDS        = @(0, 1)
$TRAIN_SAMPLE = 500
$SAMPLE       = 500
$VARIANTS     = @("E0", "E1", "E5")
$DATASETS     = @("xsum", "cnn_dailymail")
$DEVICE       = "cuda"
```

> **Check this block before every run.** `run_all_local.ps1` and
> `run_seeds_local.ps1` are currently the *same* script (identical apart from
> line endings), both configured for the seed sweep. As shipped, `run_all_local.ps1`
> writes into `summaries_seed{0,1}/`, `results_seed{0,1}/` and `logs_local_seeds/`
> — **not** into `summaries/`, `results/` and `logs_local/`. To reproduce the
> seed-42 ablation in `results/`, either run the six variants directly (below) or
> adjust the script's seed/output roots first. See
> [Known issues](#known-issues).

See [`RUN_LOCAL.md`](RUN_LOCAL.md) for the full runbook: preflight details,
retrying a single failed variant, handling CUDA OOM during BERTScore, and the
seed-sweep layout.

### Run a single variant directly

```powershell
# fastest sanity check (1 train + 1 test example)
python .\Transformer\E0.py --task all --dataset xsum --sample 1

# a full ablation cell
python .\Transformer\E5.py --task all --dataset xsum `
    --train_sample 500 --sample 500 --seed 42 --device cuda `
    --output_dir .\Transformer\summaries\E5

# score just that variant
python .\Transformer\E5.py --task evaluate `
    --output_dir .\Transformer\summaries\E5 `
    --log_path   .\Transformer\results\E5\evaluation.log
```

`--sample` and `--train_sample` accept exactly `{1, 20, 50, 100, 500}`. If
`--train_sample` is omitted it defaults to `--sample`.

---

## Results

Reference configuration for every number below: `--train_sample 500`,
`--sample 500`, 5 epochs, batch size 2, lr `3e-4`, `embed_size=128`,
`num_layers=2`, `heads=4`, `forward_expansion=2`, `dropout=0.1`,
`max_source_length=256`, `max_target_length=64`, greedy decoding. That is roughly
**3.5 M parameters** per model, and about **16–20 s of training per (variant,
dataset)** on the RTX 5060.

### Seed 42 — all six variants (500 test examples)

Source: `Transformer/results/E{0..5}/evaluation.csv`.

| Variant | Dataset | BLEU | ROUGE-L | METEOR | BERTScore F1 | final train loss |
|---------|---------|-----:|--------:|-------:|-------------:|-----------------:|
| E0 | xsum          | 0.0000 | 0.1406 | 0.1148 |  0.0188 | 5.8198 |
| E1 | xsum          | 0.0000 | **0.1439** | **0.1207** | **0.0712** | **5.2998** |
| E2 | xsum          | 0.0000 | 0.1424 | 0.1155 |  0.0318 | 5.7953 |
| E3 | xsum          | 0.0000 | 0.1389 | 0.1121 | -0.1411 | 6.0612 |
| E4 | xsum          | 0.0000 | 0.1408 | 0.1142 |  0.0179 | 5.8197 |
| E5 | xsum          | 0.0000 | 0.1194 | 0.1026 | -0.0769 | 5.7407 |
| E0 | cnn_dailymail | 0.0000 | 0.0465 | 0.0346 | -0.2387 | 6.5627 |
| E1 | cnn_dailymail | 0.0009 | **0.0983** | **0.0738** | -0.2106 | **6.1304** |
| E2 | cnn_dailymail | 0.0000 | 0.0411 | 0.0344 | **-0.1499** | 6.5474 |
| E3 | cnn_dailymail | 0.0000 | 0.0630 | 0.0404 | -0.3636 | 6.7229 |
| E4 | cnn_dailymail | 0.0000 | 0.0460 | 0.0351 | -0.2282 | 6.5624 |
| E5 | cnn_dailymail | 0.0000 | 0.0730 | 0.0436 | -0.2305 | 6.4914 |

Bold = best in that dataset block. BERTScore F1 is the raw (unrescaled) value,
so negatives are expected for weak generations.

### Cross-seed mean ± std (seeds 42, 0, 1)

`E0`, `E1` and `E5` were repeated at seeds `0` and `1`
(`Transformer/results_seed{0,1}/`). Std is the sample standard deviation over the
three per-seed scores.

| Variant | Dataset | ROUGE-L | METEOR | BERTScore F1 |
|---------|---------|--------:|-------:|-------------:|
| E0 | xsum          | 0.1455 ± 0.0096 | 0.1192 ± 0.0126 | -0.0075 ± 0.0363 |
| E1 | xsum          | 0.1434 ± 0.0038 | 0.1226 ± 0.0029 |  0.0453 ± 0.0302 |
| E5 | xsum          | 0.1375 ± 0.0158 | 0.1111 ± 0.0087 | -0.0648 ± 0.0299 |
| E0 | cnn_dailymail | 0.0644 ± 0.0341 | 0.0448 ± 0.0130 | -0.2423 ± 0.0534 |
| E1 | cnn_dailymail | 0.0877 ± 0.0137 | 0.0732 ± 0.0035 | -0.2334 ± 0.0256 |
| E5 | cnn_dailymail | 0.0874 ± 0.0130 | 0.0526 ± 0.0082 | -0.2036 ± 0.0251 |

BLEU stays at or near `0.0000` for every cell (max observed: `0.0069`, E0/xsum/seed 0).

### How to read these numbers

- **The scores are low by design.** These are ~3.5 M-parameter models trained
  from scratch for 5 epochs on 500 examples. They are a *floor* for the
  benchmark, not a competitive summarizer. BLEU ≈ 0 simply means the greedy
  output almost never reproduces a reference 4-gram.
- **XSum scores higher than CNN/DailyMail** on every variant. XSum references
  are single short sentences, which is much closer to what an undertrained model
  emits; CNN/DailyMail's multi-sentence `highlights` are far out of reach.
- **The ablation differences are inside seed noise.** On CNN/DailyMail the
  cross-seed std for ROUGE-L (±0.013 to ±0.034) is as large as, or larger than,
  the gaps between variants. With three seeds and this training budget, **no
  variant is significantly better than `E0`** on the generation metrics. Report
  them as "no measurable effect at this scale", not as a ranking.
- **Training loss is the cleaner signal.** It is far less noisy than the
  generation metrics, and there `E1` (Pre-LN) is consistently the best on both
  datasets — the expected result, since Pre-LN mainly buys optimization
  stability. `E3` (AdamW + cosine) is consistently the *worst* on train loss:
  annealing the LR to 0 over only 5 epochs cuts the effective training budget.
- **`E5` does not stack.** Combining E1+E2+E3 lands between them rather than
  above them, because it inherits E3's LR schedule.
- **`E4` is nearly identical to `E0`** (ROUGE-L 0.1408 vs 0.1406 on xsum). With
  `max_source_length=256` and `attention_window=64`, the local band removes
  little that this model was using anyway.

---

## Output paths

For variant `V`, dataset `D`, and sample size `S` (checkpoints keyed by
`--train_sample`, which defaults to `--sample`):

| Artifact          | Path                                                          |
|-------------------|---------------------------------------------------------------|
| Checkpoint        | `Transformer/checkpoints/Transformer_{V}_{D}_{S}.pt`          |
| Source vocab      | `Transformer/artifacts/Transformer_{V}_{D}_{S}_src_vocab.json`|
| Target vocab      | `Transformer/artifacts/Transformer_{V}_{D}_{S}_trg_vocab.json`|
| Training metadata | `Transformer/artifacts/Transformer_{V}_{D}_{S}_metadata.json` |
| Summaries         | `{--output_dir}/Transformer_{V}_{D}_{S}_summaries.jsonl`      |
| Evaluation        | `{--log_path}` (+ `evaluation.csv` beside it)                 |

Example: `E5.py --dataset xsum --sample 500 --output_dir Transformer/summaries/E5`
→ `Transformer/summaries/E5/Transformer_E5_xsum_500_summaries.jsonl`.

Each summary JSONL line matches the benchmark schema:

```json
{"news": "...", "reference_summary": "...", "generated_summary": "..."}
```

Each `evaluation.csv` has one row per scored `.jsonl`:

```text
file,bleu,rougeL,meteor,bertscore_f1,bertscore_f1_std,qa_eval,qa_eval_std,error
```

> `bertscore_f1_std` is the spread of BERTScore **across the 500 test samples
> within one run**. It is *not* the cross-seed std — that has to be computed over
> the per-seed `bertscore_f1` values, as in the table above. The `qa_eval`
> columns stay empty and `error` reads `qafacteval not installed` unless that
> optional dependency is present.

---

## CLI reference

| Argument                         | Default            | Description                                        |
|----------------------------------|--------------------|----------------------------------------------------|
| `--task`                         | `all`              | `train` / `summarize` / `all` / `evaluate`         |
| `--dataset`                      | `cnn_dailymail`    | `cnn_dailymail` or `xsum`                          |
| `--sample`                       | `20`               | Test examples to summarize: `{1,20,50,100,500}`    |
| `--train_sample`                 | = `--sample`       | Train examples: `{1,20,50,100,500}`                |
| `--epochs`                       | `5`                | Training epochs                                    |
| `--batch_size`                   | `2`                | Mini-batch size                                    |
| `--learning_rate`                | `3e-4`             | Optimizer learning rate                            |
| `--max_source_length`            | `256`              | Max article tokens                                 |
| `--max_target_length`            | `64`               | Max summary tokens                                 |
| `--embed_size`                   | `128`              | Embedding dimension                                |
| `--heads`                        | `4`                | Attention heads (must divide `embed_size`)         |
| `--num_layers`                   | `2`                | Encoder/decoder layers                             |
| `--forward_expansion`            | `2`                | Feed-forward expansion factor                      |
| `--dropout`                      | `0.1`              | Dropout probability                                |
| `--gradient_accumulation_steps`  | `1`                | Steps to accumulate before an optimizer step       |
| `--src_vocab_size`               | `30000`            | Max source vocabulary size                         |
| `--trg_vocab_size`               | `30000`            | Max target vocabulary size                         |
| `--seed`                         | `42`               | Random seed (subset selection + init)              |
| `--device`                       | auto               | `cuda` or `cpu`; auto-selects CUDA if available    |
| `--output_dir`                   | `Transformer/summaries` | Summary JSONL output directory                |
| `--log_path`                     | `Transformer/results/evaluation.log` | Evaluation log path          |
| `--overwrite`                    | off                | Overwrite an existing summary file                 |
| `--evaluate_after_generation`    | off                | Run the evaluator right after generating           |
| `--attention_window`             | `64`               | **`E4.py` only** — encoder local-attention band    |

---

## Evaluation

`--task evaluate` (and `--evaluate_after_generation`) delegate to the **original**
benchmark evaluator — no metric logic is duplicated. Point it at one variant's
summary directory so only that variant is scored:

```powershell
python .\Transformer\E0.py --task evaluate `
    --output_dir .\Transformer\summaries\E0 `
    --log_path   .\Transformer\results\E0\evaluation.log
```

Metrics: BLEU, ROUGE-L, METEOR, BERTScore (and optional QAFactEval).

Equivalent manual call (the files are fully benchmark-compatible):

```powershell
python .\src\main.py --task evaluate `
    --output_dir .\Transformer\summaries\E0 `
    --log_path   .\Transformer\results\E0\evaluation.log
```

> The evaluator scores **every** `.jsonl` in the directory it is pointed at.
> Keep one variant (and one seed) per directory, or old/foreign outputs will be
> silently folded into the same `evaluation.csv`.
>
> Because outputs live under `Transformer/summaries/`, a plain
> `python src/main.py --task evaluate` — which defaults to the root `summaries/`
> — will **not** pick them up. Use the commands above instead.

BERTScore uses `roberta-large` on CUDA with an internal batch size of `64`.
Neither that batch size nor the evaluation device is exposed by the `E0`–`E5`
CLI (`--batch_size` controls training only). If evaluation runs out of CUDA
memory, lower `--sample`, or hide CUDA so BERTScore falls back to CPU:

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
python .\Transformer\E4.py --task evaluate --output_dir .\Transformer\summaries\E4 --log_path .\Transformer\results\E4\evaluation.log
Remove-Item Env:\CUDA_VISIBLE_DEVICES
```

---

## Known issues

- **`run_all_local.ps1` and `run_seeds_local.ps1` are currently identical**
  (byte-for-byte apart from LF vs. CRLF line endings), both carrying the seed-sweep
  configuration. Running `run_all_local.ps1` as-is performs the seed sweep and
  writes to `summaries_seed{0,1}/` / `results_seed{0,1}/` / `logs_local_seeds/`.
  The seed-42 ablation in `summaries/` and `results/` was produced by an earlier
  revision of that script and cannot currently be reproduced by running it.
  Reproduce those cells with the direct `python Transformer/E{0..5}.py` calls above.
- **Checkpoints and vocab artifacts are not per-seed.** Every run writes to
  `Transformer/checkpoints/Transformer_{V}_{D}_{S}.pt` and the matching
  `artifacts/` files, which have no seed in the filename. The seed sweep therefore
  **overwrote the seed-42 weights** for `E0`, `E1` and `E5` with seed-1 weights —
  visible in `artifacts/*_metadata.json`, where those variants now report
  `"seed": 1`. Their seed-42 summaries and metrics are unaffected; only the
  trained weights are gone. `E2`–`E4` still hold their seed-42 checkpoints.
- **Summary filenames do not encode the seed** either, which is why each seed
  needs its own `--output_dir`. Never point two seeds at the same directory.

---

## Notes

- These are small models trained from scratch on tiny subsets, so with
  `--sample 1` or `20` the generated summaries may be short or even empty. That
  is genuine model output (the model learns to emit `<eos>` early), not a
  placeholder. Increase `--sample`, `--epochs`, `--embed_size`, or `--num_layers`
  for stronger results.
- Reruns are reproducible for a fixed `--seed`: same sampled examples, same
  vocabulary, same initialization.
- `summarize` fails with a clear message if the checkpoint/vocab files for the
  requested variant / `--dataset` / `--train_sample` do not exist yet — train first.
- Changing a sample size changes the output filename, so a new run never
  overwrites a different-size result.

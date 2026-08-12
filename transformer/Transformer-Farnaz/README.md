# Transformer Summarization Baseline

A from-scratch **encoder–decoder Transformer** abstractive-summarization baseline
for the news-summarization benchmark. It trains a small Transformer on a dataset's
official **train** split and generates summaries on its official **test** split,
producing JSONL that uses the exact same schema as the rest of the benchmark.

Everything this script produces stays **under `Transformer/`** and never mixes
with the benchmark's root-level `summaries/` or `results/` directories.

```
Transformer/
├── Main.py                 # the baseline (model + training + decoding + CLI)
├── README.md               # this file
├── checkpoints/            # trained model checkpoints (.pt)          [auto-created]
├── artifacts/              # src/trg vocab JSON + training metadata   [auto-created]
├── summaries/              # generated summary JSONL files            [auto-created]
└── results/                # evaluation.log / evaluation.csv          [auto-created]
```

---

## Requirements

Use the project's Python environment (the one that has `torch`, `datasets`, and —
for evaluation — `evaluate`, `bert-score`, `nltk`). Install once from the repo root:

```bash
python -m pip install -r requirements.txt
```

- Runs on **GPU (CUDA) or CPU** automatically. On CPU, keep the sample size small.
- HuggingFace datasets (`cnn_dailymail`, `xsum`) are downloaded automatically on
  first use and cached by the `datasets` library.

Run all commands **from the repository root** so the script can import the
benchmark's `src/dataset.py` and `src/naming.py`.

---

## Supported datasets

| `--dataset`     | Source (HuggingFace)        | Article field | Summary field |
|-----------------|-----------------------------|---------------|---------------|
| `cnn_dailymail` | `abisee/cnn_dailymail` (3.0.0) | `article`  | `highlights`  |
| `xsum`          | `EdinburghNLP/xsum`         | `document`    | `summary`     |

A **separate** model is trained per dataset. The two datasets are never combined.

---

## Split protocol (important)

- **Training** uses that dataset's official **`train`** split only.
- **Summary generation** uses that dataset's official **`test`** split only.
- The test split is used for the **same sampled articles** as every other
  benchmark model (same streaming + `shuffle(seed).take(sample)` logic).
- No validation or test example is ever used during training, and the vocabulary
  is built from the training portion only.

---

## Tasks

| `--task`    | What it does                                                             |
|-------------|--------------------------------------------------------------------------|
| `train`     | Train on the train split and save a checkpoint (+ vocab + metadata).     |
| `summarize` | Load an existing checkpoint and generate summaries on the test split.    |
| `all`       | `train` then `summarize` (default).                                      |
| `evaluate`  | Run the original benchmark evaluator over `Transformer/summaries/`.      |

---

## Quick start

Fastest sanity check (1 train + 1 test example):

```bash
python Transformer/Main.py --task all --dataset xsum --sample 1
```

Train + summarize in one go:

```bash
python Transformer/Main.py --task all --dataset cnn_dailymail --sample 100
```

Train only, then summarize later:

```bash
python Transformer/Main.py --task train     --dataset xsum --sample 50
python Transformer/Main.py --task summarize  --dataset xsum --sample 50
```

Generate and immediately score:

```bash
python Transformer/Main.py --task all --dataset xsum --sample 50 --evaluate_after_generation
```

Re-score everything already generated:

```bash
python Transformer/Main.py --task evaluate
```

---

## Full command matrix

`--sample` and `--train_sample` accept exactly `{1, 20, 50, 100, 500}`.
If `--train_sample` is omitted it defaults to `--sample`.

```bash
# CNN/DailyMail
python Transformer/Main.py --task all --dataset cnn_dailymail --sample 1
python Transformer/Main.py --task all --dataset cnn_dailymail --sample 20
python Transformer/Main.py --task all --dataset cnn_dailymail --sample 50
python Transformer/Main.py --task all --dataset cnn_dailymail --sample 100
python Transformer/Main.py --task all --dataset cnn_dailymail --sample 500

# XSum
python Transformer/Main.py --task all --dataset xsum --sample 1
python Transformer/Main.py --task all --dataset xsum --sample 20
python Transformer/Main.py --task all --dataset xsum --sample 50
python Transformer/Main.py --task all --dataset xsum --sample 100
python Transformer/Main.py --task all --dataset xsum --sample 500 --batch_size 2 --epochs 5
```

---

## Output paths

For a run with `--dataset D` and `--sample S` (checkpoints keyed by `--train_sample`,
which defaults to `S`):

| Artifact          | Path                                                              |
|-------------------|------------------------------------------------------------------|
| Checkpoint        | `Transformer/checkpoints/Transformer_{D}_{S}.pt`                 |
| Source vocab      | `Transformer/artifacts/Transformer_{D}_{S}_src_vocab.json`       |
| Target vocab      | `Transformer/artifacts/Transformer_{D}_{S}_trg_vocab.json`       |
| Training metadata | `Transformer/artifacts/Transformer_{D}_{S}_metadata.json`        |
| Summaries         | `Transformer/summaries/Transformer_{D}_{S}_summaries.jsonl`      |
| Evaluation        | `Transformer/results/evaluation.log` (+ `evaluation.csv`)        |

Example: `--dataset xsum --sample 100` →
`Transformer/checkpoints/Transformer_xsum_100.pt` and
`Transformer/summaries/Transformer_xsum_100_summaries.jsonl`.

Each summary JSONL line matches the benchmark schema:

```json
{"news": "...", "reference_summary": "...", "generated_summary": "..."}
```

---

## CLI reference

| Argument                         | Default            | Description                                        |
|----------------------------------|--------------------|----------------------------------------------------|
| `--task`                         | `all`              | `train` / `summarize` / `all` / `evaluate`         |
| `--dataset`                      | `cnn_dailymail`    | `cnn_dailymail` or `xsum`                           |
| `--sample`                       | `20`               | Test examples to summarize: `{1,20,50,100,500}`     |
| `--train_sample`                 | = `--sample`       | Train examples: `{1,20,50,100,500}`                 |
| `--epochs`                       | `5`                | Training epochs                                     |
| `--batch_size`                   | `2`                | Mini-batch size                                     |
| `--learning_rate`                | `3e-4`             | Adam learning rate                                  |
| `--max_source_length`            | `256`              | Max article tokens                                 |
| `--max_target_length`            | `64`               | Max summary tokens                                 |
| `--embed_size`                   | `128`              | Embedding dimension                                |
| `--heads`                        | `4`                | Attention heads (must divide `embed_size`)          |
| `--num_layers`                   | `2`                | Encoder/decoder layers                             |
| `--forward_expansion`            | `2`                | Feed-forward expansion factor                      |
| `--dropout`                      | `0.1`              | Dropout probability                                |
| `--gradient_accumulation_steps`  | `1`                | Steps to accumulate before an optimizer step        |
| `--src_vocab_size`               | `30000`            | Max source vocabulary size                         |
| `--trg_vocab_size`               | `30000`            | Max target vocabulary size                         |
| `--seed`                         | `42`               | Random seed (subset selection + init)              |
| `--device`                       | auto               | `cuda` or `cpu`; auto-selects CUDA if available     |
| `--output_dir`                   | `Transformer/summaries` | Summary JSONL output directory                |
| `--log_path`                     | `Transformer/results/evaluation.log` | Evaluation log path             |
| `--overwrite`                    | off                | Overwrite an existing summary file                 |
| `--evaluate_after_generation`    | off                | Run the evaluator right after generating           |

---

## Evaluation

`--task evaluate` (and `--evaluate_after_generation`) delegate to the **original**
benchmark evaluator — no metric logic is duplicated. They point it at
`Transformer/summaries/`, so only this baseline's outputs are scored:

```bash
python Transformer/Main.py --task evaluate
```

Metrics: BLEU, ROUGE-L, METEOR, BERTScore (and optional QAFactEval), written to
`Transformer/results/evaluation.log` and `Transformer/results/evaluation.csv`.

Equivalent manual call (the files are fully benchmark-compatible):

```bash
python src/main.py --task evaluate \
    --output_dir Transformer/summaries \
    --log_path   Transformer/results/evaluation.log
```

> Because outputs live under `Transformer/summaries/` (kept separate from the
> benchmark), a plain `python src/main.py --task evaluate` — which defaults to the
> root `summaries/` — will **not** pick them up. Use the commands above instead.

---

## Notes

- This is a small model trained from scratch on tiny subsets, so with `--sample 1`
  or `20` the generated summaries may be short or even empty. That is genuine model
  output (the model learns to emit `<eos>` early), not a placeholder. Increase
  `--sample`, `--epochs`, `--embed_size`, or `--num_layers` for stronger results.
- Reruns are reproducible for a fixed `--seed` (default `42`): same sampled
  examples, same vocabulary, same initialization.
- `summarize` fails with a clear message if the checkpoint/vocab files for the
  requested `--dataset` / `--train_sample` do not exist yet — train first.

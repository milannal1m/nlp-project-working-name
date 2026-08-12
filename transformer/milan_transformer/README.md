# From-scratch Transformers for News Summarization

Trains **one** from-scratch Transformer on XSum + CNN/DailyMail and scores it with the
repo's `Evaluator` (BLEU, ROUGE-L, METEOR, BERTScore). Two architectures are
available via `--model`:

| `--model` | File | Architecture | Decoding |
|---|---|---|---|
| `lecture` | `lecture_transformer.py` | encoder-decoder | greedy |
| `milan` | `milan_transformer.py` | decoder-only | temperature + nucleus (top-p) sampling |

---

## 1. Installation & Running

Install once (same env as the rest of the repo):

```bash
conda create -n nlp-project python=3.11 -y
conda activate nlp-project
python -m pip install -r requirements.txt
```

On the bwHPC cluster use `bash scripts/setup_env.sh` instead — it pins the exact
torch/CUDA build.

### Local

Run **from the repo root** (the script resolves `src/` and its output paths relative to it):

```bash
python transformer/milan_transformer/pipeline.py --model milan     # decoder-only
python transformer/milan_transformer/pipeline.py --model lecture   # encoder-decoder (default)
```

The device is picked automatically: CUDA → MPS (Apple Silicon) → CPU. CPU works but is
very slow, lower `TRAIN_N` / `EPOCHS` in `pipeline.py` for a local smoke test.

### Cluster (SLURM)

```bash
sbatch transformer/milan_transformer/run_pipeline.sh -m milan
```

Also submit from the repo root, so the `transformer/...` paths and the log directory
resolve. The job requests 1× A100, 8 CPUs, 16 GB RAM, 17:30 h; `-m/--model` defaults to
`milan`. Logs land in `transformer/milan_transformer/logs/`.

### Outputs

Everything is written to `transformer/milan_transformer/outputs/`:

- `{model}_{dataset}_summaries.jsonl` — news / reference / generated summary
- `{model}_evaluation.log` + `.csv` — metrics (overwritten per run)
- `{model}_sanity_check_gold.md` — 5 random gold-vs-generated pairs for eyeballing

---

## 2. `pipeline.py`

Shared driver for both models: it loads the data, builds the model-specific pieces
(`build_lecture` / `build_milan` return `model, loader, step, generate`), runs one shared
training loop, then generates and evaluates. Training data comes from the `train` split,
evaluation from the `test` split, so they are disjoint. Loss is cross-entropy over the
vocabulary at each position (padding ignored); for the decoder-only model the loss is
masked to the **summary** tokens only.

The only CLI flag is `--model {lecture,milan}` (default `lecture`). Everything else is a
constant at the top of `pipeline.py`:

| Constant | Default | Meaning |
|---|---|---|
| `DATASETS` | `["xsum", "cnn_dailymail"]` | trained jointly on both |
| `TRAIN_N` | `100000` | training examples **per dataset** |
| `EVAL_N` | `600` | evaluated examples per dataset |
| `EPOCHS` | `100` | training epochs |
| `BATCH` | `16` | batch size |
| `LR` | `3e-4` | Adam learning rate (grad-norm clipped at 1.0) |
| `MAX_SRC` | `400` | max article tokens |
| `MAX_TRG` | `64` | max summary tokens (also the generation limit) |
| `EMBED` | `256` | embedding / model dimension |
| `LAYERS` | `3` | transformer blocks |
| `HEADS` | `8` | attention heads (`EMBED % HEADS == 0`) |
| `TEMPERATURE` | `0.7` | sampling temperature — `milan` only |
| `TOP_P` | `0.9` | nucleus threshold — `milan` only |
| `TOKENIZER` | `bert-base-uncased` | HF tokenizer (`[CLS]` = SOS, `[SEP]` = EOS) |
| `SEED` | `42` | torch + data sampling seed |

---

## 3. Lecture vs. Milan transformer

`lecture_transformer.py` is the **encoder-decoder Transformer from the lecture**, used
unchanged as the baseline: the article goes through the encoder, the decoder cross-attends
to it and emits the summary greedily (argmax at each step).

`milan_transformer.py` is my own variant. Two changes:

1. **Decoder-only.** No encoder and no cross-attention — just a stack of causal
   self-attention blocks. Article and summary are fed as a single stream
   `[CLS] article [SEP] summary`, and the loss is masked so only the summary tokens count
   (the article acts purely as a prompt). The attention mask combines padding with a
   lower-triangular causal mask.
2. **Nucleus (top-p) sampling instead of greedy decoding.** `generate()` applies a
   temperature softmax to the last position's logits, then `sample_top_p()` keeps the
   smallest set of tokens whose cumulative probability exceeds `top_p`, renormalises and
   samples from it. This trades the repetitive, deterministic output of greedy decoding
   for more varied summaries.

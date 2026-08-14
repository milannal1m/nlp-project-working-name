# News Summarization Benchmark

A news summarization benchmark comparing instruction-tuned LLMs (Llama, Phi, Qwen2) across
prompts and quantization levels (16/8/4-bit) against extractive baselines. Summaries are
generated as JSONL and scored with BLEU, ROUGE-L, METEOR and BERTScore.

The repo also hosts several self-contained sub-projects, each with its own environment and
README. This file covers the root pipeline and points at the rest.

---

## Repo map

| Path | What it is |
|---|---|
| `src/` | The root benchmark pipeline (see the file table below) |
| `scripts/` | SLURM workers + conda env setup |
| `run_experiment.sh` | **Main entry point** — submits the whole grid in parallel, then evaluation |
| `summaries/` | Generated summaries, 124 JSONL files (108 LLM + 16 baseline) |
| `results/` | Metrics, analysis reports and LaTeX tables |
| `xu_et_all_datasets/` | Xu et al.'s released 500-article samples (local JSON) |
| `logs/` | SLURM job logs; created at runtime, not in git |
| [`QA_Evaluation/`](QA_Evaluation/README.md) | Factual consistency via QAFactEval (LERC). Needs legacy AllenNLP + Python 3.7, so it is an isolated sub-project with its own `qa-eval` env |
| [`kiet_phi3_qlora/`](kiet_phi3_qlora/README.md) | QLoRA fine-tune of `Phi-3-mini-4k-instruct`; the trained adapter is included |
| [`kiet_traditional_ml/`](kiet_traditional_ml/README.md) | Extractive summarization by sentence classification (logistic regression, naive Bayes, XGBoost). CPU-only |
| [`transformer/kiet_transformer/`](transformer/kiet_transformer/README.md) | From-scratch encoder–decoder Transformer, one checkpoint per dataset |
| [`transformer/milan_transformer/`](transformer/milan_transformer/README.md) | From-scratch Transformers, encoder–decoder and decoder-only, via `--model` |
| [`transformer/Ali_transformer/`](transformer/Ali_transformer/README.md) | From-scratch Transformer plus a six-variant ablation (`E0`–`E5`) |
| [`transformer/Transformer-Farnaz/`](transformer/Transformer-Farnaz/README.md) | From-scratch Transformer baseline |

The sub-projects have their own dependencies and commands — follow their READMEs rather
than the instructions below. (Ali's and Farnaz's still refer to `Transformer/...`; those
folders now live under `transformer/`.)

---

## Root pipeline

| File | Description |
|---|---|
| `src/main.py` | Task-based entry point — `summarize`, `baselines`, `evaluate`, or `all` |
| `src/registry.py` | `MODEL_CONFIGS` (label → HF id) and the table labels. Dependency-free, also read by `run_experiment.sh` |
| `src/model.py` | `SummarizationModel` and `RunConfig` — loading, quantization, truncation, generation |
| `src/dataset.py` | `DATASET_CONFIGS`, streaming/local loaders, per-dataset field extraction |
| `src/prompts.py` | `PROMPT_CONFIGS` — the P1–P4 prompt templates |
| `src/naming.py` | Single source of truth for output filenames, used by Python **and** the shell scripts |
| `src/evaluator.py` | Metrics (BLEU, ROUGE-L, METEOR, BERTScore, optional QAFactEval) → log + CSV; `Summary:` extraction |
| `src/baselines/` | `lead.py`, `textrank.py`, `tfidf.py` extractive baselines |
| `src/job_time.py` | Estimates each job's SLURM `--time` from historical log durations |
| `src/analysis.py` | Post-hoc analyses of generated summaries → Markdown reports |
| `src/tex_report.py` | Builds the LaTeX tables and figures in `results/tex/` from the evaluation CSVs |
| `scripts/run_summarization.sh` | SLURM worker for ONE (model, quant, prompt, dataset) |
| `scripts/run_baselines.sh` | SLURM worker for the extractive baselines (CPU) |
| `scripts/run_evaluation.sh` | SLURM worker that scores every `.jsonl` |
| `scripts/setup_env.sh` | Idempotent conda env setup (`nlp-project`) |

Models are referenced by a short label from `MODEL_CONFIGS` in
[`src/registry.py`](src/registry.py); add one there and it becomes available to the grid,
the job-status report and the LaTeX tables at once. Quantization is `16bit` (fp16, no
quantization), `8bit`, `4bit` (NF4, recommended for most GPUs), or `None` (an alias for
fp16). The extractive baselines are Lead-1, Lead-3, TextRank, and — under the `TFIDF`
prefix — sumy's LSA summarizer over the TF-IDF matrix.

### Datasets

| Dataset key | Source | Split |
|---|---|---|
| `cnn_dailymail` | `abisee/cnn_dailymail` (HuggingFace, config 3.0.0) | test[:sample] |
| `xsum` | `EdinburghNLP/xsum` (HuggingFace) | test[:sample] |
| `xu_cnndm` | local `xu_et_all_datasets/` (Xu et al.) | fixed 500-article sample |
| `xu_xsum` | local `xu_et_all_datasets/` (Xu et al.) | fixed 500-article sample |

HuggingFace datasets download on first run. The `xu_*` keys read Xu et al.'s local files
verbatim (no shuffling), so every model sees their exact 500 inputs and the results are
directly comparable to theirs. Newsroom and News-QA-Summarization are commented out in
[`src/dataset.py`](src/dataset.py).

---

## Installation

```bash
conda create -n nlp-project python=3.11 -y
conda activate nlp-project
python -m pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.5.1+cu124`, so this path assumes CUDA 12.4. On the bwHPC
cluster use the idempotent helper instead, which pins the exact torch/CUDA build:

```bash
module load devel/miniforge/25.3.1-python-3.12
bash scripts/setup_env.sh      # creates + populates 'nlp-project'
conda activate nlp-project
```

A CUDA-capable GPU is needed for reasonable performance; CPU works but is very slow.

---

## Running the full experiment

`run_experiment.sh` sets up the env, submits one SLURM job per
**model × quant × prompt × dataset** plus a baselines job, then an evaluation job gated on
`afterany` of all of them — so everything runs in parallel and the metrics are computed once
the summaries are ready, even if a combination fails. Jobs are per-dataset to stay within
their time limit, and any combination that already has an output is skipped, so re-running
only fills the gaps.

```bash
# Full grid (Llama+Phi+Qwen2 × cnn_dailymail+xsum × 16/8/4bit × P1/P2/P3), all defaults
./run_experiment.sh

# Restrict any axis — models, datasets, quants, prompts (space-separated)
./run_experiment.sh --models Llama --quants "4bit 8bit" --prompts "P1 P2"
./run_experiment.sh --datasets cnn_dailymail

# Articles per dataset (omit for the full test set); shared seed (default 42)
./run_experiment.sh --sample 500
./run_experiment.sh --seed 7

./run_experiment.sh --no-baselines   # LLM grid only
./run_experiment.sh --no-setup       # skip env setup on repeat runs
./run_experiment.sh --help
```

All jobs in a run share the same `--seed`, so the baselines and every model see the *same*
sampled articles. The seed is **not** part of the output filename, so re-running with a
different seed overwrites that combination — move `summaries/` first to keep both.

Xu et al.'s samples flow through the same pipeline; just select them with
`--datasets "xu_cnndm xu_xsum" --sample 500`. Each file holds exactly 500 records, so
`--sample N` (N<500) caps to the first N for quick tests.

### Single combination, or evaluation only

```bash
python src/main.py --task all --sample 10          # local smoke test, 10 articles
python src/main.py --task baselines
python src/main.py --task evaluate                 # or: sbatch scripts/run_evaluation.sh

# One model/quant/prompt over all datasets
python src/main.py --task summarize --model Llama --quantization_method 4bit --prompt_name P1

# Arbitrary / locally-downloaded model
python src/main.py --task summarize \
    --model_name_or_path ./Llama-3.2-3B-Instruct --model_label Llama \
    --quantization_method 4bit --prompt_name P1
```

`--task evaluate` globs **all** `.jsonl` in the summaries dir. P4 is a one-shot prompt that
draws its in-context example from the train split, so it does not work on the `xu_*` keys
and is not part of the default grid.

---

## Outputs

LLM summaries are written to `summaries/` as
`{model_label}_{prompt}_{quant}_{dataset}_{sample}_summaries.jsonl`
(e.g. `Llama_P1_4bit_cnn_dailymail_full_summaries.jsonl`), baselines as
`{Lead-1|Lead-3|TextRank|TFIDF}_{dataset}_{sample}_summaries.jsonl`. The `{sample}` tag is
`full` for the whole test set, otherwise the article count.

`results/` holds `evaluation.{log,csv}` (the metrics), `qa_evaluation.csv` (from
`QA_Evaluation/`), `tex/` (generated LaTeX), the four analysis reports below,
`Qwen_qualitycheck.txt`, and `5_sample_test/` + `faulty_generation/` from earlier runs.

```bash
python src/analysis.py                 # job_status, token_limit, summary_marker, sanity_check
python src/analysis.py --status-only   # job status only (no torch needed)
python src/tex_report.py               # results/tex/*.tex  (--grid summary|qa|all)
```

---

## Cluster

### GitHub — add SSH keys

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
git clone git@github.com:milannal1m/nlp-project-working-name.git
```

### Job time limits and partitions

Summarization jobs get a **dynamic `--time`**: before submitting each job,
`run_experiment.sh` asks [`src/job_time.py`](src/job_time.py) how long that exact
`(model, quant, prompt, dataset)` took before, parsed from the `Done in … min` lines in
`logs/` (timed-out jobs contribute a rate from their SLURM wall-clock). The rate is
**article-weighted** — total minutes over total articles, so a 5-article smoke test barely
counts against a full 11k-article run — then scaled by this run's article count, plus 5 min
overhead and a **30% margin**, clamped to [30 min, 48 h]. Unmeasured configs fall back to
`36:00:00`.

Each job is routed by that estimate: `gpu_a100_short` if ≤30 min, else `gpu_a100_il`.
Baselines (1 h, CPU) and evaluation (6 h, GPU for BERTScore) use static limits — raise the
evaluation limit before scoring the full grid. Inspect the measured rates without submitting
anything with `python src/job_time.py --report results/job_time_analysis.md`, and check your
partition's max wall time first (`sinfo -p gpu_a100_il -o "%l"`).

### Useful SLURM commands

| Command | Description |
|---|---|
| `squeue --me` | Show your jobs in the queue |
| `watch -n 5 squeue --me` | Live-refresh queue status every 5s |
| `tail -f logs/<jobname>_<jobid>.out` | Stream live log output |
| `scancel <jobid>` | Cancel a specific job |
| `scancel -u <username>` | Cancel all your jobs |
| `scontrol show job <jobid>` | Full job details (node, pending reason, etc.) |
| `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` | Runtime and memory after job ends |
| `sinfo -p gpu_a100_il` | Check partition availability |

For anything else cluster-related see the
[bwHPC wiki](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs#Batch_Jobs:_sbatch).

---

## Status

The main grid is complete: all 124 summary files (3 models × 3 prompts × 3 quants ×
4 datasets, plus 16 baseline runs) are generated and scored into `results/evaluation.csv`,
from which the LaTeX in `results/tex/` is built.

Factual consistency is measured separately in [`QA_Evaluation/`](QA_Evaluation/README.md);
the inline `qafacteval` hook in `src/evaluator.py` is unused outside the cluster, which is
why every row of `evaluation.csv` carries `qafacteval not installed`. P4 is implemented and
runnable but has not been run as part of the grid.

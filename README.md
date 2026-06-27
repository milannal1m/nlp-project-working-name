# News Summarization Benchmark

A news summarization benchmark comparing instruction-tuned LLMs (Llama, Phi, …) across prompts and quantization levels (16/8/4-bit) against extractive baselines. Generates summaries across benchmark datasets in JSONL format and scores them with BLEU, ROUGE-L, METEOR, BERTScore (and optional QAFactEval).

---

## Project Structure

| File | Description |
|------|-------------|
| `src/main.py` | Task-based entry point — `summarize`, `baselines`, `evaluate`, or `all` |
| `src/model.py` | `SummarizationModel`, `RunConfig`, and the `MODEL_CONFIGS` registry (label → HF id) |
| `src/dataset.py` | `DATASET_CONFIGS`, streaming loaders, and per-dataset field extraction |
| `src/prompts.py` | `PROMPT_CONFIGS` — the P1/P2/P3 prompt templates |
| `src/naming.py` | Single source of truth for summary output filenames (used by Python **and** the shell scripts) |
| `src/evaluator.py` | Metrics (BLEU, ROUGE-L, METEOR, BERTScore, optional QAFactEval) → log + CSV |
| `src/baselines/` | `lead.py`, `textrank.py`, `tfidf.py` extractive baselines |
| `run_experiment.sh` | **Main entry point** — orchestrator that runs the full grid in parallel and evaluates (see below) |
| `scripts/run_summarization.sh` | SLURM worker for ONE (model, quant, prompt) combination |
| `scripts/run_baselines.sh` | SLURM worker for the extractive baselines |
| `scripts/run_evaluation.sh` | SLURM worker that scores every `.jsonl` |
| `scripts/setup_env.sh` | Idempotent conda env setup (`nlp-project`) |
| `requirements.txt` | Python dependencies |

LLM summaries are written to `summaries/` as
`{model_label}_{prompt}_{quant}_{dataset}_{sample}_summaries.jsonl`
(e.g. `Llama_P1_4bit_cnn_dailymail_full_summaries.jsonl`, or `..._500_...` for a
500-article run); baselines as
`{Lead-1|Lead-3|TextRank|TFIDF}_{dataset}_{sample}_summaries.jsonl`.
The `{sample}` tag is `full` for the whole test set, otherwise the article count.
Evaluation results land in `results/evaluation.log` and `results/evaluation.csv`.

---

## Datasets

| Dataset key | Source | Split |
|---------|-----------------|-------|
| `cnn_dailymail` | `abisee/cnn_dailymail` (HuggingFace) | test[:sample] |
| `xsum` | `EdinburghNLP/xsum` (HuggingFace) | test[:sample] |
| `xu_cnndm` | local `xu_et_all_datasets/` (Xu et al.) | fixed 500-sample |
| `xu_xsum` | local `xu_et_all_datasets/` (Xu et al.) | fixed 500-sample |
| News QA Summarization | `glnmario/news-qa-summarization` | train[:500] |
(Newsroom doesnt work yet, missing HuggingFace repo)

HuggingFace datasets are downloaded automatically on first run. The `xu_*` keys
read Xu et al.'s released local files (see the Xu et al. section below).

---

## Installation

```bash
conda create -n nlp-project python=3.11 -y
conda activate nlp-project
python -m pip install -r requirements.txt
```

On the bwHPC cluster use the idempotent helper instead, which pins the exact
torch/CUDA build: `bash scripts/setup_env.sh`.

Requires a CUDA-capable GPU for reasonable performance (CPU fallback works but is very slow).

---

## Running the full experiment

`run_experiment.sh` runs the whole benchmark: it sets up the env, then submits
one SLURM job per **Model × Quant × Prompt × Dataset** combination, a baselines
job, and finally a single evaluation job that runs once they have all finished.
Jobs are per-dataset (not per-combo) so each stays within its time limit on the
full test set. Any combination that already has an output is skipped, so
re-running only fills the gaps.

Time limits are tuned for the **full test set**: summarization `36h`/job,
baselines `12h`, evaluation `24h` (BERTScore over ~44 files). Verify your
partition's max wall time first (`sinfo -p gpu_a100_il -o "%l"`) and lower
`--sample` if a cell is at risk of being killed.

```bash
# Full grid (Llama+Phi × cnn_dailymail+xsum × 16bit/8bit/4bit × P1/P2/P3), all defaults
./run_experiment.sh

# Restrict any axis — models, datasets, quants, prompts (space-separated)
./run_experiment.sh --models Llama --quants "4bit 8bit" --prompts "P1 P2"

# Restrict to a single dataset
./run_experiment.sh --datasets cnn_dailymail

# Run the LLM grid only, without the extractive baselines job
./run_experiment.sh --no-baselines

# Limit articles per dataset (omit for the full test set)
./run_experiment.sh --sample 500

# Set the random seed (shared by every job for reproducibility; default 42)
./run_experiment.sh --seed 7

# Skip the env setup step on repeat runs
./run_experiment.sh --no-setup
```

All jobs in a run share the same `--seed`, so the baselines and every model
see the *same* sampled articles. The seed defaults to 42 and is **not** part of
the output filename, so re-running with a different seed overwrites the previous
outputs for that combination — rename/move `summaries/` first if you want to
keep both.

Run `./run_experiment.sh --help` for the full list. To test the pipeline
locally without SLURM, call a single combination directly:
`python src/main.py --task all --sample 10`.

### Reproducing on Xu et al.'s samples

Xu et al. released fixed 500-article samples for CNN/DailyMail and XSum (in
`xu_et_all_datasets/`). They are registered as the dataset keys **`xu_cnndm`** and
**`xu_xsum`**, so they flow through the exact same pipeline (models × quant × prompt,
baselines, evaluation) — just select them with `--datasets`. The articles are read
verbatim from the local files (no shuffling), so every model sees Xu et al.'s exact
500 inputs, making the results directly comparable to theirs.

```bash
# Run ONLY the Xu et al. experiments (full grid + baselines + eval) on their 500 samples
./run_experiment.sh --datasets "xu_cnndm xu_xsum" --sample 500

# Just one of the two
./run_experiment.sh --datasets xu_cnndm --sample 500

# Smoke-test a single combination locally, 3 articles
python src/main.py --task summarize --model Llama --quantization_method 4bit \
    --prompt_name P1 --datasets xu_cnndm --sample 3
```

Outputs are written as `summaries/{model}_{prompt}_{quant}_xu_cnndm_500_summaries.jsonl`
(distinct from the HuggingFace `cnn_dailymail`/`xsum` runs, so the two never collide)
and evaluated into `results/evaluation.{log,csv}` like everything else. Each file is
exactly 500 records, so `--sample 500` uses all of them; `--sample N` (N<500) caps to
the first N for quick tests.

### Running a single combination

The workers can also be used directly (or submitted with `sbatch`):

```bash
# One model/quant/prompt over all datasets
python src/main.py --task summarize --model Llama --quantization_method 4bit --prompt_name P1

# Arbitrary / locally-downloaded model
python src/main.py --task summarize \
    --model_name_or_path ./Llama-3.2-3B-Instruct --model_label Llama \
    --quantization_method 4bit --prompt_name P1

# Baselines / evaluation only
python src/main.py --task baselines
python src/main.py --task evaluate
```

### Models & quantization

Models are referenced by a short label from the registry in
[`src/model.py`](src/model.py) (`MODEL_CONFIGS`); add a model there to make it
available to the grid.

| Quant | Description |
|------|-------------|
| `16bit` | No quantization (fp16) — highest quality, most VRAM |
| `8bit` | 8-bit quantization — middle ground |
| `4bit` | 4-bit NF4 quantization — recommended for most GPUs |

---

## Setup on Cluster

### GitHub — Add SSH Keys

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
git clone git@github.com:milannal1m/nlp-project-working-name.git
```

### Environment

```bash
module load devel/miniforge/25.3.1-python-3.12
bash scripts/setup_env.sh  # creates + populates the 'nlp-project' env (idempotent)
conda activate nlp-project
```

### Running on Cluster

```bash
./run_experiment.sh                 # submits the whole grid + evaluation
./run_experiment.sh --no-setup      # skip env setup on repeat runs
```

The orchestrator submits one job per combination+dataset and a final evaluation
job gated on `afterany` of all of them, so everything runs in parallel and the
metrics are computed once all summaries are ready (even if a combination
fails).

### Useful SLURM Commands

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

For anything else related to the cluster, refer to the wiki: https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs#Batch_Jobs:_sbatch
---


## Remaining Work

1. Run and document the full Phi-3 grid (confirm the HF id in `MODEL_CONFIGS`)
2. Compare summary quality and performance across quantization modes (16 / 8 / 4-bit)
3. Wire up QAFactEval on the cluster (currently optional / skipped if not installed)

# News Summarization — LLMs vs. Baselines

A news-summarization **and evaluation** pipeline. It generates two-sentence
summaries with 6 LLM configs (Llama-3.2-3B and Phi-3-mini × fp16 / 4bit / 8bit)
and 4 simple baselines, then scores them (BLEU, ROUGE-L, METEOR, BERTScore,
SummaC) across CNN/DailyMail and XSum.

See **[WORKFLOW.md](WORKFLOW.md)** for the full architecture and data flow.

---

## Run the whole experiment (cluster)

From the repo root on a bwUniCluster **login node** (connect to the VPN + SSH first):

```bash
bash slurm/run_all.sh                 # pull → set up env → submit every stage
bash slurm/run_all.sh --skip-setup    # env already built (skip the slow install)
```

> Launch with **`bash`, not `sbatch`** — `run_all.sh` is a login-node wrapper
> that submits the `sbatch` jobs for you.

Results land in `results/results.md`, `results/results.csv`, `results/charts/`.

**Change how many articles are used** by editing `SAMPLE` in `pipeline_config.py`
— `None` = the full test split, or set an int (e.g. `500`) for a quick run.

---

## Install (local / manual)

```bash
conda create -n nlp-env python=3.11 -y
conda activate nlp-env
python -m pip install -r requirements.txt
```

On the cluster, `slurm/setup_env.sh` does this for you (plus SummaC, NLTK data,
and model prefetch). A CUDA GPU is needed for the LLMs (CPU works but is very slow).

---

## Datasets

| Dataset | HuggingFace Path | Split |
|---------|-----------------|-------|
| CNN/DailyMail | `abisee/cnn_dailymail` | test (11,490) |
| XSum | `EdinburghNLP/xsum` | test (11,334) |

Downloaded automatically from HuggingFace on first run (streamed, shuffled with
seed 42 so every model scores the same articles).

---

## Quantization modes (LLMs)

| Mode | Description |
|------|-------------|
| `None` | No quantization (fp16) — highest quality, most VRAM |
| `4bit` | 4-bit NF4 — recommended for most GPUs |
| `8bit` | 8-bit — middle ground |

---

## Cluster cheatsheet

### Clone over SSH

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub          # add this key to GitHub
git clone git@github.com:milannal1m/nlp-project-working-name.git
```

### Useful SLURM commands

| Command | Description |
|---|---|
| `squeue --me` | Show your jobs in the queue |
| `watch -n 10 squeue --me` | Live-refresh queue status |
| `tail -f logs/<jobname>_<jobid>.out` | Stream live log output |
| `scancel <jobid>` | Cancel a specific job |
| `scancel -u <username>` | Cancel all your jobs |
| `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` | Runtime/memory after a job ends |
| `sinfo -p gpu_a100_il` | Check partition availability |

Cluster docs: https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs

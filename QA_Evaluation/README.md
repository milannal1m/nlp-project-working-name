# QA Evaluation (Factual Consistency)

Scores generated summaries with **QAFactEval (LERC)**: human-written questions from
`newsqasum_gold.jsonl` are answered *using only the summary*, and LERC rates each answer
against the human reference answer. High score = the summary kept the answer-relevant
information.

## Why this is a separate folder

QAFactEval depends on legacy AllenNLP and **Python 3.7**, which is incompatible with the
main project environment. This is therefore an encapsulated sub-project with its own
conda env (`qa-eval`), its own data generation, and its own results — coupled to the rest
of the repo only through the top-level `src/` package it imports.

## Layout

    QA_Evaluation/
      submit_data_prep_jobs.sh   step 1: generation grid (SLURM)
      run_eval.sh                steps 3+4: LERC + reference metrics (SLURM)
      build_env.sh               creates the Python 3.7 qa-eval env
      scripts/                   all Python entry points
      results/                   master dataset, LERC scores, metrics CSV
      Datasets/                  newsqasum_gold.jsonl (10388 articles)
      Outputs/                   one .jsonl per config, from step 1
      Outputs_full/              earlier full-dataset run, see "Data" below
      QAFactEval/ qaeval/ facebook/   offline model weights, not in git

Scripts anchor their paths to this layout, not to the working directory — they behave the
same from `QA_Evaluation/` or the repo root.

---

## Manual Placement of Dependencies

The three core dependency folders (`QAFactEval`, `qaeval`, and `facebook`) are **not tracked in git**. Attempting to push nested git repositories and >100MB model weights directly to GitHub causes upload failures and broken links. 

Until this is integrated into the cluster pipeline, these dependencies must be prepared locally and placed on the cluster by hand.

---

### What Needs to Be Prepared

Before running evaluations, you need three specific components ready on your machine:
1. **`QAFactEval/`**: The main evaluation package from Salesforce, plus its pre-trained scoring models (`models/`).
2. **`qaeval/`**: The base QA framework that QAFactEval depends on.
3. **`facebook/bart-large`**: Question generation weights required by QAFactEval during framework initialization.

---

### Automated Preparation Script

You can run the following automated commands locally to download, build, and structure all three dependencies automatically:

```bash
# 1. Clone QAFactEval and download its model weights
git clone [https://github.com/salesforce/QAFactEval.git](https://github.com/salesforce/QAFactEval.git)
cd QAFactEval && bash download_models.sh && cd ..

# 2. Clone the underlying qaeval framework
git clone [https://github.com/danieldeutsch/qaeval.git](https://github.com/danieldeutsch/qaeval.git)

# 3. Clone BART-large and nest it inside a facebook/ folder
git lfs install
git clone [https://huggingface.co/facebook/bart-large](https://huggingface.co/facebook/bart-large)
mkdir -p facebook && mv bart-large facebook/
```

#### What the Scripts actually do:
* **Step 1 (`QAFactEval`):** Clones the official repository and runs `download_models.sh`. This script fetches Salesforce's neural network weights (LERC, Answering models, and QuIP) and extracts them directly into `QAFactEval/models/`.
* **Step 2 (`qaeval`):** Downloads the legacy `qaeval` repository that `QAFactEval` inherits from.
* **Step 3 (`facebook/bart-large`):** Uses Git LFS to pull the full `bart-large` model weights from Hugging Face, then creates a `facebook/` directory and moves `bart-large` inside it so that path lookups resolve correctly.

---

### Transfer to Cluster via SCP

Once the script finishes on your local machine, run these commands to transfer the three prepared directories into `QA_Evaluation/` on your cluster:

```bash
scp -r QAFactEval <user>@<cluster>:/path/to/project/QA_Evaluation/
scp -r qaeval     <user>@<cluster>:/path/to/project/QA_Evaluation/
scp -r facebook   <user>@<cluster>:/path/to/project/QA_Evaluation/
```

---

### Expected Remote Directory Structure

After transferring, your cluster directory layout under `QA_Evaluation/` must look like this:

```text
QA_Evaluation/
├── QAFactEval/
│   └── models/          # Populated by download_models.sh
├── qaeval/              # Base framework
└── facebook/
    └── bart-large/      # Hugging Face weights
```

`scripts/qa_evaluator.py` raises `FileNotFoundError` at import if `QAFactEval/` or `qaeval/` is missing, and fails later with a `FileNotFoundError` if the weights inside them are missing.

## Setup

Once per machine:

    bash build_env.sh          # builds the qa-eval env (Python 3.7, torch 1.12+cu113)

The reference metrics in step 4 additionally need the main `nlp-project` env.

## Pipeline

**1. Generate summaries** (one SLURM job per model × quant × prompt), from the repo root:

    bash QA_Evaluation/submit_data_prep_jobs.sh --sample 500    # omit --sample for all

Per-job `--time` comes from `src/job_time.py`, scaled by the article count. Resumable:
already-generated `article_id`s are skipped, so a run can be topped up later.
→ `Outputs/<model>_<quant>_<prompt>.jsonl`

**2. Merge into one matrix:**

    python QA_Evaluation/scripts/summaries_merger.py

One row per article, one column per config, plus the four *foundation* fields
(`article_id`, `source_article`, `human_questions`, `human_answers`). Their names matter:
the evaluator scores every *other* key in a row as a candidate summary. Only articles
covered by **every** config are kept, so columns stay comparable (`--union` to disable).
→ `results/master_evaluation_dataset.jsonl`

**3 + 4. Score it** — one GPU job that does both halves:

    cd QA_Evaluation && sbatch run_eval.sh

It runs LERC in `qa-eval`, then switches to `nlp-project` for BLEU / ROUGE-L / METEOR /
BERTScore. The env switch is required: those come from `src.evaluator.Evaluator`, which
needs `evaluate` and `bert_score` — absent from the Python 3.7 env.
→ `results/final_evaluation_results.jsonl` and `results/qa_evaluation.csv`

## Output: results/qa_evaluation.csv

One row per config: `config`, `model`, `quant`, `prompt`, `n_articles`, `bleu`, `rougeL`,
`meteor`, `bertscore_f1_raw(_std)`, `bertscore_f1_scaled(_std)`, `lerc_mean`, `lerc_std`,
`lerc_n`, `lerc_zeros`, `error`.

Use `bertscore_f1_scaled` — the raw variant sits near 0.85 even for poor matches. `error`
is empty on success, otherwise `no LERC column`, `no scoreable rows`, or an exception; a
failing config is recorded and skipped so the rest still finish.

**Caveat on `lerc_zeros`.** A LERC of exactly 0.0 means QAFactEval judged *every* question
for that article unanswerable from the summary — which happens far more often than the
summaries deserve, because its answerability classifier is calibrated for the opposite
direction (questions generated from the summary, answered against the source). In the
500-article run this affects ~27 % of cells, and in ~61 % of those at least one gold
answer is verbatim in the summary. `lerc_mean` includes the zeros, so report `lerc_zeros`
next to it. Inspect cases with:

    python QA_Evaluation/scripts/show_zero_scores.py -n 10

## Data: Outputs vs Outputs_full

`Outputs_full/` holds an earlier run over the **complete** gold set (10388 articles), but
only for 8 Phi configs. Extrapolated to the full grid this would not have finished before
the project deadline, so we switched to the **first 500 articles per config**, which is
what `Outputs/` and all current results contain.

## Scripts

All in `scripts/`; "any" = standard library only, runs on a login node.

| script | purpose | env |
| --- | --- | --- |
| `data_preparation_pipeline.py` | generate summaries for one config | `nlp-project` |
| `summaries_merger.py` | `Outputs/` → master matrix | any |
| `run_qa.py` | LERC scoring (`--max-articles N` for a smoke test) | `qa-eval` |
| `qa_evaluator.py` | QAFactEval wrapper, imported by `run_qa.py` | `qa-eval` |
| `evaluate_all_metrics.py` | reference metrics + LERC → CSV | `nlp-project` |
| `aggregate_lerc.py` | LERC mean/std/min/max per config, to stdout | any |
| `show_zero_scores.py` | print zero-scored summaries with their QA pairs | any |

**Known issue:** the CPU path fails with *"Device index must not be negative"* —
without a GPU `qa_evaluator.py` passes `cuda_device = -1`, which QAFactEval turns into
`torch.device("cuda", -1)` instead of treating it as CPU. Use a GPU job.

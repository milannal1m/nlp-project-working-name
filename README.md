# News Summarization with LLaMA

A news summarization pipeline using LLaMA-family models with support for 4-bit and 8-bit quantization. Generates summaries across multiple benchmark datasets and saves results in JSONL format for downstream evaluation.

---

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point — parses arguments, loads the model and datasets, runs the pipeline |
| `model.py` | `SummarizationModel` class and `RunConfig` dataclass — handles model/tokenizer loading and inference |
| `dataset.py` | Dataset configs, loading logic, and field extraction for each dataset |
| `run_summarization.sh` | SLURM job script for running on bwUniCluster 3.0 |
| `requirements.txt` | Python dependencies |

Output summaries are written to `./summaries/` as `Llama_{quantization}_{dataset}_summaries.jsonl`.

---

## Datasets

| Dataset | HuggingFace Path | Split |
|---------|-----------------|-------|
| CNN/DailyMail | `abisee/cnn_dailymail` | test[:500] |
| XSum | `EdinburghNLP/xsum` | test[:500] |
| News QA Summarization | `glnmario/news-qa-summarization` | train[:500] |
(Newsroom doesnt work yet, missing HuggingFace repo)

Datasets are downloaded automatically from HuggingFace on first run.

---

## Installation

```bash
conda create -n nlp-env python
conda activate nlp-env
python -m pip install -r requirements.txt
```

Requires a CUDA-capable GPU for reasonable performance (CPU fallback works but is very slow).

---

## Running

### Option 1 — HuggingFace model ID (downloaded automatically)

```bash
python main.py --model_name_or_path unsloth/Llama-3.2-3B-Instruct --quantization_method 4bit
```

### Option 2 — Locally downloaded model

```bash
huggingface-cli download unsloth/Llama-3.2-3B-Instruct --local-dir ./Llama-3.2-3B-Instruct

python main.py --model_name_or_path ./Llama-3.2-3B-Instruct --quantization_method 4bit
```

### Quantization options

| Flag | Description |
|------|-------------|
| `None` | No quantization (fp16) — highest quality, most VRAM |
| `4bit` | 4-bit NF4 quantization — recommended for most GPUs |
| `8bit` | 8-bit quantization — middle ground |

---


## Remaining Work

1. Implement evaluation metrics (ROUGE, BERTScore) as specified in `NLP_Paper.pdf`
2. Run and document phi-3 model experiments
3. Compare summary quality and performance across quantization modes (None / 4bit / 8bit)

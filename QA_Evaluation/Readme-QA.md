# QAFactEval Pipeline

This directory contains the factual consistency evaluation module using QAFactEval. 

**Critical Architecture Note:** QAFactEval is a legacy 2020 module. It strictly requires Python 3.8 and PyTorch 1.6.0. **Do not** attempt to run this in the main `nlp-env` used for LLaMA 3.2 generation. The environment is explicitly isolated to prevent dependency resolution failures.

## Directory Structure
* `qa_evaluator.py`: Core execution class. Handles JSONL parsing, sliding-window chunk matching, and model initialization.
* `run_qa.py`: Pipeline entry point. Computes and outputs final Mean/Std LERC scores.
* `setup_eval.sh`: One-time cluster setup script. Bypasses the broken 3.3kb PyPI package via direct GitHub source cloning and routes around deprecated AWS S3 links by pulling base vocabulary directly from HuggingFace.
* `run_eval.sh`: SLURM batch script (`gpu_a100_il` partition).
* `newsqasum_gold.jsonl`: Ground-truth dataset for question alignment.

## Deployment Protocol
### 1. Environment Initialization
Run the setup script manually on the login node. This creates the isolated Conda environment (`qa-eval-env`) and downloads the necessary ~4.5GB of bypass files and model weights.
```bash
cd QA_Evaluation
bash setup_eval.sh
```
### Data Alignment
There is currently no text overlap between `newsqasum_gold.jsonl` and the existing CNN/DailyMail generated summaries. Factual consistency evaluation cannot proceed until the LLaMA model processes the matching NewsQA source articles.
### Execution
Once we find overlapping articles, we submit the evaluation job to the cluster
```bash
sbatch run_eval.sh
```
Final LERC metrics will be printed out as a log file.
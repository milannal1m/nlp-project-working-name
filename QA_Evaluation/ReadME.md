# QA Factuality Evaluation Pipeline 

This module executes a dual-context QA-based factual consistency evaluation for Small Language Models (SLMs) across various quantization levels and prompt structures. 

Because modern SLM generation (BitsAndBytes, PyTorch 2.0+) and the QAFactEval library have conflicting dependencies, this pipeline is strictly decoupled into two phases running in separate environments.

## Phase 1: Data Preparation & Generation (Cluster)
We use a Slurm Job Array to run 18 different SLM configurations (2 Models × 3 Quants × 3 Prompts) in parallel. This phase reads the source CNN articles from `newsqasum_gold.jsonl` and generates the corresponding summaries.

**Prerequisites:**
Ensure you are using the modern environment (`nlp-project`) containing `transformers`, `bitsandbytes`, and `accelerate`.

**Execution:**
From the root project directory, submit the batch script:
\`\`\`bash
sbatch run_data_prep.sh
\`\`\`
This will spawn 18 parallel jobs. Output summaries will be saved as individual `.jsonl` files in `QA_Evaluation/temp_outputs/`.

## Phase 2: Data Merging (Local/Login Node)
*Script pending.*
Once the cluster completes all 18 generation jobs, we run a rapid merging script. This script aggregates the 18 temporary `.jsonl` files into a single `master_evaluation_dataset.jsonl`, mapping every generated summary to its specific `article_id`.

## Phase 3: QA Evaluation (Local/Login Node)
*Script pending.*
With the master dataset compiled, we transition to the legacy environment (`qa-eval-env`) to run the factual consistency scoring.

**Methodology:**
1. **Source Pass (Context 1):** The QA model generates answers based entirely on the source article to establish reading comprehension competence. Questions failing to match the human gold-standard are filtered out.
2. **Summary Pass (Context 2):** The QA model answers the filtered questions using the generated SLM summaries.
3. **Scoring:** The Summary Pass answers are compared directly against the Source Pass answers via LERC to calculate exact information loss and factual decay caused by the summarization process.
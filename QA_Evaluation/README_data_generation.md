# Data Generation Pipeline

This directory contains the highly-optimized, cluster-ready data generation and compilation pipeline for our reference-free text summarization experiments. 

Because we evaluate 18 different LLM configurations (Models x Quantization x Prompts) over the dataset, this pipeline utilizes a two-phase **Map-Reduce architecture** to prevent cluster bottlenecks and race conditions.

---

## Execution Workflow

### Phase 1: Parallel Generation (The Cluster Run)
Do not run the Python generation script manually. Submit the job array to the SLURM cluster to process all 18 variations simultaneously on independent GPU nodes.

    sbatch QA_Evaluation/run_data_prep.sh

**What happens under the hood:**
* SLURM spawns 18 clones of the generation script.
* Each node runs a unique combination of Model (Llama-3.2-3B, Phi-3-Mini), Quantization (None, 8bit, 4bit), and Prompt (P1, P2, P3).
* To ensure scientific consistency with the reference-based baselines, the script natively calls `src.datasets.strip_dateline` to clean journalistic metadata from the input text before generation.
* To prevent hallucination penalties in downstream scoring, the script passes all outputs through `src.evaluator.extract_summary` to strictly strip away Chain-of-Thought (CoT) reasoning.
* **Output:** Each node drops a standalone JSONL preservation file into the Outputs directory containing the full article context alongside the unique summary text.

### Phase 2: Wait for Completion
Monitor your cluster queue:

    squeue -u <your_username>

You must wait until all 18 tasks have completed and 18 distinct files are present in the temporary outputs directory before proceeding to Phase 3. Running the merger early will corrupt the master matrix.

### Phase 3: Matrix Compilation
Once the cluster is completely quiet, run the merger script locally on the login node. 

    python QA_Evaluation/summaries_merger.py

**What happens under the hood:**
* The script isolates the MD5 article_id hashes to stitch the 18 generated summaries perfectly side-by-side, stripping out the heavy text payloads (like original stories and QA pairs) to reduce file size and overhead.
* It performs a structural audit to ensure exactly 19 columns exist (1 Primary Key + 18 Variations).
* **Output:** A single, lean **master_evaluation_dataset.jsonl** file perfectly optimized for the qa evaluation of the data.
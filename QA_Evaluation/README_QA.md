# QA Evaluation Pipeline (Factual Consistency)

This directory contains the isolated scoring environment used to evaluate the factual consistency of generated summaries using the QAFactEval (LERC) metric. 

Because QAFactEval relies on legacy AllenNLP frameworks (Python 3.7) and massive offline model weights, this pipeline is intentionally decoupled from the primary data generation environment.

---

## Environment Setup (Do This First)

You must build the isolated Python 3.7 sandbox before running any evaluations. The provided build script utilizes pre-compiled `conda-forge` binaries to bypass legacy C++ compilation errors.

Run this command once on the machine/cluster where the evaluation will take place:

    bash build_env.sh

This will create a new environment named `qa-eval` containing the exact PyTorch 1.12 + CUDA 11.3 bindings required by the pipeline.

---

## 📦 Step 0: The Offline Weights (Critical Pre-Requisite)

Because GitHub blocks files larger than 100MB, the pre-trained QAFactEval PyTorch weights are hosted externally on Cloudstore. The pipeline will immediately crash with a `FileNotFoundError` if these are missing.

1. Download the heavy `models` and `facebook` folders from the shared Cloudstore link.
2. Use `scp` (Secure Copy Protocol) from your local terminal to push these heavy folders to their specific target directories on the cluster:

       scp -r facebook <username>@<cluster_address>:/path/to/project/QA_Evaluation/
       scp -r models <username>@<cluster_address>:/path/to/project/QA_Evaluation/QAFactEval/

---

## Execution: Local Run (Mac/PC Testing)

If you need to run a fast, downscaled CPU test on your local machine using a small test dataset:

1. Move the downloaded `facebook` folder into `QA_Evaluation/`.
2. Move the downloaded `models` folder into `QA_Evaluation/QAFactEval/`.
3. Ensure your local test `master_evaluation_dataset.jsonl` is in the `QA_Evaluation` directory.
4. Activate the legacy environment:
   
       conda activate qa-eval

5. Execute the pipeline. The script will auto-detect the lack of a GPU and scale down the batch sizes automatically:

       python run_qa.py

---

## Execution: Cluster Run (Production)

To run the full evaluation across the 18 summary variations, you must submit it as an A100 GPU job on the SLURM cluster.

**Prerequisite:** Ensure the Data Preparation Pipeline has finished and the compiled `master_evaluation_dataset.jsonl` is already sitting in your cluster's `QA_Evaluation` directory.

SSH into the cluster, navigate to the `QA_Evaluation` directory, and submit the SLURM batch script:

    sbatch run_eval.sh

**What happens under the hood:**
* The cluster job automatically activates the legacy `qa-eval` environment.
* The script detects the A100 GPU and automatically scales up the batch sizes for maximum throughput.
* It calculates the LERC scores to determine how well each summary retained the critical information required to answer the human-authored questions.
* **Output:** A lean `final_evaluation_results.jsonl` file containing the `article_id` mapped directly to the calculated scores for every model variation.
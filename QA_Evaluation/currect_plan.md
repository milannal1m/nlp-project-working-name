# Current Plan: QAFactEval Data Alignment Strategy

## 1. Proposed Solution: Targeted Subset Inference
Rather than trying to force mismatched datasets together, we will invert the workflow: **generate new summaries for the articles attached to the human questions we already have.**

This isolates the QA evaluation into a controlled sub-experiment to measure factual decay across quantization levels, utilizing our existing inference architecture (`dataset.py` and `model.py`).

### Execution Steps:
1. **Targeted Generation:** Write a localized script (`generate_qa_summaries.py`) to run our 6 model configurations (Llama 3.2 3B-Instruct & Phi-3 Mini across full-precision, 8-bit, and 4-bit) directly on the source articles found within the local `newsqasum.jsonl` file.
2. **Schema Alignment:** Format the output of this inference run to perfectly match our required schema (`news`, `reference_summary`, `generated_summary`), appending the human-authored questions to each record.
3. **Streamlined Evaluation:** Refactor `qa_evaluator.py`. Because our newly generated summaries will map 1:1 with the original `newsqasum` dataset, we can remove computationally heavy sliding-window searches and use a direct line-by-line `zip()` mapping to feed the data into `QAFactEval`.

## 4. Methodological Validation
To ensure this subset experiment is statistically valid and comparable to our main benchmarking results:
* We will run our standard lexical baselines (BLEU, ROUGE-L) and NLI metrics (SummaC, BERTScore) on this new subset.
* Demonstrating that the lexical performance on the `newsqasum` subset mirrors the performance on the main CNN/DailyMail test split will justify using this subset specifically for probing deeper factual consistency.
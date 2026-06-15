# 1. Setup & Methods (the facts behind the numbers)

## The two datasets

| Property | CNN/DailyMail | XSum |
|---|---:|---:|
| Articles (full test split) | 11,490 | 11,334 |
| Avg article length | 694.6 words | 385.4 words |
| Avg reference length | 52.9 words | 21.7 words |
| Avg reference sentences | 3.86 | 1.02 |
| Reference words **not in the source** | 14.6% (extractive) | 35.7% (abstractive) |

- Same articles for every system: shuffled with `seed=42`, so comparisons are fair ([`dataset.py`](../dataset.py)).
- CNN datelines (e.g. `"LONDON (CNN) -- "`) are stripped; XSum is not.

## The 10 systems

- **4 baselines** (extractive, copy source sentences): Lead-1, Lead-3 (first n sentences), TextRank, TF-IDF — all output 2 sentences ([`baseline_*.py`](../baseline_lead.py)).
- **6 LLMs**: {Llama-3.2-3B-Instruct, Phi-3-mini-4k-instruct} × {fp16, 4-bit NF4, 8-bit} ([`model.py`](../model.py)).
- Prompt: `News: {article}\nSummarize the news in two sentences. Summary:`
- Decoding: greedy (`do_sample=False`), max 150 new tokens, input capped at 2048 tokens.
- ⚠️ **No chat template** is applied — the instruct models get a raw prompt. This is the root cause of Phi-3's problems (see [03](03_per_config_analysis.md)).

## The 6 metrics ([`evaluator.py`](../evaluator.py))

| Metric | Compared to | Measures |
|---|---|---|
| BLEU, ROUGE-L, METEOR | reference summary | word/sequence **overlap** |
| BERTScore-F1 | reference summary | **meaning** (embeddings) |
| SummaC | source article | factual consistency (NLI) |
| QAFactEval | source article | factual consistency (QA) — **not installed → empty** |

- Overlap metrics reward matching the reference's exact words → they depend heavily on reference style.
- BERTScore is the only metric that judges meaning, not surface words.
- SummaC/QAFactEval ignore the reference and check faithfulness to the article.

## Pipeline

`config → generate → summaries/*.jsonl → evaluate → metrics/*.json → aggregate → results.{md,csv}, charts/`

- Four SLURM stages, one launch (`bash slurm/run_all.sh`); `afterany` deps so one failure never blocks the report.
- Each metric runs in its own `try/except` → a failure leaves that cell empty, not the whole run.
- Every score is averaged over the **whole** split, so small differences are stable.

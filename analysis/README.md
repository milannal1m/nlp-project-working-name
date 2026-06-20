# Evaluation Analysis — News Summarization (LLMs vs. Baselines)

Full test splits: **CNN/DailyMail = 11,490**, **XSum = 11,334**. 10 systems × 2 datasets × 6 metrics. All numbers from [`results/results.csv`](../results/results.csv).

Read in order:
- [01_setup_and_methods.md](01_setup_and_methods.md) — what we ran and how (background facts).
- [02_why_xsum_scores_are_lower.md](02_why_xsum_scores_are_lower.md) — the XSum question.
- [03_per_config_analysis.md](03_per_config_analysis.md) — why each system scores what it scores.
- [04_metric_reliability_caveats.md](04_metric_reliability_caveats.md) — which metrics to trust.
- [05_cluster_analysis.md](05_cluster_analysis.md) — data-driven clustering of the 20 configs into 4 meaningful groups, with proof (run `python cluster_analysis.py`).

Every point below follows the same thread: **Question → Answer → Reason → Proof.**

---

## The red thread (5 questions)

**Q1 — Why are XSum scores lower than CNN/DailyMail?**
- **Answer:** Because of how XSum *references* are written, not because the summaries are worse.
- **Reason:** XSum references are 1 short, rewritten sentence; BLEU/ROUGE/METEOR only reward shared words, so they are capped low. CNN references are 3–4 near-copied sentences, so overlap is easy.
- **Proof:** XSum ref = 21.7 words, 1.0 sentence, **35.7% of its words are not in the article**. CNN ref = 52.9 words, 3.9 sentences, only 14.6% novel. → BLEU: CNN 0.115 vs XSum 0.026.

**Q2 — Is XSum actually summarized worse?**
- **Answer:** No.
- **Reason:** A *meaning* metric removes the word-overlap penalty.
- **Proof:** BERTScore (Llama) is **higher** on XSum (0.8736) than CNN (0.8729).

**Q3 — Which system is best?**
- **Answer:** Llama-3.2-3B-Instruct.
- **Reason:** Best LLM on every reference metric, best BERTScore overall; clean ~2-sentence output.
- **Proof:** XSum: best BLEU 0.0256, ROUGE-L 0.1646, METEOR 0.2762, BERTScore 0.8736. On CNN a trivial Lead-3 still wins *word-overlap* metrics (extractive references).

**Q4 — Does 4-bit/8-bit quantization hurt quality?**
- **Answer:** No measurable loss.
- **Reason:** Same model, three precisions → same scores.
- **Proof:** Llama XSum ROUGE-L = 0.1639 / 0.1646 / 0.1631 (fp16 / 4bit / 8bit); BERTScore 0.8736 / 0.8735 / 0.8734.

**Q5 — Can we trust the metrics?**
- **Answer:** Trust BERTScore for meaning; treat lexical scores as reference-style-relative; treat SummaC as coarse; ignore QAFactEval.
- **Reason:** The metrics disagree, and SummaC contradicts itself.
- **Proof:** Two *verbatim-copy* baselines on CNN score SummaC 0.89 (Lead-3) vs 0.10 (TextRank). QAFactEval = empty everywhere (not installed).

---

## Headline numbers (full split)

### CNN/DailyMail (11,490)

| Model | BLEU | ROUGE-L | METEOR | BERTScore-F1 | SummaC | Gen Len |
|-------|------:|------:|------:|------:|------:|--------:|
| Lead-1 | 0.0365 | 0.1831 | 0.1682 | 0.8593 | 0.7739 | 25.7 |
| **Lead-3** | **0.1150** | **0.2428** | **0.3854** | 0.8691 | 0.8914 | 82.2 |
| TextRank | 0.0597 | 0.1808 | 0.2540 | 0.8488 | 0.1034 | 78.3 |
| TFIDF | 0.0648 | 0.1733 | 0.2163 | 0.8504 | 0.8476 | 47.6 |
| Llama_None | 0.0836 | 0.2389 | 0.3381 | 0.8729 | 0.0375 | 73.4 |
| Llama_4bit | 0.0813 | 0.2359 | 0.3289 | 0.8717 | 0.0461 | 71.7 |
| Llama_8bit | 0.0842 | 0.2392 | 0.3373 | **0.8731** | 0.0618 | 72.6 |
| Phi-3_None | 0.0485 | 0.1846 | 0.3008 | 0.8521 | 0.7257 | 103.0 |
| Phi-3_4bit | 0.0455 | 0.1836 | 0.2880 | 0.8510 | 0.2898 | 98.4 |
| Phi-3_8bit | 0.0500 | 0.1854 | 0.3035 | 0.8521 | 0.9718 | 103.9 |

### XSum (11,334)

| Model | BLEU | ROUGE-L | METEOR | BERTScore-F1 | SummaC | Gen Len |
|-------|------:|------:|------:|------:|------:|--------:|
| Lead-1 | 0.0070 | 0.1178 | 0.1312 | 0.8554 | 0.2265 | 24.3 |
| Lead-3 | 0.0077 | 0.1155 | 0.2089 | 0.8546 | 0.2276 | 69.4 |
| TextRank | 0.0092 | 0.1147 | 0.1974 | 0.8497 | 0.2245 | 65.0 |
| TFIDF | 0.0080 | 0.1151 | 0.1812 | 0.8513 | 0.2267 | 44.2 |
| **Llama_None** | **0.0256** | 0.1639 | **0.2762** | **0.8736** | **0.8417** | 60.4 |
| Llama_4bit | 0.0251 | **0.1646** | 0.2721 | 0.8735 | 0.7845 | 57.7 |
| Llama_8bit | 0.0250 | 0.1631 | 0.2751 | 0.8734 | 0.7028 | 60.3 |
| Phi-3_None | 0.0121 | 0.1131 | 0.2257 | 0.8527 | 0.4035 | 102.9 |
| Phi-3_4bit | 0.0122 | 0.1189 | 0.2261 | 0.8537 | 0.5644 | 97.3 |
| Phi-3_8bit | 0.0124 | 0.1136 | 0.2267 | 0.8526 | 0.2091 | 103.3 |

Bold = best in column for that dataset.

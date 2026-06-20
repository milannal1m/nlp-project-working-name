---
marp: true
title: Summarization Results — What We Found and Why
description: 3-slide deck on the evaluation results and their causes
paginate: true
---

<!--
3-slide deck for an NLP course / technical-peer audience.
Focus: the results and *why* they look that way, with proof.
Render with Marp (`marp presentation.md`) or read as plain markdown.
All numbers from results/results.csv and results/cluster_stats.json.
Figures: results/charts/cluster_*.png (run `python cluster_analysis.py`).
-->

# 20 configs, but only **4 behaviors**

**Setup.** 10 systems × 2 datasets × 5 metrics
- Systems: 4 extractive baselines (Lead-1, Lead-3, TextRank, TFIDF) + **Llama-3.2-3B** & **Phi-3-mini**, each at {fp16, 8-bit, 4-bit}
- Datasets: **CNN/DM** (extractive references) vs **XSum** (abstractive references)
- Metrics: lexical (BLEU/ROUGE-L/METEOR), semantic (BERTScore), factuality (SummaC)

**Result.** Unsupervised clustering (Ward + K-Means, silhouette **0.61–0.69**) → **4 clusters**:
**Abstractive-LLM** (Llama×3) · **Prompt-degraded-LLM** (Phi-3×3) · **Positional-extractive** (Lead-1/3) · **Salience-extractive** (TextRank/TFIDF)

![h:330](../results/charts/cluster_dendrogram_xsum.png)

<!--
SAY: Twenty configs, four behaviors. The dendrogram already previews the three "why"s:
each precision triplet merges at ~0 distance (quantization is invisible), model family
dominates, and the deepest split is Llama vs everything — the abstractive/extractive divide.
-->

---

# Why: **reference style × output length** explain 86%

- **Two axes (PCA):** PC1 (49%) = reference overlap, PC2 (37%) = verbosity → **86%** of all variance.
- **The XSum "myth":** XSum *looks* worse only because its refs are 1 sentence, **35.7% novel words** → lexical overlap is floored.
  *Proof:* Llama BLEU **0.084 → 0.026** (CNN→XSum), but BERTScore **0.873 → 0.874** (parity). Meaning is fine; the metric is the problem.
- **CNN lead bias:** trivial Lead-3 *ties* Llama on ROUGE-L (**0.243 vs 0.239**) → Lead-3 clusters *with* Llama on CNN, drops out on XSum.
- **Quantization is invisible:** precision variants are **12–33× closer** to each other than to anything else; BERTScore spread ≤ **0.0014**.

![h:300](../results/charts/cluster_pca_combined.png)

<!--
SAY: Walk PC1/PC2. Then the headline insight: "XSum isn't summarized worse — its references are
unmatchable by word overlap." Then lead bias. Then: fp16/8-bit/4-bit are statistically the same model —
the Llama points are stacked on top of each other in the PCA.
-->

---

# Real vs **artifact** → what to conclude

- **Phi-3's cluster is a harness bug, not weak ability:** **90–93%** of outputs leak the instruction template (`Document:`, `## Your task`), ~100 words vs Llama ~60–73; Llama leak rate **0%**.
  → its ranking is **invalid** pending a re-run with chat template + stop tokens.
- **SummaC is unreliable:** swings **0.68** across one model's precision variants; two verbatim-copy baselines score 0.89 vs 0.10. QAFactEval = null (not installed).
  → rank with **BERTScore**; treat SummaC as coarse.

**Takeaways**
1. Best system = **Llama-3.2-3B, any precision → ship 4-bit (free).**
2. **Never compare lexical scores across datasets** — only within.
3. Caveats: small *n* (10/dataset); BERTScore saturates at 0.85–0.87.

<!--
SAY: Owning the Phi-3 bug reads as rigor, not weakness. Land on the three takeaways.
BACKUP slides for Q&A: results/results.md tables + cluster_dendrogram_cnn_dailymail.png.
Likely Qs: why trust clustering at n=10 (read dendrogram, not just k; both algorithms agree);
why drop SummaC from features (robustness check: Rand 0.93–1.0 with vs without);
BERTScore saturation (directionally valid); full split vs sample (metrics full 11k+, leak rate on 500).
-->

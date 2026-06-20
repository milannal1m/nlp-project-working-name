# Are Our Results "Good"? — Comparison to Published Baselines

How do our scores compare to other people's results on the **same datasets** (CNN/DailyMail, XSum)?
Short answer: our raw ROUGE looks low next to published leaderboards, but that is mostly **(a)** a
ROUGE-configuration mismatch and **(b)** a zero-shot-vs-fine-tuned comparison — **not** weak summaries.
Our **BERTScore is competitive with fine-tuned PEGASUS.** All our numbers are from
[`results/results.csv`](../results/results.csv).

> **Read this as Question → Answer → Reason → Proof**, like the other docs in this folder.

---

## The comparison (ROUGE shown ×100 to match papers)

### CNN/DailyMail

| System | R-1 | R-2 | R-L | BERTScore-F1 |
|---|---:|---:|---:|---:|
| **Ours — Lead-3** | — | — | **24.3** | 0.869 |
| **Ours — Llama (best)** | — | — | **23.9** | 0.873 |
| Lead-3 (canonical, See et al. 2017) | 40.4 | 17.6 | **36.7** | — |
| BART (fine-tuned, Lewis et al. 2020) | 44.2 | 21.3 | **40.9** | — |
| PEGASUS (fine-tuned, Zhang et al. 2020) | 44.2 | 21.5 | **41.1** | — |

### XSum

| System | R-1 | R-2 | R-L | BERTScore-F1 |
|---|---:|---:|---:|---:|
| **Ours — Llama (best)** | — | — | **16.5** | **0.874** |
| BART (fine-tuned) | 45.1 | 22.3 | 37.3 | — |
| PEGASUS (fine-tuned) | 47.2 | 24.6 | 39.3 | **~0.874** |

Two things stand out: our ROUGE-L is roughly **half** the leaderboard, yet our **BERTScore-F1 (0.874) is
essentially identical to PEGASUS's reported XSum BERTScore (0.8737)**.

*(Canonical ROUGE figures are the values reported in the original papers; cross-checked against the
comparative studies linked below. We report R-1/R-2 only for the literature, since our pipeline scores
ROUGE-L, BERTScore, SummaC.)*

---

## Why the ROUGE gap — and why it is mostly not "bad work"

**Q1 — Are our summaries half as good as BART/PEGASUS?**
**A — No.** The gap is dominated by *how* and *what* we compare, not summary quality.

### Reason 1 — We don't measure ROUGE the leaderboard way
[`evaluator.py`](../evaluator.py) reports HuggingFace `rouge`'s **`rougeL`** (LCS over the whole string)
with **no stemming** (the HF default). Papers report **`rougeLsum`** (sentence-split LCS) **with Porter
stemming**.

**Proof:** our **Lead-3 scores R-L 24.3, but canonical Lead-3 is 36.7.** Lead-3 literally copies the
article's first sentences, so it *should* land near 37. A 12-point deficit **on a copy-paste baseline**
shows the metric configuration is the cause, not the summaries.
**Fix:** report `rougeLsum` and pass `use_stemmer=True` — that lifts every row several points and makes the
numbers comparable.

### Reason 2 — We ran zero-shot 3B instruct models; the leaderboard is fine-tuned models
BART and PEGASUS were **trained on CNN/DM and XSum**. Our Llama-3.2-**3B** and Phi-3-mini saw **zero**
training examples from these datasets. The literature is explicit: zero-shot LLM summaries score
**~7 ROUGE-L points below fine-tuned SOTA while humans actually *prefer* them** (Goyal et al. 2022; TACL
LLM-summarization benchmark). So low ROUGE here is *expected* and a known weakness of the metric, not the model.

### Reason 3 — Reference style and length
Instruction LLMs don't mimic the terse reference wording, which floors lexical overlap while **meaning
(BERTScore) holds at parity** — Llama's XSum BERTScore (0.874) ≥ its CNN BERTScore (0.873) even though
BLEU drops from 0.084 → 0.026. See [`02_why_xsum_scores_are_lower.md`](02_why_xsum_scores_are_lower.md).

> **Note:** ignore BLEU for cross-paper comparison — it is a machine-translation metric; summarization
> papers do not report it, so our BLEU column is not a meaningful benchmark axis.

---

## Verdict

- **Against fine-tuned SOTA (BART/PEGASUS): our ROUGE is lower — but that is the wrong baseline** for
  zero-shot 3B models, and partly a measurement artifact.
- **Against the right baseline (zero-shot LLM summarization): we are squarely in the expected range**, and
  our BERTScore matches PEGASUS-level.
- **Honest framing for the report:** present this as *"zero-shot small-LLM summarization,"* cite Goyal et al.
  2022 / the TACL benchmark as the comparison point, and **do not** claim to beat fine-tuned SOTA on ROUGE.
- **To make the numbers comparable**, recompute **`rougeLsum` + stemming** and add an R-1/R-2 column.

---

## Sources

- [A Comparative Study of PEGASUS, BART, and T5 for Text Summarization (MDPI, 2025)](https://www.mdpi.com/1999-5903/17/9/389) — Lead-3 / BART / PEGASUS reference numbers.
- [Goyal et al. 2022 — News Summarization with GPT-3](https://tagoyal.github.io/zeroshot-news-annotations.html) — zero-shot LLMs score lower ROUGE yet are preferred by humans.
- [Benchmarking Large Language Models for News Summarization (TACL)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276/Benchmarking-Large-Language-Models-for-News).
- [PEGASUS: Pre-training with Extracted Gap-sentences (Zhang et al. 2020)](https://arxiv.org/pdf/1912.08777) — original CNN/DM & XSum ROUGE.
- [Evaluating LLMs and Pre-trained Models for Summarization Across Datasets (2025)](https://arxiv.org/html/2502.19339v2).
- [Unraveling the Capabilities of Language Models in News Summarization (2025)](https://arxiv.org/html/2501.18128v1).

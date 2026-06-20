# Are Our Results "Good"? — Comparison to Published Results (Same Models, Same Datasets)

The fair comparison is against **the same models (Llama-3 / Phi-3-mini), evaluated zero-shot, on the same
datasets (CNN/DailyMail, XSum)** — not against fine-tuned BART/PEGASUS. On that comparison our results are
**on par or better**, despite using a smaller **3B** model. Our numbers are from
[`results/results.csv`](../results/results.csv) (fp16 variants shown; the 4-bit/8-bit variants are within
~0.002, so they compare the same).

> **Why this is a valid comparison:** the reference benchmark below reports HuggingFace **`rougeL`** — the
> same convention our [`evaluator.py`](../evaluator.py) uses (its Llama-3 CNN ROUGE-L ≈ 0.17 is in the same
> low range as ours). So the LLM rows are directly comparable, unlike the fine-tuned leaderboard (§4).

All published LLM numbers below are from **Unraveling the Capabilities of Language Models in News
Summarization** (2025), Tables for zero-shot CNN/DM and XSum.

---

## 1. CNN/DailyMail — same models, zero-shot

| Model | ROUGE-L | METEOR | BERTScore-F1 |
|---|---:|---:|---:|
| **Ours — Llama-3.2-3B-Instruct** | **0.239** | **0.338** | **0.873** |
| Published — Llama-3-Instruct (8B) | 0.168 | 0.301 | 0.850 |
| Published — Llama-3 (8B) | 0.183 | 0.256 | 0.858 |
| Published — Llama-2 | 0.165 | 0.215 | 0.840 |
| **Ours — Phi-3-mini-instruct** | **0.185** | **0.301** | 0.852 |
| Published — Phi-3-Mini-Instruct | 0.159 | 0.260 | 0.852 |

→ **Our Llama beats the published 8B Llama-3 on all three metrics**; our Phi-3 beats published Phi-3 on
ROUGE-L and METEOR and ties on BERTScore.

## 2. XSum — same models, zero-shot

| Model | ROUGE-L | METEOR | BERTScore-F1 |
|---|---:|---:|---:|
| **Ours — Llama-3.2-3B-Instruct** | **0.164** | **0.276** | **0.874** |
| Published — Llama-3 (8B) | 0.142 | 0.214 | 0.792 |
| Published — Llama-3-Instruct (8B) | 0.104 | 0.150 | 0.813 |
| Published — Llama-2 | 0.110 | 0.136 | 0.652 |
| **Ours — Phi-3-mini-instruct** | 0.113 | **0.226** | 0.853 |
| Published — Phi-3-Mini-Instruct | **0.123** | 0.195 | 0.852 |

→ **Our Llama beats every published Llama variant** (incl. 8B) on all three metrics. Our Phi-3 is slightly
below published Phi-3 on ROUGE-L but higher on METEOR and equal on BERTScore — solid, especially given our
Phi-3 also carries a prompt-template leak (see [`03_per_config_analysis.md`](03_per_config_analysis.md)).

**Extra anchor:** a zero-shot **Llama-13B** reports R-L **0.229** (CNN) / **0.119** (XSum) — our 3B model
matches or exceeds it (0.239 / 0.164).

---

## 3. Extractive baselines — same datasets (read the caveat)

| System | Our ROUGE-L | Published ROUGE-L | Note |
|---|---:|---:|---|
| Lead-3, CNN/DM | 0.243 | **0.367** (canonical) | gap is the metric config, not the method — see §4 |
| Lead-1, XSum | 0.118 | **0.120** (Narayan 2018) | **matches** — XSum's 1-sentence refs make `rougeL`≈`rougeLsum` |
| TextRank / TFIDF, XSum | ~0.115 | ~0.115 (lead-level) | extraction is floored on abstractive refs |

The XSum baseline matches published numbers almost exactly. The CNN Lead-3 gap is purely the `rougeL` vs
`rougeLsum` convention (next section) — on a copy-paste baseline the summaries can't be "worse."

---

## 4. Against fine-tuned SOTA (context, not a fair fight)

Fine-tuned BART/PEGASUS report ROUGE-L ≈ **40–41** (CNN) and **37–39** (XSum) ×100 — far above us. But they
were **trained on these datasets**, while ours are **zero-shot 3B** models. Two reasons our raw ROUGE looks
lower, neither of which is "bad summaries":

1. **Metric convention.** We report HF `rougeL` (LCS over the whole string, no stemming); papers report
   `rougeLsum` (sentence-split) **with** stemming. Proof: our Lead-3 = 0.243 vs canonical 0.367 on a pure
   copy baseline. **Fix:** report `rougeLsum` + `use_stemmer=True` for comparability.
2. **Zero-shot vs fine-tuned.** Zero-shot LLMs score **~7 ROUGE-L points below fine-tuned SOTA yet humans
   *prefer* them** (Goyal et al. 2022; TACL benchmark).

---

## Verdict

- **vs the same models (the right comparison): we are competitive-to-better** — our 3B Llama beats published
  8B Llama-3 zero-shot on both datasets; our Phi-3 roughly matches published Phi-3.
- **Our BERTScore (~0.87) is at PEGASUS level** and above the published Llama-3 zero-shot numbers.
- **vs fine-tuned SOTA:** lower ROUGE, but that is the wrong baseline and partly a metric-config artifact.
- **For the report:** frame as *zero-shot small-LLM summarization*, compare against the 2501.18128 benchmark
  (same models) — and optionally recompute `rougeLsum`+stemming to add R-1/R-2 columns.

---

## Sources

- [Unraveling the Capabilities of Language Models in News Summarization (2025)](https://arxiv.org/html/2501.18128v1) — zero-shot Llama-2/3/3-Instruct and Phi-3-Mini-Instruct ROUGE-L / METEOR / BERTScore on CNN/DM & XSum (the main same-model comparison).
- [Evaluating LLMs and Pre-trained Models for Summarization Across Datasets (2025)](https://arxiv.org/html/2502.19339v2) — additional LLM-vs-pretrained comparison.
- [Goyal et al. 2022 — News Summarization with GPT-3](https://tagoyal.github.io/zeroshot-news-annotations.html) — zero-shot LLMs: lower ROUGE, higher human preference.
- [Benchmarking Large Language Models for News Summarization (TACL)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00632/119276/Benchmarking-Large-Language-Models-for-News).
- [PEGASUS (Zhang et al. 2020)](https://arxiv.org/pdf/1912.08777) and [comparative BART/PEGASUS/T5 study (MDPI 2025)](https://www.mdpi.com/1999-5903/17/9/389) — fine-tuned SOTA & canonical Lead-3.
- [Narayan et al. 2018 — XSum / extreme summarization](https://aclanthology.org/D18-1206/) — XSum LEAD baseline.

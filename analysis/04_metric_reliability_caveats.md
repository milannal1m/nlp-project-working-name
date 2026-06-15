# 4. Which metrics can we trust?

## Q2 — Can we trust SummaC's numbers?

- **Answer:** Only loosely. It contradicts itself, so never rank systems by it across styles.
- **Reason:** SummaC scores each summary *sentence* by NLI; it punishes decontextualized sentences (loose pronouns), not actual lies.
- **Proof — two verbatim-copy baselines, 9× apart on CNN:**

| CNN, both copy the source word-for-word | SummaC |
|---|---:|
| Lead-3 (opening sentences) | 0.89 |
| TextRank (central mid-article sentences) | 0.10 |

  - Real line (`TextRank_cnn_dailymail`, line 2): *"The group has gained territory, cash and recruits…"* — an **exact substring of the article** (verified), yet scored 0.10 because it opens with "The group" / "those young Muslim men" (no context).
  - More contradictions: Llama CNN 0.04 vs XSum 0.84 (same model); Phi-3_8bit CNN 0.97 vs Llama CNN 0.04 (same articles).
- **Use it as:** a coarse signal within one summary style only.

## Q3 — Can we compare BLEU across datasets?

- **Answer:** No — XSum BLEU is at its floor.
- **Reason:** One 21-word reference vs 2+ sentence summaries → almost all n-grams miss.
- **Proof:** XSum BLEU range 0.007–0.026; gaps like Phi-3 0.0121 vs 0.0124 are noise. Use rank order only.

## Q4 — How much does BERTScore tell us apart?

- **Answer:** Direction is reliable, but it barely separates models.
- **Reason:** Any fluent English scores high; everything lands in a 0.85–0.874 band.
- **Proof:** Llama 0.873 vs Phi-3 0.852 is real; sub-0.002 gaps (quant modes) are not worth reading.

## What to actually conclude

- **Llama-3.2-3B is the best summarizer** (clearest on BERTScore). *(High confidence.)*
- **4/8-bit quantization is free** on reference metrics. *(High confidence.)*
- **Lead-3 wins CNN word-overlap** because CNN references are extractive — a fact about the data, not summary quality.
- **XSum's lower lexical scores are a reference-style artifact**, not worse summaries (BERTScore parity).
- **Phi-3's ranking is invalid** until re-run with its chat template.
- **SummaC and QAFactEval give no usable factuality ranking** in this run.

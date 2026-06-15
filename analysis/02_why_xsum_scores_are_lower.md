# 2. Why are XSum scores lower than CNN/DailyMail?

## Q1 — Why is XSum's BLEU/ROUGE/METEOR so much lower?

- **Answer:** Because XSum references are single, rewritten sentences — word-overlap metrics can't score them high, no matter how good the summary is.
- **Reason:**
  - XSum reference = 1 sentence (21.7 words), and **35.7% of its words never appear in the article**. A model can't copy words that aren't there.
  - CNN reference = 3.9 sentences (52.9 words), only 14.6% novel — almost copied from the article, so overlap is easy.
- **Proof — the gap (best system):**

| Metric | CNN best | XSum best | XSum ÷ CNN |
|---|---:|---:|---:|
| BLEU | 0.1150 (Lead-3) | 0.0256 (Llama) | 0.22× |
| ROUGE-L | 0.2428 (Lead-3) | 0.1646 (Llama) | 0.68× |
| METEOR | 0.3854 (Lead-3) | 0.2762 (Llama) | 0.72× |

- **Proof — real data line** (`summaries/Lead-1_xsum_summaries.jsonl`, line 1):
  - Reference: *"Uzbekistan's Hasanboy Dusmatov won Olympic gold in the men's light-flyweight…"*
  - The words **"Uzbekistan", "gold", "light-flyweight" appear nowhere in the 86-word article**. The editor added them from outside knowledge.

## Q2 — Are the XSum summaries actually worse, or just scored lower?

- **Answer:** Just scored lower. The summaries are fine.
- **Reason:** Switch from word-overlap to a *meaning* metric and the gap disappears.
- **Proof:**
  - BERTScore (Llama): XSum **0.8736** ≥ CNN **0.8729**.
  - Real example (`summaries/Llama_None_xsum_summaries.jsonl`, line 2): a correct, fluent summary scores only **ROUGE-L 0.135 / unigram precision 0.18**, because the reference's framing words **"Great Britain", "1-1", "day one", "comfortable", "defence" are not in the article** (the editor inferred "tied at 1-1"). "Japan", "Davis Cup", "Birmingham" are.

## Q3 — Why do the extractive baselines collapse on XSum?

- **Answer:** XSum is built so the article's own sentences are *not* the summary; copying can't win.
- **Reason:** The reference is a separate abstractive sentence, so any extracted sentence misses it.
- **Proof:**
  - All 4 baselines pile up at the floor on XSum: BLEU 0.007–0.009, ROUGE-L ~0.115 (vs Lead-1 CNN BLEU 0.0365).
  - **3.6% (407/11,334)** of XSum articles start with BBC boilerplate, so Lead-1 returns junk. Real line (`summaries/Lead-1_xsum_summaries.jsonl`, line 1): generated = *"Media playback is not supported on this device\nDusmatov, 23, was awarded a unanimous points victory…"*

## Bottom line

- "XSum is lower" is a statement about **word-overlap metrics × XSum's abstractive references**.
- On meaning (BERTScore) and faithfulness (SummaC, for LLMs: XSum 0.84 vs CNN 0.04), XSum is summarized **at least as well** as CNN.

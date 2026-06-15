# 3. Why each system scores what it scores

## Q1 — Why does Lead-3 win on CNN but lose on XSum?

- **Answer:** It copies the first 3 sentences — perfect for CNN's extractive references, useless for XSum's abstractive ones.
- **Reason:** CNN highlights are near-copies of the article lead; XSum references are rewritten.
- **Proof:** CNN — best BLEU 0.1150, ROUGE-L 0.2428, METEOR 0.3854. XSum — floor (BLEU 0.0077).

## Q2 — Why is Llama the best system?

- **Answer:** It writes clean ~2-sentence paraphrases that match the reference *meaning*.
- **Reason:** Abstractive style fits XSum; faithful rewriting beats extraction on meaning.
- **Proof:**
  - Best LLM on every reference metric; best BERTScore overall (CNN 0.8731 @8bit, XSum 0.8736 @None).
  - On CNN it loses ROUGE-L to Lead-3 (0.239 vs 0.243) but **wins BERTScore** (0.873 vs 0.869) — it rewrites instead of copying.
  - Clean output: instruction/meta junk in only 0.6% (XSum) / 5.4% (CNN) of summaries.
- **Caveat (real line, `Llama_None_xsum`, line 1):** on thin sources it hallucinates — *"The 2020 Olympic… her country… Note: I've added a brief summary…"* ("2020" wrong; bout was Rio 2016).

## Q3 — Why does Phi-3 underperform everywhere?

- **Answer:** A prompt-format bug, **not** weak ability.
- **Reason:** The pipeline sends a raw prompt with no chat template, so Phi-3 never stops cleanly and dumps instruction-style text.
- **Proof:**
  - Template-leak markers (`## Your task`, `Document:`) in **90.9% of XSum / 81.9% of CNN** outputs; avg length ~103 words (vs Llama ~60–73).
  - Real line (`Phi-3_None_xsum`, line 2): *"…The British team… will compete in doubles… ## Your task:Based on the provided document, create a comprehensive analysis…"* (also hallucinates "final" and "win over Belgium").
  - This junk tanks BLEU/ROUGE (precision) and lowers BERTScore (0.852 vs Llama 0.873).
- **Action:** apply the chat template + stop tokens and re-run. **Phi-3's current numbers are not a fair ranking.**

## Q4 — Does 4-bit / 8-bit quantization change anything?

- **Answer:** No measurable change in quality.
- **Reason:** Same model at three precisions produces the same summaries.
- **Proof:**

| (XSum) | fp16 | 4bit | 8bit | spread |
|---|---:|---:|---:|---:|
| Llama ROUGE-L | 0.1639 | 0.1646 | 0.1631 | 0.0015 |
| Llama BERTScore | 0.8736 | 0.8735 | 0.8734 | 0.0002 |

  - Real lines (same Murray article): fp16 and 8-bit are almost token-identical; 4-bit rephrases but keeps every fact.
  - **Conclusion: run the cheap 4-bit model.** (SummaC does move under quantization, but that's metric noise — see [04](04_metric_reliability_caveats.md).)

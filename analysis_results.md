# Evaluation Results — Does Quantization Hurt?

**Project**: Evaluating Quantized Small Language Models for News Summarization  
**Datasets evaluated**: CNN/DailyMail (test, n=11,490), XSum (test, n=11,334)  
**Metrics computed**: ROUGE-1, ROUGE-2, ROUGE-L (F1, lightweight tokenizer)  
**Pending metrics**: BLEU, METEOR, BERTScore, SummaC_Conv, QA-Eval (require cluster environment)

---

## 1. Models Evaluated

| Family | Type | Model | Precision | Status |
|--------|------|-------|-----------|--------|
| Conventional | Extractive | Lead-1 | — | ✅ Evaluated |
| Conventional | Extractive | Lead-3 | — | ✅ Evaluated |
| Conventional | Extractive | TF-IDF | — | ✅ Evaluated |
| Conventional | Extractive | TextRank | — | ✅ Evaluated |
| Conventional | Extractive | BERTSumExt | — | ⬜ Pending |
| Conventional | Abstractive | Pointer-Generator | — | ⬜ Pending |
| Conventional | Abstractive | PEGASUS | — | ⬜ Pending |
| Conventional | Abstractive | BART | — | ⬜ Pending |
| Conventional | Abstractive | BERTSumAbs | — | ⬜ Pending |
| Conventional | Hybrid | BERTSumExtAbs | — | ⬜ Pending |
| SLM-based | Hybrid | Llama 3.2 3B-Instruct | Full (16-bit) | ⬜ Pending |
| SLM-based | Hybrid | Llama 3.2 3B-Instruct | 8-bit | ⬜ Pending |
| SLM-based | Hybrid | Llama 3.2 3B-Instruct | **4-bit** | ✅ Evaluated |
| SLM-based | Hybrid | Phi-3 Mini | Full (16-bit) | ⬜ Pending |
| SLM-based | Hybrid | Phi-3 Mini | 8-bit | ⬜ Pending |
| SLM-based | Hybrid | Phi-3 Mini | 4-bit | ⬜ Pending |
| SLM-based | Hybrid | Qwen2-1.5B-Instruct | Full (16-bit) | ⬜ Pending |
| LLM-based | Hybrid | Qwen2-7B-Instruct | Full | ⬜ Pending |
| LLM-based | Hybrid | Llama 3-70B-Instruct | Full | ⬜ Pending |
| LLM-based | Hybrid | Qwen2-72B-Instruct | Full | ⬜ Pending |

---

## 2. CNN/DailyMail Results (Test Set, n = 11,490)

### 2.1 Lexical Overlap (ROUGE F1)

| Family | Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|-------|--------:|--------:|--------:|
| Conventional / Extractive | **Lead-3** | **0.3838** ± 0.1193 | **0.1692** ± 0.1126 | **0.2428** ± 0.1024 |
| SLM / 4-bit | **Llama 3.2 3B-Ins (4-bit)** | 0.3690 ± 0.1021 | 0.1371 ± 0.0816 | 0.2359 ± 0.0804 |
| Conventional / Extractive | TextRank | 0.2844 ± 0.1109 | 0.0875 ± 0.0911 | 0.1808 ± 0.0867 |
| Conventional / Extractive | TF-IDF | 0.2684 ± 0.1097 | 0.0800 ± 0.0910 | 0.1733 ± 0.0893 |
| Conventional / Extractive | Lead-1 | 0.2640 ± 0.1219 | 0.0941 ± 0.0962 | 0.1832 ± 0.0977 |

### 2.2 Summary Length & Compression

| Model | Avg Doc Len | Avg Ref Len | Avg Gen Len (± std) | Avg Gen Sents | Compression Ratio |
|-------|------------:|------------:|--------------------:|--------------:|------------------:|
| Lead-3 | 694.6 | 52.9 | 82.2 ± 23.7 | 3.4 | 0.1529 |
| Llama 3.2 3B-Ins (4-bit) | 694.6 | 52.9 | 71.7 ± 19.8 | 2.4 | 0.1285 |
| TextRank | 694.6 | 52.9 | 78.3 ± 21.7 | 2.3 | 0.1394 |
| TF-IDF | 694.6 | 52.9 | 47.6 ± 11.9 | 2.1 | 0.0881 |
| Lead-1 | 694.6 | 52.9 | 25.7 ± 10.5 | 1.1 | 0.0475 |

> **Observation (CNN/DM):** Lead-3 achieves the highest ROUGE scores, marginally outperforming Llama 3.2 3B-Instruct (4-bit). This is consistent with the known lead bias in CNN/DailyMail, where the first three sentences of an article heavily overlap with the reference highlights. Llama 3.2 (4-bit) generates more concise summaries (71.7 vs 82.2 tokens) while maintaining competitive ROUGE-L (0.2359 vs 0.2428), suggesting effective compression. The extractive baselines TF-IDF and TextRank lag behind, despite generating longer summaries than Lead-1.

---

## 3. XSum Results (Test Set, n = 11,334)

### 3.1 Lexical Overlap (ROUGE F1)

| Family | Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|-------|--------:|--------:|--------:|
| SLM / 4-bit | **Llama 3.2 3B-Ins (4-bit)** | **0.2361** ± 0.0811 | **0.0637** ± 0.0570 | **0.1646** ± 0.0626 |
| Conventional / Extractive | Lead-3 | 0.1782 ± 0.0666 | 0.0254 ± 0.0307 | 0.1154 ± 0.0440 |
| Conventional / Extractive | TF-IDF | 0.1736 ± 0.0717 | 0.0211 ± 0.0338 | 0.1151 ± 0.0486 |
| Conventional / Extractive | TextRank | 0.1660 ± 0.0677 | 0.0270 ± 0.0352 | 0.1147 ± 0.0467 |
| Conventional / Extractive | Lead-1 | 0.1592 ± 0.0809 | 0.0160 ± 0.0322 | 0.1177 ± 0.0597 |

### 3.2 Summary Length & Compression

| Model | Avg Doc Len | Avg Ref Len | Avg Gen Len (± std) | Avg Gen Sents | Compression Ratio |
|-------|------------:|------------:|--------------------:|--------------:|------------------:|
| Llama 3.2 3B-Ins (4-bit) | 385.4 | 21.7 | 57.7 ± 13.6 | 2.2 | 0.2606 |
| Lead-3 | 385.4 | 21.7 | 69.4 ± 37.8 | 3.7 | 0.3024 |
| TextRank | 385.4 | 21.7 | 65.0 ± 26.4 | 2.1 | 0.2605 |
| TF-IDF | 385.4 | 21.7 | 44.2 ± 13.8 | 2.1 | 0.1946 |
| Lead-1 | 385.4 | 21.7 | 24.3 ± 16.2 | 1.2 | 0.1124 |

> **Observation (XSum):** Llama 3.2 3B-Instruct (4-bit) substantially outperforms all extractive baselines on XSum (ROUGE-L 0.1646 vs ≤ 0.1177). This is expected: XSum requires single-sentence abstractive summaries with 83.45% novel bigrams, a regime where extractive methods fundamentally cannot match the reference style. All extractive baselines cluster at ROUGE-L ≈ 0.115, confirming the dataset's high abstractiveness. Notably, all generated summaries are considerably longer than the references (avg 21.7 tokens), indicating room for improved compression.

---

## 4. Cross-Dataset Comparison

| Model | CNN/DM ROUGE-L | XSum ROUGE-L | Δ (CNN→XSum) |
|-------|---------------:|-------------:|-------------:|
| Lead-3 | 0.2428 | 0.1154 | −0.1274 |
| Llama 3.2 3B-Ins (4-bit) | 0.2359 | 0.1646 | −0.0713 |
| TextRank | 0.1808 | 0.1147 | −0.0661 |
| TF-IDF | 0.1733 | 0.1151 | −0.0582 |
| Lead-1 | 0.1832 | 0.1177 | −0.0655 |

> **Observation:** All models degrade on XSum relative to CNN/DM, but the degradation is most severe for Lead-3 (Δ = −0.1274), which loses its lead-bias advantage. Llama 3.2 (4-bit) shows the smallest relative drop among the top performers, suggesting that the SLM's abstractive capability generalizes better across summarization styles.

---

## 5. Key Findings (Preliminary)

1. **Lead bias dominates CNN/DailyMail:** Lead-3 achieves the highest ROUGE scores, confirming the well-documented lead bias in CNN/DM where reference highlights heavily overlap with opening sentences.

2. **Abstractive capability matters on XSum:** Llama 3.2 3B-Instruct (4-bit) is the only model that meaningfully exceeds the extractive baselines on XSum (+43% ROUGE-L over the best extractive method), demonstrating genuine abstractive summarization.

3. **4-bit quantization retains competitive quality:** Even at 4-bit precision, Llama 3.2 3B-Instruct nearly matches Lead-3 on CNN/DM and decisively outperforms all extractive baselines on XSum. Full-precision and 8-bit comparisons are needed to quantify the quantization quality gap.

4. **Generated summaries are too long on XSum:** All models produce summaries 2–3× longer than XSum references (avg 21.7 tokens), suggesting the prompt or model behavior may need tuning for extreme compression tasks.

5. **Extractive baselines plateau on XSum:** TF-IDF, TextRank, Lead-1, and Lead-3 all cluster at ROUGE-L ≈ 0.115 on XSum, reflecting a hard ceiling for extractive methods on highly abstractive datasets.

---

## 6. Pending Evaluation Steps

| Step | Description | Environment |
|------|-------------|-------------|
| Full-precision & 8-bit Llama | Generate summaries with full-precision and 8-bit Llama 3.2 3B-Instruct | GPU cluster |
| Phi-3 Mini variants | Generate summaries with full, 8-bit, 4-bit Phi-3 Mini | GPU cluster |
| Newsroom dataset | Run all models on Newsroom test set | GPU cluster |
| BLEU, METEOR | Compute additional lexical metrics | Cluster (evaluate lib) |
| BERTScore | Compute semantic similarity metric | Cluster (GPU) |
| SummaC_Conv | Compute NLI-based factual consistency | Cluster (GPU) |
| QA-Eval on NewsQASum | Run dual-context QA factuality pipeline | Cluster (GPU) |
| LLM baselines | Qwen2-7B, Llama3-70B, Qwen2-72B | GPU cluster |
| Efficiency metrics | Inference time and memory usage | GPU cluster |

---

## 7. Methodology Notes

- **ROUGE computation:** Scores were computed using a lightweight Python tokenizer without stemming or stopword removal. For publication-quality numbers, use the `evaluate` library with the standard `rouge_score` backend.
- **Compression ratio:** Defined as `avg_gen_len / avg_doc_len` (token-level).
- **Test sets:** CNN/DailyMail test (11,490 samples), XSum test (11,334 samples) — matching the dataset statistics in Table 1 of the paper.
- **Prompt template:** `"News: {news}\nSummarize the news in two sentences. Summary:"` (zero-shot, as described in Section 3.1 of the paper).
- **Source files:** Generated from JSONL files in `summaries/` directory. Each record contains `news`, `reference_summary`, and `generated_summary` fields.

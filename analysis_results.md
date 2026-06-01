# Summarization Analysis Results

**Generated on**: Analysis of `summaries/` folder  
**Total files analyzed**: 10  
**Models**: Lead-1, Lead-3, Llama_4bit, TFIDF, TextRank  
**Datasets**: CNN/DailyMail, XSum  

---

## Overview — ROUGE Scores

| File | Samples | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|------|--------:|-----------:|-----------:|-----------:|
| Lead-1_cnn_dailymail_summaries.jsonl | 11490 | 0.2640 ± 0.1219 | 0.0941 ± 0.0962 | 0.1832 ± 0.0977 |
| Lead-1_xsum_summaries.jsonl | 11334 | 0.1592 ± 0.0809 | 0.0160 ± 0.0322 | 0.1177 ± 0.0597 |
| Lead-3_cnn_dailymail_summaries.jsonl | 11490 | 0.3838 ± 0.1193 | 0.1692 ± 0.1126 | 0.2428 ± 0.1024 |
| Lead-3_xsum_summaries.jsonl | 11334 | 0.1782 ± 0.0666 | 0.0254 ± 0.0307 | 0.1154 ± 0.0440 |
| Llama_4bit_cnn_dailymail_summaries.jsonl | 11490 | 0.3690 ± 0.1021 | 0.1371 ± 0.0816 | 0.2359 ± 0.0804 |
| Llama_4bit_xsum_summaries.jsonl | 11334 | 0.2361 ± 0.0811 | 0.0637 ± 0.0570 | 0.1646 ± 0.0626 |
| TFIDF_cnn_dailymail_summaries.jsonl | 11490 | 0.2684 ± 0.1097 | 0.0800 ± 0.0910 | 0.1733 ± 0.0893 |
| TFIDF_xsum_summaries.jsonl | 11334 | 0.1736 ± 0.0717 | 0.0211 ± 0.0338 | 0.1151 ± 0.0486 |
| TextRank_cnn_dailymail_summaries.jsonl | 11490 | 0.2844 ± 0.1109 | 0.0875 ± 0.0911 | 0.1808 ± 0.0867 |
| TextRank_xsum_summaries.jsonl | 11334 | 0.1660 ± 0.0677 | 0.0270 ± 0.0352 | 0.1147 ± 0.0467 |

---

## Summary Length & Compression

| File | Avg News Len | Avg Ref Len | Avg Gen Len (± std) | Avg Gen Sents | Compression Ratio |
|------|------------:|------------:|--------------------:|--------------:|------------------:|
| Lead-1_cnn_dailymail_summaries.jsonl | 694.6 | 52.9 | 25.7 ± 10.5 | 1.1 | 0.0475 |
| Lead-1_xsum_summaries.jsonl | 385.4 | 21.7 | 24.3 ± 16.2 | 1.2 | 0.1124 |
| Lead-3_cnn_dailymail_summaries.jsonl | 694.6 | 52.9 | 82.2 ± 23.7 | 3.4 | 0.1529 |
| Lead-3_xsum_summaries.jsonl | 385.4 | 21.7 | 69.4 ± 37.8 | 3.7 | 0.3024 |
| Llama_4bit_cnn_dailymail_summaries.jsonl | 694.6 | 52.9 | 71.7 ± 19.8 | 2.4 | 0.1285 |
| Llama_4bit_xsum_summaries.jsonl | 385.4 | 21.7 | 57.7 ± 13.6 | 2.2 | 0.2606 |
| TFIDF_cnn_dailymail_summaries.jsonl | 694.6 | 52.9 | 47.6 ± 11.9 | 2.1 | 0.0881 |
| TFIDF_xsum_summaries.jsonl | 385.4 | 21.7 | 44.2 ± 13.8 | 2.1 | 0.1946 |
| TextRank_cnn_dailymail_summaries.jsonl | 694.6 | 52.9 | 78.3 ± 21.7 | 2.3 | 0.1394 |
| TextRank_xsum_summaries.jsonl | 385.4 | 21.7 | 65.0 ± 26.4 | 2.1 | 0.2605 |

---

## CNN/DailyMail — Model Comparison

| Model | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Avg Gen Len | Compression |
|-------|----------:|-----------:|-----------:|------------:|------------:|
| Lead-3 | 0.3838 | 0.1692 | 0.2428 | 82.2 | 0.1529 |
| Llama_4bit | 0.3690 | 0.1371 | 0.2359 | 71.7 | 0.1285 |
| Lead-1 | 0.2640 | 0.0941 | 0.1832 | 25.7 | 0.0475 |
| TextRank | 0.2844 | 0.0875 | 0.1808 | 78.3 | 0.1394 |
| TFIDF | 0.2684 | 0.0800 | 0.1733 | 47.6 | 0.0881 |

> **Best model on CNN/DailyMail**: **Lead-3** (ROUGE-L F1 = 0.2428)

---

## XSum — Model Comparison

| Model | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Avg Gen Len | Compression |
|-------|----------:|-----------:|-----------:|------------:|------------:|
| Llama_4bit | 0.2361 | 0.0637 | 0.1646 | 57.7 | 0.2606 |
| Lead-1 | 0.1592 | 0.0160 | 0.1177 | 24.3 | 0.1124 |
| Lead-3 | 0.1782 | 0.0254 | 0.1154 | 69.4 | 0.3024 |
| TFIDF | 0.1736 | 0.0211 | 0.1151 | 44.2 | 0.1946 |
| TextRank | 0.1660 | 0.0270 | 0.1147 | 65.0 | 0.2605 |

> **Best model on XSum**: **Llama_4bit** (ROUGE-L F1 = 0.1646)

---

## Key Takeaways

- **Best overall ROUGE-L**: Lead-3 on CNN/DailyMail (0.2428)
- **Shortest summaries**: Lead-1 (avg 24.3 tokens)
- **Longest summaries**: Lead-3 (avg 82.2 tokens)
- **Note**: ROUGE scores are computed using a lightweight tokenizer (no stemming). For publication-quality numbers, run the full evaluator with BERTScore and SummaC.

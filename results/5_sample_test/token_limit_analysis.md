# Token-limit analysis

Generated summaries whose re-tokenized length reached the prompt's `max_new_tokens` cap (i.e. generation was cut off by length, not a stop token). Baseline files have no cap and are skipped.

| file | model | prompt | cap | n | at_cap | at_cap % |
| --- | --- | --- | --- | --- | --- | --- |
| Lead-1_cnn_dailymail_5_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-1_xsum_5_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-3_cnn_dailymail_5_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-3_xsum_5_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Llama_P1_16bit_cnn_dailymail_5_summaries.jsonl | Llama | P1 | 1000 | 5 | 0 | 0.0% |
| Llama_P1_16bit_xsum_5_summaries.jsonl | Llama | P1 | 1000 | 5 | 0 | 0.0% |
| Llama_P1_4bit_cnn_dailymail_5_summaries.jsonl | Llama | P1 | 1000 | 5 | 0 | 0.0% |
| Llama_P1_4bit_xsum_5_summaries.jsonl | Llama | P1 | 1000 | 5 | 0 | 0.0% |
| Llama_P1_8bit_cnn_dailymail_5_summaries.jsonl | Llama | P1 | 1000 | 5 | 0 | 0.0% |
| Llama_P1_8bit_xsum_5_summaries.jsonl | Llama | P1 | 1000 | 5 | 0 | 0.0% |
| Llama_P2_16bit_cnn_dailymail_5_summaries.jsonl | Llama | P2 | 1000 | 5 | 0 | 0.0% |
| Llama_P2_16bit_xsum_5_summaries.jsonl | Llama | P2 | 1000 | 5 | 0 | 0.0% |
| Llama_P2_4bit_cnn_dailymail_5_summaries.jsonl | Llama | P2 | 1000 | 5 | 0 | 0.0% |
| Llama_P2_4bit_xsum_5_summaries.jsonl | Llama | P2 | 1000 | 5 | 0 | 0.0% |
| Llama_P2_8bit_cnn_dailymail_5_summaries.jsonl | Llama | P2 | 1000 | 5 | 0 | 0.0% |
| Llama_P2_8bit_xsum_5_summaries.jsonl | Llama | P2 | 1000 | 5 | 0 | 0.0% |
| Llama_P3_16bit_cnn_dailymail_5_summaries.jsonl | Llama | P3 | 1000 | 5 | 0 | 0.0% |
| Llama_P3_16bit_xsum_5_summaries.jsonl | Llama | P3 | 1000 | 5 | 0 | 0.0% |
| Llama_P3_4bit_cnn_dailymail_5_summaries.jsonl | Llama | P3 | 1000 | 5 | 0 | 0.0% |
| Llama_P3_4bit_xsum_5_summaries.jsonl | Llama | P3 | 1000 | 5 | 0 | 0.0% |
| Llama_P3_8bit_cnn_dailymail_5_summaries.jsonl | Llama | P3 | 1000 | 5 | 0 | 0.0% |
| Llama_P3_8bit_xsum_5_summaries.jsonl | Llama | P3 | 1000 | 5 | 0 | 0.0% |
| Phi_P1_16bit_cnn_dailymail_5_summaries.jsonl | Phi | P1 | 1000 | 5 | 0 | 0.0% |
| Phi_P1_16bit_xsum_5_summaries.jsonl | Phi | P1 | 1000 | 5 | 0 | 0.0% |
| Phi_P1_4bit_cnn_dailymail_5_summaries.jsonl | Phi | P1 | 1000 | 5 | 0 | 0.0% |
| Phi_P1_4bit_xsum_5_summaries.jsonl | Phi | P1 | 1000 | 5 | 0 | 0.0% |
| Phi_P1_8bit_cnn_dailymail_5_summaries.jsonl | Phi | P1 | 1000 | 5 | 0 | 0.0% |
| Phi_P1_8bit_xsum_5_summaries.jsonl | Phi | P1 | 1000 | 5 | 0 | 0.0% |
| Phi_P2_16bit_cnn_dailymail_5_summaries.jsonl | Phi | P2 | 1000 | 5 | 0 | 0.0% |
| Phi_P2_16bit_xsum_5_summaries.jsonl | Phi | P2 | 1000 | 5 | 0 | 0.0% |
| Phi_P2_4bit_cnn_dailymail_5_summaries.jsonl | Phi | P2 | 1000 | 5 | 0 | 0.0% |
| Phi_P2_4bit_xsum_5_summaries.jsonl | Phi | P2 | 1000 | 5 | 0 | 0.0% |
| Phi_P2_8bit_cnn_dailymail_5_summaries.jsonl | Phi | P2 | 1000 | 5 | 0 | 0.0% |
| Phi_P2_8bit_xsum_5_summaries.jsonl | Phi | P2 | 1000 | 5 | 0 | 0.0% |
| Phi_P3_16bit_cnn_dailymail_5_summaries.jsonl | Phi | P3 | 1000 | 5 | 0 | 0.0% |
| Phi_P3_16bit_xsum_5_summaries.jsonl | Phi | P3 | 1000 | 5 | 0 | 0.0% |
| Phi_P3_4bit_cnn_dailymail_5_summaries.jsonl | Phi | P3 | 1000 | 5 | 0 | 0.0% |
| Phi_P3_4bit_xsum_5_summaries.jsonl | Phi | P3 | 1000 | 5 | 0 | 0.0% |
| Phi_P3_8bit_cnn_dailymail_5_summaries.jsonl | Phi | P3 | 1000 | 5 | 0 | 0.0% |
| Phi_P3_8bit_xsum_5_summaries.jsonl | Phi | P3 | 1000 | 5 | 0 | 0.0% |
| TFIDF_cnn_dailymail_5_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TFIDF_xsum_5_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TextRank_cnn_dailymail_5_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TextRank_xsum_5_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| **TOTAL** |  |  |  | 180 | 0 | 0.0% |

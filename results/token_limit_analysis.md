# Token-limit analysis

Generated summaries whose re-tokenized length reached the prompt's `max_new_tokens` cap (i.e. generation was cut off by length, not a stop token). Baseline files have no cap and are skipped.

| file | model | prompt | cap | n | at_cap | at_cap % |
| --- | --- | --- | --- | --- | --- | --- |
| Lead-1_cnn_dailymail_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-1_xsum_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-3_cnn_dailymail_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-3_xsum_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Llama_P1_16bit_cnn_dailymail_full_summaries.jsonl | Llama | P1 | 150 | 11490 | 20 | 0.2% |
| Llama_P1_16bit_xsum_full_summaries.jsonl | Llama | P1 | 150 | 11334 | 22 | 0.2% |
| Llama_P1_4bit_cnn_dailymail_full_summaries.jsonl | Llama | P1 | 150 | 11490 | 14 | 0.1% |
| Llama_P1_4bit_xsum_full_summaries.jsonl | Llama | P1 | 150 | 11334 | 20 | 0.2% |
| Llama_P1_8bit_cnn_dailymail_full_summaries.jsonl | Llama | P1 | 150 | 11490 | 22 | 0.2% |
| Llama_P1_8bit_xsum_full_summaries.jsonl | Llama | P1 | 150 | 11334 | 24 | 0.2% |
| Llama_P2_16bit_cnn_dailymail_full_summaries.jsonl | Llama | P2 | 150 | 11490 | 69 | 0.6% |
| Llama_P2_16bit_xsum_full_summaries.jsonl | Llama | P2 | 150 | 11334 | 15 | 0.1% |
| Llama_P2_4bit_cnn_dailymail_full_summaries.jsonl | Llama | P2 | 150 | 11490 | 43 | 0.4% |
| Llama_P2_4bit_xsum_full_summaries.jsonl | Llama | P2 | 150 | 11334 | 11 | 0.1% |
| Llama_P2_8bit_cnn_dailymail_full_summaries.jsonl | Llama | P2 | 150 | 11490 | 77 | 0.7% |
| Llama_P2_8bit_xsum_full_summaries.jsonl | Llama | P2 | 150 | 11334 | 24 | 0.2% |
| Llama_P3_16bit_cnn_dailymail_full_summaries.jsonl | Llama | P3 | 300 | 11490 | 905 | 7.9% |
| Llama_P3_16bit_xsum_full_summaries.jsonl | Llama | P3 | 300 | 11334 | 373 | 3.3% |
| Llama_P3_4bit_cnn_dailymail_full_summaries.jsonl | Llama | P3 | 300 | 11490 | 833 | 7.2% |
| Llama_P3_4bit_xsum_full_summaries.jsonl | Llama | P3 | 300 | 11334 | 332 | 2.9% |
| Llama_P3_8bit_cnn_dailymail_full_summaries.jsonl | Llama | P3 | 300 | 8048 | 611 | 7.6% |
| Llama_P3_8bit_xsum_full_summaries.jsonl | Llama | P3 | 300 | 8589 | 337 | 3.9% |
| Phi_P1_16bit_cnn_dailymail_full_summaries.jsonl | Phi | P1 | 150 | 11490 | 10 | 0.1% |
| Phi_P1_16bit_xsum_full_summaries.jsonl | Phi | P1 | 150 | 11334 | 11 | 0.1% |
| Phi_P1_4bit_cnn_dailymail_full_summaries.jsonl | Phi | P1 | 150 | 11490 | 24 | 0.2% |
| Phi_P1_4bit_xsum_full_summaries.jsonl | Phi | P1 | 150 | 11334 | 24 | 0.2% |
| Phi_P1_8bit_cnn_dailymail_full_summaries.jsonl | Phi | P1 | 150 | 11490 | 8 | 0.1% |
| Phi_P1_8bit_xsum_full_summaries.jsonl | Phi | P1 | 150 | 11334 | 11 | 0.1% |
| Phi_P2_16bit_cnn_dailymail_full_summaries.jsonl | Phi | P2 | 150 | 11490 | 1009 | 8.8% |
| Phi_P2_16bit_xsum_full_summaries.jsonl | Phi | P2 | 150 | 11334 | 368 | 3.2% |
| Phi_P2_4bit_cnn_dailymail_full_summaries.jsonl | Phi | P2 | 150 | 11490 | 861 | 7.5% |
| Phi_P2_4bit_xsum_full_summaries.jsonl | Phi | P2 | 150 | 11334 | 328 | 2.9% |
| Phi_P2_8bit_cnn_dailymail_full_summaries.jsonl | Phi | P2 | 150 | 11490 | 1135 | 9.9% |
| Phi_P2_8bit_xsum_full_summaries.jsonl | Phi | P2 | 150 | 11334 | 426 | 3.8% |
| Phi_P3_16bit_cnn_dailymail_full_summaries.jsonl | Phi | P3 | 300 | 11490 | 157 | 1.4% |
| Phi_P3_16bit_xsum_full_summaries.jsonl | Phi | P3 | 300 | 11334 | 97 | 0.9% |
| Phi_P3_4bit_cnn_dailymail_full_summaries.jsonl | Phi | P3 | 300 | 11490 | 157 | 1.4% |
| Phi_P3_4bit_xsum_full_summaries.jsonl | Phi | P3 | 300 | 11334 | 76 | 0.7% |
| Phi_P3_8bit_cnn_dailymail_full_summaries.jsonl | Phi | P3 | 300 | 11490 | 224 | 1.9% |
| Phi_P3_8bit_xsum_full_summaries.jsonl | Phi | P3 | 300 | 11334 | 115 | 1.0% |
| TFIDF_cnn_dailymail_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TFIDF_xsum_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TextRank_cnn_dailymail_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TextRank_xsum_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| **TOTAL** |  |  |  | 404645 | 8793 | 2.2% |

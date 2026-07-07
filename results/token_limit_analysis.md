# Token-limit analysis

Generated summaries whose re-tokenized length reached the prompt's `max_new_tokens` cap (i.e. generation was cut off by length, not a stop token). Baseline files have no cap and are skipped.

| file | model | prompt | cap | n | at_cap | at_cap % |
| --- | --- | --- | --- | --- | --- | --- |
| Lead-1_cnn_dailymail_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-1_xsum_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-1_xu_cnndm_500_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-1_xu_xsum_500_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-3_cnn_dailymail_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-3_xsum_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-3_xu_cnndm_500_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Lead-3_xu_xsum_500_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| Llama_P1_16bit_cnn_dailymail_full_summaries.jsonl | Llama | P1 | 1000 | 11490 | 0 | 0.0% |
| Llama_P1_16bit_xsum_full_summaries.jsonl | Llama | P1 | 1000 | 11334 | 1 | 0.0% |
| Llama_P1_16bit_xu_cnndm_500_summaries.jsonl | Llama | P1 | 1000 | 500 | 0 | 0.0% |
| Llama_P1_16bit_xu_xsum_500_summaries.jsonl | Llama | P1 | 1000 | 500 | 0 | 0.0% |
| Llama_P1_4bit_cnn_dailymail_full_summaries.jsonl | Llama | P1 | 1000 | 11490 | 0 | 0.0% |
| Llama_P1_4bit_xsum_full_summaries.jsonl | Llama | P1 | 1000 | 11334 | 1 | 0.0% |
| Llama_P1_4bit_xu_cnndm_500_summaries.jsonl | Llama | P1 | 1000 | 500 | 0 | 0.0% |
| Llama_P1_4bit_xu_xsum_500_summaries.jsonl | Llama | P1 | 1000 | 500 | 0 | 0.0% |
| Llama_P1_8bit_cnn_dailymail_full_summaries.jsonl | Llama | P1 | 1000 | 11490 | 0 | 0.0% |
| Llama_P1_8bit_xsum_full_summaries.jsonl | Llama | P1 | 1000 | 11334 | 3 | 0.0% |
| Llama_P1_8bit_xu_cnndm_500_summaries.jsonl | Llama | P1 | 1000 | 500 | 0 | 0.0% |
| Llama_P1_8bit_xu_xsum_500_summaries.jsonl | Llama | P1 | 1000 | 500 | 0 | 0.0% |
| Llama_P2_16bit_cnn_dailymail_full_summaries.jsonl | Llama | P2 | 1000 | 11490 | 0 | 0.0% |
| Llama_P2_16bit_xsum_full_summaries.jsonl | Llama | P2 | 1000 | 11334 | 2 | 0.0% |
| Llama_P2_16bit_xu_cnndm_500_summaries.jsonl | Llama | P2 | 1000 | 500 | 0 | 0.0% |
| Llama_P2_16bit_xu_xsum_500_summaries.jsonl | Llama | P2 | 1000 | 500 | 0 | 0.0% |
| Llama_P2_4bit_cnn_dailymail_full_summaries.jsonl | Llama | P2 | 1000 | 11490 | 0 | 0.0% |
| Llama_P2_4bit_xsum_full_summaries.jsonl | Llama | P2 | 1000 | 11334 | 0 | 0.0% |
| Llama_P2_4bit_xu_cnndm_500_summaries.jsonl | Llama | P2 | 1000 | 500 | 0 | 0.0% |
| Llama_P2_4bit_xu_xsum_500_summaries.jsonl | Llama | P2 | 1000 | 500 | 0 | 0.0% |
| Llama_P2_8bit_cnn_dailymail_full_summaries.jsonl | Llama | P2 | 1000 | 11490 | 0 | 0.0% |
| Llama_P2_8bit_xsum_full_summaries.jsonl | Llama | P2 | 1000 | 11334 | 3 | 0.0% |
| Llama_P2_8bit_xu_cnndm_500_summaries.jsonl | Llama | P2 | 1000 | 500 | 0 | 0.0% |
| Llama_P2_8bit_xu_xsum_500_summaries.jsonl | Llama | P2 | 1000 | 500 | 0 | 0.0% |
| Llama_P3_16bit_cnn_dailymail_full_summaries.jsonl | Llama | P3 | 1000 | 11490 | 2 | 0.0% |
| Llama_P3_16bit_xsum_full_summaries.jsonl | Llama | P3 | 1000 | 11334 | 3 | 0.0% |
| Llama_P3_16bit_xu_cnndm_500_summaries.jsonl | Llama | P3 | 1000 | 500 | 0 | 0.0% |
| Llama_P3_16bit_xu_xsum_500_summaries.jsonl | Llama | P3 | 1000 | 500 | 0 | 0.0% |
| Llama_P3_4bit_cnn_dailymail_full_summaries.jsonl | Llama | P3 | 1000 | 11490 | 1 | 0.0% |
| Llama_P3_4bit_xsum_full_summaries.jsonl | Llama | P3 | 1000 | 11334 | 8 | 0.1% |
| Llama_P3_4bit_xu_cnndm_500_summaries.jsonl | Llama | P3 | 1000 | 500 | 0 | 0.0% |
| Llama_P3_4bit_xu_xsum_500_summaries.jsonl | Llama | P3 | 1000 | 500 | 0 | 0.0% |
| Llama_P3_8bit_cnn_dailymail_full_summaries.jsonl | Llama | P3 | 1000 | 11490 | 2 | 0.0% |
| Llama_P3_8bit_xsum_full_summaries.jsonl | Llama | P3 | 1000 | 11334 | 2 | 0.0% |
| Llama_P3_8bit_xu_cnndm_500_summaries.jsonl | Llama | P3 | 1000 | 500 | 0 | 0.0% |
| Llama_P3_8bit_xu_xsum_500_summaries.jsonl | Llama | P3 | 1000 | 500 | 0 | 0.0% |
| Phi_P1_16bit_cnn_dailymail_full_summaries.jsonl | Phi | P1 | 1000 | 11490 | 0 | 0.0% |
| Phi_P1_16bit_xsum_full_summaries.jsonl | Phi | P1 | 1000 | 11334 | 0 | 0.0% |
| Phi_P1_16bit_xu_cnndm_500_summaries.jsonl | Phi | P1 | 1000 | 500 | 0 | 0.0% |
| Phi_P1_16bit_xu_xsum_500_summaries.jsonl | Phi | P1 | 1000 | 500 | 0 | 0.0% |
| Phi_P1_4bit_cnn_dailymail_full_summaries.jsonl | Phi | P1 | 1000 | 11490 | 34 | 0.3% |
| Phi_P1_4bit_xsum_full_summaries.jsonl | Phi | P1 | 1000 | 11334 | 20 | 0.2% |
| Phi_P1_4bit_xu_cnndm_500_summaries.jsonl | Phi | P1 | 1000 | 500 | 1 | 0.2% |
| Phi_P1_4bit_xu_xsum_500_summaries.jsonl | Phi | P1 | 1000 | 500 | 0 | 0.0% |
| Phi_P1_8bit_cnn_dailymail_full_summaries.jsonl | Phi | P1 | 1000 | 11490 | 0 | 0.0% |
| Phi_P1_8bit_xsum_full_summaries.jsonl | Phi | P1 | 1000 | 11334 | 0 | 0.0% |
| Phi_P1_8bit_xu_cnndm_500_summaries.jsonl | Phi | P1 | 1000 | 500 | 0 | 0.0% |
| Phi_P1_8bit_xu_xsum_500_summaries.jsonl | Phi | P1 | 1000 | 500 | 0 | 0.0% |
| Phi_P2_16bit_cnn_dailymail_full_summaries.jsonl | Phi | P2 | 1000 | 11490 | 0 | 0.0% |
| Phi_P2_16bit_xsum_full_summaries.jsonl | Phi | P2 | 1000 | 11334 | 0 | 0.0% |
| Phi_P2_16bit_xu_cnndm_500_summaries.jsonl | Phi | P2 | 1000 | 500 | 0 | 0.0% |
| Phi_P2_16bit_xu_xsum_500_summaries.jsonl | Phi | P2 | 1000 | 500 | 0 | 0.0% |
| Phi_P2_4bit_cnn_dailymail_full_summaries.jsonl | Phi | P2 | 1000 | 11490 | 4 | 0.0% |
| Phi_P2_4bit_xsum_full_summaries.jsonl | Phi | P2 | 1000 | 11334 | 3 | 0.0% |
| Phi_P2_4bit_xu_cnndm_500_summaries.jsonl | Phi | P2 | 1000 | 500 | 0 | 0.0% |
| Phi_P2_4bit_xu_xsum_500_summaries.jsonl | Phi | P2 | 1000 | 500 | 0 | 0.0% |
| Phi_P2_8bit_cnn_dailymail_full_summaries.jsonl | Phi | P2 | 1000 | 11490 | 0 | 0.0% |
| Phi_P2_8bit_xsum_full_summaries.jsonl | Phi | P2 | 1000 | 11334 | 0 | 0.0% |
| Phi_P2_8bit_xu_cnndm_500_summaries.jsonl | Phi | P2 | 1000 | 500 | 0 | 0.0% |
| Phi_P2_8bit_xu_xsum_500_summaries.jsonl | Phi | P2 | 1000 | 500 | 0 | 0.0% |
| Phi_P3_16bit_cnn_dailymail_full_summaries.jsonl | Phi | P3 | 1000 | 11490 | 18 | 0.2% |
| Phi_P3_16bit_xsum_full_summaries.jsonl | Phi | P3 | 1000 | 11334 | 40 | 0.4% |
| Phi_P3_16bit_xu_cnndm_500_summaries.jsonl | Phi | P3 | 1000 | 500 | 0 | 0.0% |
| Phi_P3_16bit_xu_xsum_500_summaries.jsonl | Phi | P3 | 1000 | 500 | 3 | 0.6% |
| Phi_P3_4bit_cnn_dailymail_full_summaries.jsonl | Phi | P3 | 1000 | 11490 | 59 | 0.5% |
| Phi_P3_4bit_xsum_full_summaries.jsonl | Phi | P3 | 1000 | 11334 | 63 | 0.6% |
| Phi_P3_4bit_xu_cnndm_500_summaries.jsonl | Phi | P3 | 1000 | 500 | 1 | 0.2% |
| Phi_P3_4bit_xu_xsum_500_summaries.jsonl | Phi | P3 | 1000 | 500 | 3 | 0.6% |
| Phi_P3_8bit_cnn_dailymail_full_summaries.jsonl | Phi | P3 | 1000 | 11490 | 13 | 0.1% |
| Phi_P3_8bit_xsum_full_summaries.jsonl | Phi | P3 | 1000 | 11334 | 34 | 0.3% |
| Phi_P3_8bit_xu_cnndm_500_summaries.jsonl | Phi | P3 | 1000 | 500 | 0 | 0.0% |
| Phi_P3_8bit_xu_xsum_500_summaries.jsonl | Phi | P3 | 1000 | 500 | 0 | 0.0% |
| TFIDF_cnn_dailymail_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TFIDF_xsum_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TFIDF_xu_cnndm_500_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TFIDF_xu_xsum_500_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TextRank_cnn_dailymail_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TextRank_xsum_full_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TextRank_xu_cnndm_500_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| TextRank_xu_xsum_500_summaries.jsonl | — | — | — | — | — | baseline (no cap) |
| **TOTAL** |  |  |  | 428832 | 324 | 0.1% |

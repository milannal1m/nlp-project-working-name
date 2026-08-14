# Traditional-ML Summarization Results

Extractive sentence classifiers (LogReg / NB / XGBoost) with validation-tuned selection length. Compared against the published **Lead-3** bar.

## CNN/DailyMail

| Model | BLEU | ROUGE-L | METEOR | BERTScore-F1 | Gen Len |
|---|---|---|---|---|---|
| _Lead-3 (bar)_ | 0.1150 | 0.2428 | 0.3854 | 0.8691 | — |
| ML-LogReg | 0.1350 | 0.2604 | 0.3899 | 0.8727 | 70.3650 |
| ML-NB | 0.0997 | 0.2262 | 0.3510 | 0.8626 | 80.6863 |
| ML-XGB | 0.1309 | 0.2605 | 0.3934 | 0.8724 | 73.1358 |

**Beats Lead-3 ROUGE-L (0.2428):** ML-LogReg, ML-XGB

## XSum

| Model | BLEU | ROUGE-L | METEOR | BERTScore-F1 | Gen Len |
|---|---|---|---|---|---|
| _Lead-3 (bar)_ | 0.0077 | 0.1155 | 0.2089 | 0.8546 | — |
| ML-LogReg | 0.0136 | 0.1400 | 0.2013 | 0.8586 | 39.0258 |
| ML-NB | 0.0110 | 0.1280 | 0.1940 | 0.8541 | 44.9109 |
| ML-XGB | 0.0124 | 0.1348 | 0.1890 | 0.8584 | 36.4155 |

**Beats Lead-3 ROUGE-L (0.1155):** ML-LogReg, ML-NB, ML-XGB


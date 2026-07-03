# Job-time analysis

Per-article rate and estimated full-run time per `(model, quant, prompt, dataset)`, parsed from `logs/**/sum_*.out`. Timed-out runs contribute a rate from their SLURM wall-clock; they are flagged `timeout` in the status column.

| model | quant | prompt | dataset | runs | status | min/article | est full (h) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama | 16bit | P1 | cnn_dailymail | 1 | ok | 0.0318 | 6.1 |
| Llama | 16bit | P1 | xsum | 1 | ok | 0.0277 | 5.2 |
| Llama | 16bit | P2 | cnn_dailymail | 1 | ok | 0.0356 | 6.8 |
| Llama | 16bit | P2 | xsum | 1 | ok | 0.0328 | 6.2 |
| Llama | 16bit | P3 | cnn_dailymail | 1 | ok | 0.0689 | 13.2 |
| Llama | 16bit | P3 | xsum | 1 | ok | 0.0616 | 11.6 |
| Llama | 4bit | P1 | cnn_dailymail | 1 | ok | 0.0640 | 12.3 |
| Llama | 4bit | P1 | xsum | 1 | ok | 0.0554 | 10.5 |
| Llama | 4bit | P2 | cnn_dailymail | 1 | ok | 0.0678 | 13.0 |
| Llama | 4bit | P2 | xsum | 1 | ok | 0.0651 | 12.3 |
| Llama | 4bit | P3 | cnn_dailymail | 1 | ok | 0.1352 | 25.9 |
| Llama | 4bit | P3 | xsum | 1 | ok | 0.1182 | 22.3 |
| Llama | 8bit | P1 | cnn_dailymail | 1 | ok | 0.1218 | 23.3 |
| Llama | 8bit | P1 | xsum | 1 | ok | 0.1085 | 20.5 |
| Llama | 8bit | P2 | cnn_dailymail | 1 | ok | 0.1336 | 25.6 |
| Llama | 8bit | P2 | xsum | 1 | ok | 0.1207 | 22.8 |
| Llama | 8bit | P3 | cnn_dailymail | 1 | timeout | 0.2544 | 48.7 |
| Llama | 8bit | P3 | xsum | 1 | timeout | 0.2361 | 44.6 |
| Phi | 16bit | P1 | cnn_dailymail | 1 | ok | 0.0321 | 6.1 |
| Phi | 16bit | P1 | xsum | 1 | ok | 0.0292 | 5.5 |
| Phi | 16bit | P2 | cnn_dailymail | 1 | ok | 0.0355 | 6.8 |
| Phi | 16bit | P2 | xsum | 1 | ok | 0.0311 | 5.9 |
| Phi | 16bit | P3 | cnn_dailymail | 1 | ok | 0.0639 | 12.2 |
| Phi | 16bit | P3 | xsum | 1 | ok | 0.0600 | 11.3 |
| Phi | 4bit | P1 | cnn_dailymail | 1 | ok | 0.0449 | 8.6 |
| Phi | 4bit | P1 | xsum | 1 | ok | 0.0425 | 8.0 |
| Phi | 4bit | P2 | cnn_dailymail | 1 | ok | 0.0529 | 10.1 |
| Phi | 4bit | P2 | xsum | 1 | ok | 0.0456 | 8.6 |
| Phi | 4bit | P3 | cnn_dailymail | 1 | ok | 0.0772 | 14.8 |
| Phi | 4bit | P3 | xsum | 1 | ok | 0.0763 | 14.4 |
| Phi | 8bit | P1 | cnn_dailymail | 1 | ok | 0.0810 | 15.5 |
| Phi | 8bit | P1 | xsum | 1 | ok | 0.0754 | 14.2 |
| Phi | 8bit | P2 | cnn_dailymail | 1 | ok | 0.0911 | 17.4 |
| Phi | 8bit | P2 | xsum | 1 | ok | 0.0819 | 15.5 |
| Phi | 8bit | P3 | cnn_dailymail | 1 | ok | 0.1752 | 33.6 |
| Phi | 8bit | P3 | xsum | 1 | ok | 0.1664 | 31.4 |

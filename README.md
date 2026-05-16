# NLP Project Status

This project aims to build a news summarization pipeline based on LLaMA-family models and to implement evaluation metrics according to the requirements in `NLP_Paper.pdf`.

## Completed Work

- Implemented `summary_llama.py` to load a local pretrained model and generate summaries.
- Supported `None`, `4bit`, and `8bit` quantization modes.
- Integrated the following datasets:
  - `cnn_dailymail`
  - `xsum`
  - `newsroom`
  - `news-qa-summarization`
- Added local dataset loading logic with fallback from `datasets.load_dataset` to `load_from_disk`.
- Output generated summaries to the `./summaries` directory in `JSONL` format.
- Added progress logging for dataset name, sample count, sample index, and elapsed time.
- Fixed compatibility for `questions` in `news-qa-summarization` to avoid `list` object attribute errors.

## Current Status

- The script can load the model and generate summaries for different datasets.
- The end-to-end generation flow is working, but the formal evaluation metric implementation is still pending.
- Additional work is required to document the `phi-3` model summary and compare results across quantization precision modes.

## Remaining Work

1. Implement the summary quality evaluation module strictly according to `NLP_Paper.pdf`.
2. Write the experimental summary and performance conclusions for the `phi-3` model.
3. Add comparison results for different quantization precision modes.
4. Improve logging, exception handling, and model loading robustness in `summary_llama.py`.
5. Finalize the generated results and evaluation outputs into a deliverable experiment report.

## Next Priorities

### 1. Evaluation Metric

The next main task is to implement evaluation metrics exactly as required by `NLP_Paper.pdf`.


### 2. `phi-3` Model Summary

Document the `phi-3` experiment with:

- model loading and parameter setup
- qualitative summary quality observations
- runtime resource and speed performance

### 3. Quantization Precision Summary

Add analysis for precision modes:

- `None` (no quantization)
- `4bit`
- `8bit`

Compare:

- summary generation quality
- computational performance
- resource usage

## Example Run Command

```bash
python summary_llama.py --model_name_or_path E:\University_assginment\Ulm\second_semester\NLP\Project\Llama-3.2-3B-Instruct --quantization_method 4bit
```

---

> Current focus: **implement the evaluation metrics required by `NLP_Paper.pdf`, then complete the `phi-3` summary and quantization precision comparisons.**
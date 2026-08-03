"""Self-contained traditional-ML extractive summarization pipeline.

This package is intentionally independent of the repository's shared modules
(``dataset.py``, ``evaluator.py``, ``pipeline_config.py``, ``aggregate.py``): it
ships its own dataset loader (``data.py``) and metric scoring (``evaluate.py``)
so it can be merged to ``main`` as a pure directory addition with no risk of a
merge conflict. It reads the same CNN/DailyMail and XSum splits and writes its
own results under ``traditional_ml/`` only.

Pipeline: train (extract -> hyperparameter search -> fit -> validation
calibration) -> generate -> evaluate -> aggregate.
"""

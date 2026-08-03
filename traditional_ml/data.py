"""Self-contained dataset loading for the traditional-ML pipeline.

Streams CNN/DailyMail and XSum via HuggingFace ``datasets`` and extracts the
(news, reference) fields. This is an independent copy of the shared loader's
behavior (shuffle seed 42, dateline stripping) so the package imports nothing
from the main repo and stays merge-independent.
"""


import re


from datasets import load_dataset


from . import config


def strip_dateline(text: str) -> str:

    """Strip a leading dateline such as ``LONDON (CNN) --`` from an article."""

    prefix = text[:120]

    if " -- " in prefix:

        cleaned = text[text.index(" -- ") + 4:]

        if len(cleaned) > len(text) * 0.5:

            return cleaned.lstrip()


    m = re.match(r"^[A-Za-z][\w .,'-]{0,40}?\([^)]{1,25}\)\s*(?:--\s*)?", prefix)

    if m:

        cleaned = text[m.end():].lstrip()

        if len(cleaned) > len(text) * 0.5:

            return cleaned

    cleaned = re.sub(r"^\([^)]+\)", "", text).lstrip()

    if len(cleaned) > len(text) * 0.5:

        return cleaned

    return text


def _load(dataset: str, split: str, streaming: bool):

    cfg = config.DATASET_CONFIGS[dataset]

    kwargs = {"split": split, "streaming": streaming}

    if "name" in cfg:

        kwargs["name"] = cfg["name"]

    return load_dataset(cfg["path"], **kwargs)


def load_test(dataset: str, sample: int | None = None, seed: int = config.SEED):

    """Stream the test split, shuffled, optionally capped at ``sample``."""

    ds = _load(dataset, "test", streaming=True).shuffle(seed=seed)

    if sample is not None:

        ds = ds.take(sample)

    return ds


def load_split(dataset: str, split: str = "train", sample: int | None = None,

               seed: int = config.SEED, buffer_size: int = 10000):

    """Stream a train or validation split.

    Train is shuffled through a buffer because streaming has no global shuffle;
    validation is deliberately left in dataset order so calibration is reproducible.
    """

    ds = _load(dataset, split, streaming=True)

    if split == "train":

        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    if sample is not None:

        ds = ds.take(sample)

    return ds


def extract_fields(dataset: str, item: dict) -> tuple[str, str]:

    """Return ``(news_text, reference_summary)`` for one record, stripping datelines."""

    text_key, summary_key = config.FIELD_MAP[dataset]

    news_text = item[text_key]

    if dataset in config.DATASETS_WITH_DATELINES:

        news_text = strip_dateline(news_text)

    return news_text, item[summary_key]

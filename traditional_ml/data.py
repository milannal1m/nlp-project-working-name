import re


from datasets import load_dataset


from . import config


def strip_dateline(text: str) -> str:


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


    ds = _load(dataset, "test", streaming=True).shuffle(seed=seed)

    if sample is not None:

        ds = ds.take(sample)

    return ds


def load_split(dataset: str, split: str = "train", sample: int | None = None,

               seed: int = config.SEED, buffer_size: int = 10000):


    ds = _load(dataset, split, streaming=True)

    if split == "train":

        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    if sample is not None:

        ds = ds.take(sample)

    return ds


def extract_fields(dataset: str, item: dict) -> tuple[str, str]:


    text_key, summary_key = config.FIELD_MAP[dataset]

    news_text = item[text_key]

    if dataset in config.DATASETS_WITH_DATELINES:

        news_text = strip_dateline(news_text)

    return news_text, item[summary_key]

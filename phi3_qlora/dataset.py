import re

from datasets import load_dataset


DATASET_CONFIGS = {

    "cnn_dailymail":          {"path": "abisee/cnn_dailymail",              "split": "test",  "name": "3.0.0"},

    "xsum":                   {"path": "EdinburghNLP/xsum",                "split": "test"},


}


_FIELD_MAP = {

    "cnn_dailymail":          ("article",  "highlights"),

    "xsum":                   ("document", "summary"),


}

_DATASETS_WITH_DATELINES = {"cnn_dailymail", "news-qa-summarization"}


def strip_dateline(text: str) -> str:


    prefix = text[:120]

    if ' -- ' in prefix:

        cleaned = text[text.index(' -- ') + 4:]

        if len(cleaned) > len(text) * 0.5:

            return cleaned.lstrip()


    cleaned = re.sub(r'^\([^)]+\)', '', text).lstrip()

    if len(cleaned) > len(text) * 0.5:

        return cleaned


    return text


def load_datasets_streaming(sample: int = None, seed: int = 42) -> dict:

    result = {}

    for name, cfg in DATASET_CONFIGS.items():

        kwargs = {"split": cfg["split"], "streaming": True}

        if "name" in cfg:

            kwargs["name"] = cfg["name"]

        ds = load_dataset(cfg["path"], **kwargs).shuffle(seed=seed)

        if sample is not None:

            ds = ds.take(sample)

        result[name] = ds

        label = f"up to {sample} samples" if sample is not None else "full split"

        print(f"  [ready] {name} ({label}, streaming)", flush=True)

    return result


def load_train_split(name: str, split: str = "train", sample: int | None = None,

                     seed: int = 42, buffer_size: int = 10000):

    cfg = DATASET_CONFIGS[name]

    kwargs = {"split": split, "streaming": True}

    if "name" in cfg:

        kwargs["name"] = cfg["name"]

    ds = load_dataset(cfg["path"], **kwargs)

    if split == "train":

        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    if sample is not None:

        ds = ds.take(sample)

    return ds


def load_pairs(name: str, split: str, limit: int, seed: int = 42) -> list[tuple[str, str]]:

    stream = load_train_split(name, split=split, sample=limit, seed=seed)

    pairs = []

    for item in stream:

        news, ref, _ = extract_fields(name, item)

        if news and ref:

            pairs.append((news, ref))

    return pairs


def extract_fields(dataset_name: str, item: dict) -> tuple[str, str, list | None]:

    if dataset_name not in _FIELD_MAP:

        raise ValueError(f"Unsupported dataset: {dataset_name}")


    text_key, summary_key = _FIELD_MAP[dataset_name]

    qa_pairs = None


    if dataset_name == "news-qa-summarization":

        questions = item.get("questions", [])

        if isinstance(questions, list):

            qa_pairs = [

                {"q": qa["q"], "a": qa["a"]}

                for qa in questions

                if isinstance(qa, dict) and "q" in qa and "a" in qa

            ]

    news_text = item[text_key]

    if dataset_name in _DATASETS_WITH_DATELINES:

        news_text = strip_dateline(news_text)

    return news_text, item[summary_key], qa_pairs


if __name__ == "__main__":

    all_ok = True

    for name, cfg in DATASET_CONFIGS.items():

        try:

            kwargs = {"split": cfg["split"]}

            if "name" in cfg:

                kwargs["name"] = cfg["name"]

            ds = load_dataset(cfg["path"], **kwargs)

            print(f"[OK]   {name} ({len(ds)} samples)")

        except Exception as e:

            print(f"[FAIL] {name}: {e}")

            all_ok = False


    print()

    print("All datasets available." if all_ok else "Some datasets failed — see above.")

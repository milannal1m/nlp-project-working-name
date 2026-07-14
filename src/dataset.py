from __future__ import annotations  # allow `list | None` etc. on Python < 3.10

import json
import os
import re

DATASET_CONFIGS = {
    "cnn_dailymail":          {"path": "abisee/cnn_dailymail",              "split": "test",  "name": "3.0.0"},
    "xsum":                   {"path": "EdinburghNLP/xsum",                "split": "test"},
    # Xu et al.'s released 500-article samples (local files, read verbatim).
    "xu_cnndm":               {"local_path": "xu_et_all_datasets/cnndm_sample_500_0k5_1k5_qwen_summary.jsonl"},
    "xu_xsum":                {"local_path": "xu_et_all_datasets/xsum_sample_500_0k5_1k5_qwen_summary.jsonl"},
    #"newsroom":               {"path": "lil-lab/newsroom",                  "split": "test"},
    #"news-qa-summarization":  {"path": "glnmario/news-qa-summarization",    "split": "train"},
}

_FIELD_MAP = {
    "cnn_dailymail":          ("article",  "highlights"),
    "xsum":                   ("document", "summary"),
    "xu_cnndm":               ("article",  "qwen_reference_summary"),
    "xu_xsum":                ("article",  "qwen_reference_summary"),
    #"newsroom":               ("text",     "summary"),
    #"news-qa-summarization":  ("story",    "summary"),
}
_DATASETS_WITH_DATELINES = {"cnn_dailymail", "news-qa-summarization"}

def strip_dateline(text: str) -> str:
    """
    Remove datelines like '(CNN) --' or 'LONDON (CNN) --' from the start
    of CNN/DailyMail articles. Only strips if ' -- ' appears in the first
    120 characters to avoid cutting real content.
    """
    # Pattern 1: '(CNN) --' or 'LONDON (CNN) --'
    prefix = text[:120]
    if ' -- ' in prefix:
        cleaned = text[text.index(' -- ') + 4:]
        if len(cleaned) > len(text) * 0.5:
            return cleaned.lstrip()
    
    # Pattern 2: '(CNN)The' or '(SOURCE)Word' — no space or dash
    cleaned = re.sub(r'^\([^)]+\)', '', text).lstrip()
    if len(cleaned) > len(text) * 0.5:
        return cleaned

    return text

def load_local_jsonl(path: str, sample: int = None) -> list:
    """Load a local dataset stored as a (pretty-printed) JSON array of records.

    These files hold a fixed, pre-selected sample, so we do NOT shuffle; --sample
    just caps to the first N records if N is smaller than the file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if sample is None else data[:sample]


def load_datasets_streaming(sample: int = None, seed: int = 42, names=None, split: str = None) -> dict:
    """Return {name: iterable-of-records} for the requested datasets.

    HuggingFace datasets are streamed (shuffled, then truncated to `sample`); local
    datasets (those with a `local_path`) are read verbatim. `names` restricts which
    datasets are loaded (default: all configured) so a local-only run never touches
    the network and an HF run never reads the local files. `split` overrides each HF
    dataset's configured split (e.g. "train"); omit it to use the configured default.
    """
    selected = names if names is not None else list(DATASET_CONFIGS.keys())
    result = {}
    for name in selected:
        cfg = DATASET_CONFIGS[name]
        if "local_path" in cfg:
            result[name] = load_local_jsonl(cfg["local_path"], sample)
            label = f"first {sample}" if sample is not None else "all"
            print(f"  [ready] {name} ({label} records, local file)", flush=True)
            continue
        from datasets import load_dataset  # lazy: only needed for HuggingFace sources
        kwargs = {"split": split or cfg["split"], "streaming": True}
        if "name" in cfg:
            kwargs["name"] = cfg["name"]
        ds = load_dataset(cfg["path"], **kwargs).shuffle(seed=seed)
        if sample is not None:
            ds = ds.take(sample)
        result[name] = ds
        label = f"up to {sample} samples" if sample is not None else "full split"
        print(f"  [ready] {name}/{kwargs['split']} ({label}, streaming)", flush=True)
    return result


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
            if "local_path" in cfg:
                if not os.path.exists(cfg["local_path"]):
                    raise FileNotFoundError(cfg["local_path"])
                n = len(load_local_jsonl(cfg["local_path"]))
                print(f"[OK]   {name} ({n} records, local)")
                continue
            from datasets import load_dataset
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

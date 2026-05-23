from datasets import load_dataset

DATASET_CONFIGS = {
    "cnn_dailymail":          {"path": "abisee/cnn_dailymail",              "split": "test[:500]",  "name": "3.0.0"},
    "xsum":                   {"path": "EdinburghNLP/xsum",                "split": "test[:500]"},
    #"newsroom":               {"path": "lil-lab/newsroom",                  "split": "test[:500]"},
    "news-qa-summarization":  {"path": "glnmario/news-qa-summarization",    "split": "train[:500]"},
}

_FIELD_MAP = {
    "cnn_dailymail":          ("article",  "highlights"),
    "xsum":                   ("document", "summary"),
    #"newsroom":               ("text",     "summary"),
    "news-qa-summarization":  ("story",    "summary"),
}


def load_all_datasets() -> dict:
    result = {}
    for name, cfg in DATASET_CONFIGS.items():
        kwargs = {"split": cfg["split"]}
        if "name" in cfg:
            kwargs["name"] = cfg["name"]
        result[name] = load_dataset(cfg["path"], **kwargs)
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

    return item[text_key], item[summary_key], qa_pairs


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

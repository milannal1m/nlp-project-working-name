"""Generate extractive summaries with a trained traditional-ML model.

Loads ``model.joblib`` + ``meta.json`` (for the validation-tuned ``best_k`` and
``use_blocking``), scores each test sentence, selects the top-k with trigram
blocking, and writes JSONL summaries in the standard schema.

    python -m traditional_ml.generate --model logreg
    python -m traditional_ml.generate --model xgb --sample 20
"""


import argparse

import json

import os

import time


import joblib


from . import config, data, features


def _count_records(path: str) -> int:

    if not os.path.exists(path):

        return 0

    with open(path, "r", encoding="utf-8") as f:

        return sum(1 for _ in f)


def _policy_from_meta(meta: dict, dataset: str) -> dict:

    """Resolve the selection policy: new `selection` dict → legacy keys → fallback."""

    sel = meta.get("selection")

    if sel:

        return {k: sel[k] for k in ("mode", "k", "budget", "redundancy", "lambda") if k in sel}

    if meta.get("best_k") is not None:

        return {"mode": "topk", "k": meta["best_k"],

                "redundancy": "block" if meta.get("use_blocking") else "none"}

    return config.FALLBACK_SELECTION[dataset]


def summarize(model, name: str, dataset: str, news: str, policy: dict) -> str:

    sents = features.split_sentences(news, dataset)

    if not sents:

        return ""

    scores = model.predict_proba(features.build_model_matrix(name, sents))[:, 1]

    idx = features.select_sentences_v2(scores, sents, policy)

    return " ".join(sents[i] for i in idx)


def run_model(name: str, dataset: str, sample, summaries_dir: str, models_dir: str,

              seed: int) -> None:

    label = config.LABELS[name]

    out_dir = config.model_dir(models_dir, dataset, name)

    model_path = os.path.join(out_dir, "model.joblib")

    meta_path = os.path.join(out_dir, "meta.json")

    if not os.path.exists(model_path):

        raise FileNotFoundError(f"No trained model at {model_path} — run traditional_ml.train first")


    meta = {}

    if os.path.exists(meta_path):

        with open(meta_path, "r", encoding="utf-8") as f:

            meta = json.load(f)

    policy = _policy_from_meta(meta, dataset)


    os.makedirs(summaries_dir, exist_ok=True)

    out_path = os.path.join(summaries_dir, config.output_filename(label, dataset))

    target = sample if sample is not None else config.FULL_TEST_SIZES[dataset]

    if _count_records(out_path) >= target:

        print(f"[skip] {label}/{dataset}: {out_path} already has >= {target}", flush=True)

        return


    model = joblib.load(model_path)

    print("=" * 80, flush=True)

    print(f"[{label}] {dataset} (policy={policy}, target {target}) -> {out_path}", flush=True)

    start = time.time()

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:

        for idx, item in enumerate(data.load_test(dataset, sample=sample, seed=seed), start=1):

            news, ref = data.extract_fields(dataset, item)

            generated = summarize(model, name, dataset, news, policy)

            f.write(json.dumps({"news": news, "reference_summary": ref,

                                "generated_summary": generated}, ensure_ascii=False) + "\n")

            if idx == 1 or idx % 500 == 0:

                print(f"  sample {idx}/{target}", flush=True)

    print(f"  Done in {(time.time() - start) / 60:.2f} min -> {out_path}", flush=True)


def parse_args():

    p = argparse.ArgumentParser(description="Generate traditional-ML summaries")

    p.add_argument("--model", type=str, default=None, help="logreg | nb | xgb (default: all)")

    p.add_argument("--datasets", nargs="+", default=config.DATASETS)

    p.add_argument("--sample", type=int, default=config.SAMPLE)

    p.add_argument("--summaries_dir", type=str, default=config.SUMMARIES_DIR)

    p.add_argument("--models_dir", type=str, default=config.MODELS_DIR)

    p.add_argument("--seed", type=int, default=config.SEED)

    return p.parse_args()


def main():

    args = parse_args()

    names = [args.model] if args.model else config.MODEL_NAMES

    for name in names:

        for dataset in args.datasets:

            run_model(name, dataset, args.sample, args.summaries_dir, args.models_dir, args.seed)


if __name__ == "__main__":

    main()

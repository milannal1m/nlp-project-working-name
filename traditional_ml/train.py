"""Train the traditional-ML extractive summarizers.

Four resumable phases per dataset:
  1. extract  — stream the full train split, oracle-label sentences, cache
                features to gzip chunks (resumable via a manifest).
  2. search   — hyperparameter search per model on an in-memory subsample.
  3. fit      — fit each model on the full split with the best params
                (out-of-core partial_fit for logreg/nb; full-batch for xgb).
  4. calibrate— tune the inference selection policy (k, trigram blocking) on the
                validation split against the balanced ROUGE-L+METEOR+BLEU composite.

    python -m traditional_ml.train --dataset cnn_dailymail
    python -m traditional_ml.train --dataset xsum --sample 300 --val_docs 60 --hparam_subsample 2000
"""


import argparse

import gzip

import json

import os

import random

import time


import joblib

import numpy as np

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


from . import config, data, features

from .evaluate import MetricScorer


def _chunk_dir(features_dir: str, dataset: str) -> str:

    return os.path.join(features_dir, dataset)


def _manifest_path(features_dir: str, dataset: str) -> str:

    return os.path.join(_chunk_dir(features_dir, dataset), "manifest.json")


def _load_manifest(features_dir: str, dataset: str) -> dict:

    path = _manifest_path(features_dir, dataset)

    if os.path.exists(path):

        with open(path, "r", encoding="utf-8") as f:

            return json.load(f)

    return {"docs_done": 0, "chunks": 0, "pos": 0, "neg": 0, "complete": False, "target": None}


def _save_manifest(features_dir: str, dataset: str, manifest: dict) -> None:

    with open(_manifest_path(features_dir, dataset), "w", encoding="utf-8") as f:

        json.dump(manifest, f, indent=2)


def _chunk_paths(features_dir: str, dataset: str) -> list[str]:

    d = _chunk_dir(features_dir, dataset)

    return sorted(os.path.join(d, fn) for fn in os.listdir(d)

                  if fn.startswith("chunk_") and fn.endswith(".jsonl.gz")) if os.path.isdir(d) else []


def _iter_rows(path: str):

    with gzip.open(path, "rt", encoding="utf-8") as f:

        for line in f:

            yield json.loads(line)


def extract_features(dataset: str, sample, features_dir: str, seed: int) -> dict:

    """Stream the train split and write oracle-labeled feature chunks (resumable)."""

    out_dir = _chunk_dir(features_dir, dataset)

    os.makedirs(out_dir, exist_ok=True)

    target = sample if sample is not None else config.FULL_TRAIN_SIZES[dataset]

    manifest = _load_manifest(features_dir, dataset)


    if manifest.get("complete") and manifest.get("docs_done", 0) >= target:

        print(f"[extract] {dataset}: cache complete ({manifest['docs_done']} docs) — skip", flush=True)

        return manifest


    max_k = config.ORACLE_MAX_K[dataset]

    stream = data.load_split(dataset, "train", sample=None, seed=seed)

    processed = manifest["docs_done"]

    if processed:

        stream = stream.skip(processed)

        print(f"[extract] {dataset}: resuming after {processed} docs", flush=True)


    buffer, since_flush = [], 0

    start = time.time()


    def flush():

        nonlocal buffer, since_flush

        if not buffer:

            return

        path = os.path.join(out_dir, f"chunk_{manifest['chunks']:05d}.jsonl.gz")

        with gzip.open(path, "wt", encoding="utf-8") as f:

            for row in buffer:

                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest["chunks"] += 1

        manifest["docs_done"] = processed

        _save_manifest(features_dir, dataset, manifest)

        buffer, since_flush = [], 0


    for item in stream:

        if processed >= target:

            break

        news, ref = data.extract_fields(dataset, item)

        processed += 1

        since_flush += 1

        sents = features.split_sentences(news, dataset)

        if news and ref and len(sents) >= 2:

            labels = features.oracle_labels(sents, ref, max_k)

            dense = features.dense_features(sents)

            for j, s in enumerate(sents):

                buffer.append({"doc": processed, "s": s, "y": int(labels[j]), "f": dense[j].tolist()})

                manifest["pos" if labels[j] else "neg"] += 1

        if since_flush >= config.CHUNK_DOCS:

            flush()

            print(f"  [extract] {dataset}: {processed}/{target} docs "

                  f"({(time.time() - start) / 60:.1f} min)", flush=True)


    flush()

    manifest["complete"] = True

    manifest["docs_done"] = processed

    manifest["target"] = target

    _save_manifest(features_dir, dataset, manifest)

    print(f"[extract] {dataset}: done — {processed} docs, "

          f"{manifest['pos']}+/{manifest['neg']}- sentences", flush=True)

    return manifest


def _load_rows(dataset: str, features_dir: str, limit=None, shuffle_chunks=False, seed=0,

               with_texts: bool = True):

    """Return (dense ndarray, texts list|None, y ndarray) from the cache, up to ``limit`` rows.

    ``with_texts=False`` skips materializing the sentence strings — used by the XGBoost
    full-split fit (dense-only), which otherwise holds ~8.6M strings in RAM.
    """

    paths = _chunk_paths(features_dir, dataset)

    if shuffle_chunks:

        random.Random(seed).shuffle(paths)

    dense, texts, ys = [], ([] if with_texts else None), []

    n = 0

    for path in paths:

        for row in _iter_rows(path):

            dense.append(row["f"])

            if with_texts:

                texts.append(row["s"])

            ys.append(row["y"])

            n += 1

            if limit is not None and n >= limit:

                return np.asarray(dense, dtype=np.float32), texts, np.asarray(ys, dtype=np.int8)

    return np.asarray(dense, dtype=np.float32), texts, np.asarray(ys, dtype=np.int8)


def _xgb_classifier(seed: int, **params):

    """XGBoost hist-tree classifier (imported lazily so the module loads without it)."""

    from xgboost import XGBClassifier


    return XGBClassifier(

        tree_method="hist",

        eval_metric="logloss",

        random_state=seed,

        n_jobs=config.HPARAM_N_JOBS,

        **params,

    )


def _base_estimator(name: str, seed: int):

    if name == "logreg":

        return SGDClassifier(loss="log_loss", random_state=seed, max_iter=5, tol=1e-3)

    if name == "nb":

        return MultinomialNB()

    if name == "xgb":

        return _xgb_classifier(seed)

    raise ValueError(name)


def hparam_search(dataset: str, name: str, features_dir: str, seed: int) -> dict:

    """Search hyperparameters over the cached feature rows.

    ``config.HPARAM_SUBSAMPLE = None`` means every cached row. The intermediates
    are deleted before the search runs because ``GridSearchCV``/``RandomizedSearchCV``
    fork worker processes, and anything still referenced here is copied into each
    one -- on the full matrix that is enough to exhaust memory.
    """

    dense, texts, y = _load_rows(dataset, features_dir, limit=config.HPARAM_SUBSAMPLE,

                                 shuffle_chunks=True, seed=seed)

    scope = "ALL cached rows" if config.HPARAM_SUBSAMPLE is None else f"{config.HPARAM_SUBSAMPLE} rows"

    print(f"  [search] {name}: {len(y)} rows ({scope}), "

          f"{int(y.sum())} positive", flush=True)

    if len(np.unique(y)) < 2:

        print(f"  [search] {name}: only one class in subsample — using defaults", flush=True)

        return {}

    X = features.build_model_matrix(name, texts, dense=dense)

    del dense, texts

    grid = config.HPARAM_GRIDS[name]

    est = _base_estimator(name, seed)

    common = dict(scoring=config.HPARAM_SCORING, cv=config.HPARAM_CV_FOLDS,

                  n_jobs=config.HPARAM_N_JOBS)

    if name == "xgb":

        search = RandomizedSearchCV(est, grid, n_iter=config.HPARAM_N_ITER,

                                    random_state=seed, **common)

    else:

        search = GridSearchCV(est, grid, **common)

    search.fit(X, y)

    print(f"  [search] {name}: best {config.HPARAM_SCORING}={search.best_score_:.4f} "

          f"params={search.best_params_}", flush=True)

    return {"best_params": search.best_params_, "cv_score": float(search.best_score_)}


def _class_weights(manifest: dict) -> dict:

    pos, neg = max(manifest.get("pos", 0), 1), max(manifest.get("neg", 0), 1)

    total = pos + neg

    return {0: total / (2 * neg), 1: total / (2 * pos)}


def fit_full(dataset: str, name: str, features_dir: str, manifest: dict,

             best_params: dict, seed: int, epochs: int = 2):

    weights = _class_weights(manifest)

    if name == "xgb":

        dense, _, y = _load_rows(dataset, features_dir, with_texts=False)

        model = _xgb_classifier(seed, **best_params)

        sw = np.where(y == 1, weights[1], weights[0])

        model.fit(dense, y, sample_weight=sw)

        return model


    if name == "logreg":

        model = SGDClassifier(loss="log_loss", random_state=seed, **best_params)

    else:

        model = MultinomialNB(**best_params)

    paths = _chunk_paths(features_dir, dataset)

    for epoch in range(epochs):

        order = paths[:]

        random.Random(seed + epoch).shuffle(order)

        for path in order:

            dense, texts, y = [], [], []

            for row in _iter_rows(path):

                dense.append(row["f"]); texts.append(row["s"]); y.append(row["y"])

            if not y:

                continue

            dense = np.asarray(dense, dtype=np.float32)

            y = np.asarray(y, dtype=np.int8)

            X = features.build_model_matrix(name, texts, dense=dense)

            sw = np.where(y == 1, weights[1], weights[0])

            model.partial_fit(X, y, classes=[0, 1], sample_weight=sw)

    return model


def _score_doc(model, name: str, sentences: list[str]) -> np.ndarray:

    X = features.build_model_matrix(name, sentences)

    return model.predict_proba(X)[:, 1]


def _legacy_from_policy(policy: dict, dataset: str) -> tuple[int, bool]:

    """Back-compat best_k/use_blocking for a policy (budget modes fall back)."""

    if policy["mode"] == "topk":

        return policy["k"], policy.get("redundancy") == "block"

    fb = config.FALLBACK[dataset]

    return fb["k"], fb["blocking"]


def calibrate(dataset: str, name: str, model, val_docs, seed: int,

              scorer: MetricScorer) -> dict:


    print(f"  [calibrate] {name}/{dataset}: scoring "

          f"{'FULL validation split' if val_docs is None else f'{val_docs} val docs'}", flush=True)

    docs = []

    for item in data.load_split(dataset, "validation", sample=val_docs, seed=seed):

        news, ref = data.extract_fields(dataset, item)

        sents = features.split_sentences(news, dataset)

        if news and ref and sents:

            docs.append((sents, ref, _score_doc(model, name, sents)))

    if not docs:

        fb = config.FALLBACK_SELECTION[dataset]

        return {"selection": {**fb, "val_composite": None}, "best_k": fb.get("k"),

                "use_blocking": fb.get("redundancy") == "block", "val_composite": None,

                "note": "no validation docs"}


    policies = config.selection_policy_grid(dataset)

    rows = []

    for policy in policies:

        gen, refs = [], []

        for sents, ref, scores in docs:

            idx = features.select_sentences_v2(scores, sents, policy)

            gen.append(" ".join(sents[i] for i in idx))

            refs.append(ref)

        rows.append(scorer.lexical(gen, refs))


    comp = np.zeros(len(rows))

    for key in config.COMPOSITE_METRICS:

        vals = np.array([r[key] for r in rows])

        lo, hi = vals.min(), vals.max()

        comp += (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)

    comp /= len(config.COMPOSITE_METRICS)

    best = int(np.argmax(comp))

    policy, metrics = policies[best], rows[best]

    best_k, best_block = _legacy_from_policy(policy, dataset)

    selection = {**policy, "val_composite": float(comp[best]), "val_metrics": metrics}

    print(f"  [calibrate] {name}/{dataset}: best {policy} "

          f"(rougeL={metrics['rougeL']:.4f} meteor={metrics['meteor']:.4f} "

          f"bleu={metrics['bleu']:.4f})", flush=True)

    return {"selection": selection, "best_k": best_k, "use_blocking": best_block,

            "val_composite": float(comp[best]), "val_metrics": metrics}


def train_one(dataset: str, names: list[str], sample, features_dir: str, models_dir: str,

              val_docs: int, seed: int, force: bool, skip_hparam: bool, skip_calibration: bool):

    manifest = extract_features(dataset, sample, features_dir, seed)

    scorer = MetricScorer() if not skip_calibration else None


    for name in names:

        out_dir = config.model_dir(models_dir, dataset, name)

        model_path = os.path.join(out_dir, "model.joblib")

        if os.path.exists(model_path) and not force:

            print(f"[skip] {name}/{dataset}: model exists (use --force to retrain)", flush=True)

            continue

        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()


        search = {} if skip_hparam else hparam_search(dataset, name, features_dir, seed)

        best_params = search.get("best_params", {})

        model = fit_full(dataset, name, features_dir, manifest, best_params, seed)


        if skip_calibration:

            fb = config.FALLBACK_SELECTION[dataset]

            calib = {"selection": {**fb, "val_composite": None},

                     "best_k": fb.get("k"), "use_blocking": fb.get("redundancy") == "block",

                     "val_composite": None}

        else:

            calib = calibrate(dataset, name, model, val_docs, seed, scorer)


        joblib.dump(model, model_path)

        meta = {

            "label": config.LABELS[name], "name": name, "dataset": dataset,

            "n_docs": manifest["docs_done"], "n_sentences": manifest["pos"] + manifest["neg"],

            "pos_rate": manifest["pos"] / max(manifest["pos"] + manifest["neg"], 1),

            "splitter": features.SPLITTER,

            "cv_score": search.get("cv_score"), "best_params": best_params,

            "selection": calib.get("selection"),

            "best_k": calib["best_k"], "use_blocking": calib["use_blocking"],

            "val_composite": calib.get("val_composite"), "val_metrics": calib.get("val_metrics"),

            "train_minutes": round((time.time() - t0) / 60, 2),

        }

        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:

            json.dump(meta, f, indent=2, ensure_ascii=False)

        sel = calib.get("selection") or {}

        print(f"[ok] {name}/{dataset} -> {model_path} "

              f"(selection={ {k: sel[k] for k in ('mode','k','budget','redundancy','lambda') if k in sel} }, "

              f"splitter={features.SPLITTER}, {meta['train_minutes']} min)", flush=True)


def parse_args():

    p = argparse.ArgumentParser(description="Train traditional-ML extractive summarizers")

    p.add_argument("--dataset", type=str, default=None,

                   help="cnn_dailymail | xsum | (default from $SLURM_ARRAY_TASK_ID or all)")

    p.add_argument("--models", nargs="+", default=config.MODEL_NAMES)

    p.add_argument("--sample", type=int, default=config.SAMPLE)

    p.add_argument("--val_docs", type=int, default=config.VAL_DOCS,

                   help="Validation docs for policy calibration; 0 means the FULL split")

    p.add_argument("--features_dir", type=str, default=config.FEATURES_DIR)

    p.add_argument("--models_dir", type=str, default=config.MODELS_DIR)

    p.add_argument("--seed", type=int, default=config.SEED)

    p.add_argument("--hparam_subsample", type=int, default=None,

                   help="override config.HPARAM_SUBSAMPLE (for quick local runs)")

    p.add_argument("--force", action="store_true")

    p.add_argument("--skip_hparam", action="store_true")

    p.add_argument("--skip_calibration", action="store_true")

    return p.parse_args()


def main():

    """Entry point.

    ``--sample 0``, ``--val_docs 0`` and ``--hparam_subsample 0`` all mean "use
    everything", matching the ``None`` defaults in :mod:`traditional_ml.config`.
    """

    args = parse_args()

    if args.hparam_subsample is not None:


        config.HPARAM_SUBSAMPLE = args.hparam_subsample or None

    val_docs = args.val_docs or None


    if args.dataset:

        datasets = [args.dataset]

    else:

        task = os.environ.get("SLURM_ARRAY_TASK_ID")

        datasets = [config.DATASETS[int(task)]] if task is not None else config.DATASETS


    for dataset in datasets:

        print(f"=== train {dataset} ({', '.join(args.models)}) ===", flush=True)

        train_one(dataset, args.models, args.sample, args.features_dir, args.models_dir,

                  val_docs, args.seed, args.force, args.skip_hparam, args.skip_calibration)


if __name__ == "__main__":

    main()

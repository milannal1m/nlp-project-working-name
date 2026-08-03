"""Self-contained metric scoring for the traditional-ML pipeline.

Wraps HuggingFace ``evaluate`` (BLEU / ROUGE-L / METEOR / optional BERTScore)
with the same key names the shared evaluator uses, so numbers are directly
comparable to the main ``results/results.csv``. Also exposes ``MetricScorer``
for the training-time validation calibration (lexical metrics only).

    python -m traditional_ml.evaluate --models logreg --datasets xsum --no_bertscore
"""


import argparse

import json

import os

import statistics


from . import config

from .features import sent_tokenize


class MetricScorer:

    """Lazily loads HF metric backends once and reuses them."""


    def __init__(self):

        self._bleu = None

        self._rouge = None

        self._meteor = None

        self._bertscore = None


    def _load_lexical(self):

        if self._bleu is None:

            import evaluate

            self._bleu = evaluate.load("bleu")

            self._rouge = evaluate.load("rouge")

            self._meteor = evaluate.load("meteor")


    def _safe(self, fn, default=0.0):

        """Run ``fn``; on failure report loudly and fall back to ``default``.

        A silent fallback here is dangerous: METEOR needs the nltk ``wordnet``
        and ``omw-1.4`` corpora, and without them every score would read as a
        plausible 0.0 in the published table and in the calibration composite.
        """

        try:

            return fn()

        except Exception as e:

            print(f"  [metric] FAILED ({type(e).__name__}: {e}) -> {default}", flush=True)

            return default


    def lexical(self, generated: list[str], references: list[str]) -> dict:

        """BLEU / ROUGE-L / METEOR — used by validation calibration."""

        self._load_lexical()


        preds = [g if g and g.strip() else " " for g in generated]

        bleu = self._safe(lambda: self._bleu.compute(predictions=preds, references=references)["bleu"])

        rougeL = self._safe(lambda: self._rouge.compute(predictions=preds, references=references)["rougeL"])

        meteor = self._safe(lambda: self._meteor.compute(predictions=preds, references=references)["meteor"])

        return {"bleu": bleu, "rougeL": rougeL, "meteor": meteor}


    def bertscore_f1(self, generated: list[str], references: list[str]):

        """Mean BERTScore-F1, or (None, None) if the backend is unavailable."""

        try:

            if self._bertscore is None:

                import evaluate

                self._bertscore = evaluate.load("bertscore")

            preds = [g if g and g.strip() else " " for g in generated]

            f1 = self._bertscore.compute(predictions=preds, references=references, lang="en")["f1"]

            mean = statistics.mean(f1)

            std = statistics.stdev(f1) if len(f1) > 1 else 0.0

            return mean, std

        except Exception as e:

            print(f"  [bertscore] unavailable, skipping: {e}", flush=True)

            return None, None


def _word_len(text: str) -> int:

    return len(text.split())


def score_file(path: str, label: str, dataset: str, scorer: MetricScorer,

               with_bertscore: bool = True) -> dict:

    """Score one summaries JSONL file into a metrics dict (+ length stats)."""

    generated, references, news = [], [], []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            d = json.loads(line)

            generated.append(d["generated_summary"])

            references.append(d["reference_summary"])

            news.append(d["news"])


    result = {"label": label, "dataset": dataset, "file": os.path.basename(path),

              "num_samples": len(generated)}

    result.update(scorer.lexical(generated, references))


    if with_bertscore:

        mean, std = scorer.bertscore_f1(generated, references)

        result["bertscore_f1"] = mean

        result["bertscore_f1_std"] = std

    else:

        result["bertscore_f1"] = None

        result["bertscore_f1_std"] = None


    gen_lens = [_word_len(g) for g in generated]

    news_lens = [_word_len(n) for n in news]

    result["avg_news_len"] = statistics.mean(news_lens) if news_lens else 0.0

    result["avg_ref_len"] = statistics.mean(_word_len(r) for r in references) if references else 0.0

    result["avg_gen_len"] = statistics.mean(gen_lens) if gen_lens else 0.0

    result["std_gen_len"] = statistics.stdev(gen_lens) if len(gen_lens) > 1 else 0.0

    result["avg_gen_sents"] = statistics.mean(len(sent_tokenize(g)) for g in generated) if generated else 0.0

    comp = [g / n for g, n in zip(gen_lens, news_lens) if n > 0]

    result["avg_compression"] = statistics.mean(comp) if comp else 0.0

    return result


def main() -> None:

    parser = argparse.ArgumentParser(description="Score traditional-ML summaries")

    parser.add_argument("--summaries_dir", type=str, default=config.SUMMARIES_DIR)

    parser.add_argument("--metrics_dir", type=str, default=config.METRICS_DIR)

    parser.add_argument("--datasets", nargs="+", default=config.DATASETS)

    parser.add_argument("--models", nargs="+", default=config.MODEL_NAMES)

    parser.add_argument("--no_bertscore", action="store_true")

    parser.add_argument("--skip_existing", action="store_true",

                        help="Leave already-scored metric JSONs alone (safe rerun after a kill)")

    args = parser.parse_args()


    os.makedirs(args.metrics_dir, exist_ok=True)

    scorer = MetricScorer()

    for name in args.models:

        label = config.LABELS[name]

        for dataset in args.datasets:

            path = os.path.join(args.summaries_dir, config.output_filename(label, dataset))

            out_path = os.path.join(args.metrics_dir, f"{label}__{dataset}.json")

            if args.skip_existing and os.path.exists(out_path):

                print(f"[skip] {label}/{dataset}: already scored -> {out_path}", flush=True)

                continue

            if not os.path.exists(path):

                print(f"[skip] {label}/{dataset}: no summary file — generate first", flush=True)

                continue

            result = score_file(path, label, dataset, scorer, with_bertscore=not args.no_bertscore)

            with open(out_path, "w", encoding="utf-8") as f:

                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[ok] {label}/{dataset}: rougeL={result['rougeL']:.4f} "

                  f"meteor={result['meteor']:.4f} bleu={result['bleu']:.4f} -> {out_path}", flush=True)


if __name__ == "__main__":

    main()

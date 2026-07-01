import csv
import json
import evaluate
import os
import logging
import random
import statistics

class Evaluator:
    def __init__(self):
        self.bleu = None
        self.rouge = None
        self.meteor = None
        self.bertscore = None

    def _load_metrics(self):
        if self.bleu is None:
            self.bleu = evaluate.load("bleu")
            self.rouge = evaluate.load("rouge")
            self.meteor = evaluate.load("meteor")
            self.bertscore = evaluate.load("bertscore")

    def evaluate_metrics(self, file_path):
        """Calculates BLEU, ROUGE, METEOR and BERTScore against the reference summary."""
        self._load_metrics()
        generated_summaries = []
        reference_summaries = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                generated_summaries.append(data['generated_summary'])
                reference_summaries.append(data['reference_summary'])

        bleu_score = self.bleu.compute(predictions=generated_summaries, references=reference_summaries)
        rouge_score = self.rouge.compute(predictions=generated_summaries, references=reference_summaries)
        meteor_score = self.meteor.compute(predictions=generated_summaries, references=reference_summaries)
        bert_score = self.bertscore.compute(
            predictions=generated_summaries,
            references=reference_summaries,
            lang="en",
            rescale_with_baseline=True,  # spread raw ~0.85 scores into an interpretable range
        )

        return {
            "bleu": bleu_score['bleu'],
            "rougeL": rouge_score['rougeL'],
            "meteor": meteor_score['meteor'],
            "bertscore_f1": statistics.mean(bert_score['f1']),
            "bertscore_f1_std": statistics.stdev(bert_score['f1']) if len(bert_score['f1']) > 1 else 0.0,
        }

    def evaluate_qa(self, file_path):  # Highly recommended to call only in a cluster environment.
        """Calculates factual consistency using the dual-context QA pipeline."""
        try:
            from qafacteval import QAFactEval
        except ImportError:
            return {"qa_eval": None, "error": "qafacteval not installed"}

        kwargs = {"model_folder": "models/qafacteval", "device": "cuda"}
        qa_evaluator = QAFactEval(**kwargs)

        original_texts = []
        generated_summaries = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                original_texts.append(data.get('news', ''))
                generated_summaries.append(data.get('generated_summary', ''))

        results = qa_evaluator.score_batch_qg(
            inputs=original_texts,
            predictions=generated_summaries,
            return_qa_pairs=True
        )

        final_scores = [res[0]['qa-eval']['lerc_quac'] for res in results]

        if final_scores:
            return {
                "qa_eval": statistics.mean(final_scores),
                "qa_eval_std": statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0,
            }
        return {"qa_eval": None, "qa_eval_std": None}

    @staticmethod
    def write_sanity_check(jsonl_files, out_path, n=5, seed=42):
        """Write a human-readable Markdown spot-check of generated summaries.

        For each .jsonl in `jsonl_files`, writes the filename as a `##` heading and
        each sampled generated summary under a `###` heading. The SAME `n` record
        indices (chosen once with `seed`) are used for every file, so you can compare
        the exact same articles across all systems side by side.
        """
        files = sorted(jsonl_files)
        indices = None
        lines = ["# Sanity check — generated summaries", ""]
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

            # Pick the shared indices once, from the first (non-empty) file.
            if indices is None and records:
                k = min(n, len(records))
                indices = sorted(random.Random(seed).sample(range(len(records)), k))

            lines.append(f"## {os.path.basename(file_path)}")
            lines.append("")
            for i in (indices or []):
                if i < len(records):
                    summary = records[i].get("generated_summary", "").strip()
                    lines.append(f"### Summary {i}")
                    lines.append("")
                    lines.append(summary)
                    lines.append("")

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Sanity check written to {out_path}", flush=True)

    # Columns written to the CSV, in order.
    CSV_FIELDS = [
        "file",
        "bleu",
        "rougeL",
        "meteor",
        "bertscore_f1",
        "bertscore_f1_std",
        "qa_eval",
        "qa_eval_std",
        "error",
    ]

    def _append_csv(self, csv_path, file_name, all_metrics):
        """Appends one row of metrics to the CSV, writing a header if the file is new."""
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.CSV_FIELDS, extrasaction="ignore", restval=""
            )
            if write_header:
                writer.writeheader()
            writer.writerow({"file": file_name, **all_metrics})

    def run_and_log(self, file_path, log_path="evaluation.log", csv_path=None):
        """Runs all metrics, appends results to the log, and stores them as CSV."""
        if csv_path is None:
            csv_path = os.path.splitext(log_path)[0] + ".csv"

        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter('%(asctime)s  %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        file_name = os.path.basename(file_path)
        logger.info(f"=== Evaluation: {file_name} ===")

        all_metrics = {}
        all_metrics.update(self.evaluate_metrics(file_path))
        all_metrics.update(self.evaluate_qa(file_path))

        logged = set()
        for key, value in all_metrics.items():
            if key in logged or key.endswith("_std"):
                continue
            std = all_metrics.get(f"{key}_std")
            if std is not None:
                logger.info(f"  {key}: {value:.4f} ± {std:.4f}")
            elif isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
            logged.add(key)

        logger.info("")

        self._append_csv(csv_path, file_name, all_metrics)
        return all_metrics

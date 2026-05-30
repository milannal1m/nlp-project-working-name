import json
import evaluate
import os
import logging
from summac.model_summac import SummaCConv

class Evaluator:
    def __init__(self):
        self.bleu = None
        self.rouge = None
        self.meteor = None
        self.bertscore = None
        self.summac_model = None

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
        bert_score = self.bertscore.compute(predictions=generated_summaries, references=reference_summaries, lang="en")
        avg_bert_f1 = sum(bert_score['f1']) / len(bert_score['f1'])

        return {
            "bleu": bleu_score['bleu'],
            "rougeL": rouge_score['rougeL'],
            "meteor": meteor_score['meteor'],
            "bertscore_f1": avg_bert_f1,
        }

    def evaluate_summac(self, file_path):
        """Calculates factual consistency against the original news text using SummaC."""
        if self.summac_model is None:
            self.summac_model = SummaCConv(models=["vitc"], bins="percentile", granularity="sentence", device="cpu")

        original_text = []
        generated_summaries = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                original_text.append(data['news'])
                generated_summaries.append(data['generated_summary'])

        results = self.summac_model.score(original_text, generated_summaries)
        avg_score = sum(results["scores"]) / len(results["scores"])

        return {"summac": avg_score}

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
            return {"qa_eval": sum(final_scores) / len(final_scores)}
        return {"qa_eval": None}

    def run_and_log(self, file_path, log_path="evaluation.log"):
        """Runs all metrics and appends results to evaluation.log."""
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
        all_metrics.update(self.evaluate_summac(file_path))
        all_metrics.update(self.evaluate_qa(file_path))

        for key, value in all_metrics.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        logger.info("")
        return all_metrics

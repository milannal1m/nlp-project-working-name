import json
import evaluate
import os
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
            print("Loading standard Hugging face metrics...")
            self.bleu = evaluate.load("bleu")
            self.rouge = evaluate.load("rouge")
            self.meteor = evaluate.load("meteor")
            self.bertscore = evaluate.load("bertscore")

    def evaluate_metrics(self, file_path):
        """Calculates BLEU, ROUGE, METEOR and BERTScore against the reference summary."""
        self._load_metrics()
        generated_summaries = []
        reference_summaries = []
        
        print(f"\n--- Evaluating Standard Metrics for {os.path.basename(file_path)} ---")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                generated_summaries.append(data['generated_summary'])
                reference_summaries.append(data['reference_summary'])

        print("Computing scores...")
        bleu_score = self.bleu.compute(predictions=generated_summaries, references=reference_summaries)
        rouge_score = self.rouge.compute(predictions=generated_summaries, references=reference_summaries)
        meteor_score = self.meteor.compute(predictions=generated_summaries, references=reference_summaries)

        # Specifying language for BERT
        bert_score = self.bertscore.compute(predictions=generated_summaries, references=reference_summaries, lang="en")
        avg_bert_f1 = sum(bert_score['f1']) / len(bert_score['f1'])

        print(f"BLEU:        {bleu_score['bleu']:.4f}")
        print(f"ROUGE-L:     {rouge_score['rougeL']:.4f}")
        print(f"METEOR:      {meteor_score['meteor']:.4f}")
        print(f"BERTScore F1:{avg_bert_f1:.4f}\n")
    
    def evaluate_summac(self, file_path):
        """Calculates factual consistency against the original news text using SummaC"""
        if self.summac_model is None:
            print("Loading SummaC NLI model...")
            self.summac_model = SummaCConv(models=["vitc"], bins="percentile", granularity="sentence", device="cpu")

        original_text = []
        generated_summaries = []

        print(f"\n--- Evaluating SummaC for {os.path.basename(file_path)} ---")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                original_text.append(data['news'])
                generated_summaries.append(data['generated_summary'])

        print("Computing NLI scores...")
        results = self.summac_model.score(original_text, generated_summaries)
        avg_score = sum(results["scores"]) / len(results["scores"])

        print(f"Average SummaC Score: {avg_score:.4f}\n")

    def evaluate_qa(self, file_path): #Highly recommended to call only in a cluster environment as it would likely crash any local machine.
        """Calculates factual consistency using the dual-context QA pipeline."""

        file_name = os.path.basename(file_path)
        print(f"\n--- Evaluating QA Factuality for {file_name} ---")
        try:
            from qafacteval import QAFactEval
        except ImportError:
            print("ERROR: 'qafacteval' library not found. Make sure it's installed on the cluster.")
            return

        print(f"Initializing QAFactEval pipeline (requires CUDA for full execution)...")
        # Handles question generation, answering and LERC scoring
        kwargs = {"model_folder": "models/qafacteval", "device": "cuda"}
        qa_evaluator = QAFactEval(**kwargs)
        
        original_texts = []
        generated_summaries = []

        print(f"Reading {file_name} and preparing QA pipeline...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                original_texts.append(data.get('news', ''))
                generated_summaries.append(data.get('generated_summary', ''))
        print(f"Computing QA-based factuality for {len(generated_summaries)} summaries...")

        #This generates questions, answers them using the source and the summary and compares the factual overlap
        results = qa_evaluator.score_batch_qg(
            inputs=original_texts,
            predictions=generated_summaries,
            return_qa_pairs=True
        )

        #This extracts final LERC scores
        final_scores = [res[0]['qa-eval']['lerc_quac'] for res in results]

        if final_scores:
            avg_score = sum(final_scores) / len(final_scores)
            print(f"Average QA-Eval Factual Consistency Score: {avg_score:.4f}\n")
        else:
            print("No QA scores were generated. \n")


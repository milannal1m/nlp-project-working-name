import json
import os
import re
import csv
import logging
import statistics
from collections import Counter
from pathlib import Path


# ===========================================================================
# Evaluator — heavy metrics (requires evaluate, summac, torch, etc.)
# ===========================================================================

class Evaluator:
    def __init__(self):
        self.bleu = None
        self.rouge = None
        self.meteor = None
        self.bertscore = None
        self.summac_model = None

    def _load_metrics(self):
        if self.bleu is None:
            import evaluate
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

        return {
            "bleu": bleu_score['bleu'],
            "rouge1": rouge_score['rouge1'],
            "rouge2": rouge_score['rouge2'],
            "rougeL": rouge_score['rougeL'],
            "meteor": meteor_score['meteor'],
            "bertscore_f1": statistics.mean(bert_score['f1']),
            "bertscore_f1_std": statistics.stdev(bert_score['f1']),
        }

    def evaluate_summac(self, file_path):
        """Calculates factual consistency against the original news text using SummaC."""
        if self.summac_model is None:
            import torch
            from summac.model_summac import SummaCConv
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.summac_model = SummaCConv(models=["vitc"], bins="percentile", granularity="sentence", device=device)

        original_text = []
        generated_summaries = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                original_text.append(data['news'])
                generated_summaries.append(data['generated_summary'])

        results = self.summac_model.score(original_text, generated_summaries)
        scores = results["scores"]

        return {
            "summac": statistics.mean(scores),
            "summac_std": statistics.stdev(scores),
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
                "qa_eval_std": statistics.stdev(final_scores),
            }
        return {"qa_eval": None, "qa_eval_std": None}

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
        return all_metrics


# ===========================================================================
# Lightweight analysis — no ML dependencies, runs anywhere
# ===========================================================================

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def sent_count(text: str) -> int:
    """Rough sentence count."""
    return max(len(re.split(r"[.!?]+", text.strip())) - 1, 1)


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(reference: str, hypothesis: str, n: int) -> dict:
    """Compute ROUGE-N precision, recall, F1."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    ref_ng = ngrams(ref_tokens, n)
    hyp_ng = ngrams(hyp_tokens, n)
    overlap = sum((ref_ng & hyp_ng).values())
    prec = overlap / max(sum(hyp_ng.values()), 1)
    rec = overlap / max(sum(ref_ng.values()), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1}


def rouge_l(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-L via LCS."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    m, n_ = len(ref_tokens), len(hyp_tokens)
    # LCS length via DP
    prev = [0] * (n_ + 1)
    for i in range(1, m + 1):
        curr = [0] * (n_ + 1)
        for j in range(1, n_ + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n_]
    prec = lcs / max(n_, 1)
    rec = lcs / max(m, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1}


def analyze_file(filepath: str) -> dict:
    """Return aggregate metrics for one JSONL file."""
    news_lens = []
    ref_lens = []
    gen_lens = []
    ref_sents = []
    gen_sents = []
    compression_ratios = []
    r1_scores = []
    r2_scores = []
    rl_scores = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            news = rec.get("news", "")
            ref = rec.get("reference_summary", "")
            gen = rec.get("generated_summary", "")

            news_tok = tokenize(news)
            ref_tok = tokenize(ref)
            gen_tok = tokenize(gen)

            news_lens.append(len(news_tok))
            ref_lens.append(len(ref_tok))
            gen_lens.append(len(gen_tok))
            ref_sents.append(sent_count(ref))
            gen_sents.append(sent_count(gen))

            if len(news_tok) > 0:
                compression_ratios.append(len(gen_tok) / len(news_tok))

            r1 = rouge_n(ref, gen, 1)
            r2 = rouge_n(ref, gen, 2)
            rl = rouge_l(ref, gen)
            r1_scores.append(r1["f1"])
            r2_scores.append(r2["f1"])
            rl_scores.append(rl["f1"])

    n = len(news_lens)

    def safe_mean(lst):
        return statistics.mean(lst) if lst else 0.0

    def safe_stdev(lst):
        return statistics.stdev(lst) if len(lst) > 1 else 0.0

    return {
        "file": os.path.basename(filepath),
        "num_samples": n,
        # Length stats
        "avg_news_len": safe_mean(news_lens),
        "avg_ref_len": safe_mean(ref_lens),
        "avg_gen_len": safe_mean(gen_lens),
        "std_gen_len": safe_stdev(gen_lens),
        "avg_ref_sents": safe_mean(ref_sents),
        "avg_gen_sents": safe_mean(gen_sents),
        "avg_compression": safe_mean(compression_ratios),
        # ROUGE scores
        "rouge1_f1": safe_mean(r1_scores),
        "rouge1_f1_std": safe_stdev(r1_scores),
        "rouge2_f1": safe_mean(r2_scores),
        "rouge2_f1_std": safe_stdev(r2_scores),
        "rougeL_f1": safe_mean(rl_scores),
        "rougeL_f1_std": safe_stdev(rl_scores),
    }


# ===========================================================================
# Report generation — Markdown & CSV
# ===========================================================================

def _model_name(filename: str) -> str:
    name = filename.replace("_summaries.jsonl", "")
    for suffix in ["_cnn_dailymail", "_xsum"]:
        name = name.replace(suffix, "")
    return name


def write_markdown(all_results: list[dict], path: Path):
    """Write a Markdown analysis report with ROUGE-1, ROUGE-2, ROUGE-L."""
    cnn_results = [r for r in all_results if "cnn_dailymail" in r["file"]]
    xsum_results = [r for r in all_results if "xsum" in r["file"]]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Summarization Analysis Results\n\n")
        f.write(f"**Generated on**: Analysis of `summaries/` folder  \n")
        f.write(f"**Total files analyzed**: {len(all_results)}  \n")
        f.write(f"**Models**: {', '.join(sorted(set(_model_name(r['file']) for r in all_results)))}  \n")
        f.write(f"**Datasets**: CNN/DailyMail, XSum  \n\n")
        f.write("---\n\n")

        # Overview table
        f.write("## Overview — ROUGE Scores\n\n")
        f.write("| File | Samples | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |\n")
        f.write("|------|--------:|-----------:|-----------:|-----------:|\n")
        for r in all_results:
            f.write(
                f"| {r['file']} | {r['num_samples']} "
                f"| {r['rouge1_f1']:.4f} ± {r['rouge1_f1_std']:.4f} "
                f"| {r['rouge2_f1']:.4f} ± {r['rouge2_f1_std']:.4f} "
                f"| {r['rougeL_f1']:.4f} ± {r['rougeL_f1_std']:.4f} |\n"
            )
        f.write("\n---\n\n")

        # Length / compression table
        f.write("## Summary Length & Compression\n\n")
        f.write("| File | Avg News Len | Avg Ref Len | Avg Gen Len (± std) | Avg Gen Sents | Compression Ratio |\n")
        f.write("|------|------------:|------------:|--------------------:|--------------:|------------------:|\n")
        for r in all_results:
            f.write(
                f"| {r['file']} "
                f"| {r['avg_news_len']:.1f} "
                f"| {r['avg_ref_len']:.1f} "
                f"| {r['avg_gen_len']:.1f} ± {r['std_gen_len']:.1f} "
                f"| {r['avg_gen_sents']:.1f} "
                f"| {r['avg_compression']:.4f} |\n"
            )
        f.write("\n---\n\n")

        # Per-dataset breakdowns
        for ds_name, ds_results in [("CNN/DailyMail", cnn_results), ("XSum", xsum_results)]:
            f.write(f"## {ds_name} — Model Comparison\n\n")
            f.write("| Model | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Avg Gen Len | Compression |\n")
            f.write("|-------|----------:|-----------:|-----------:|------------:|------------:|\n")
            # Sort by ROUGE-L descending
            for r in sorted(ds_results, key=lambda x: x["rougeL_f1"], reverse=True):
                mn = _model_name(r["file"])
                f.write(
                    f"| {mn} "
                    f"| {r['rouge1_f1']:.4f} "
                    f"| {r['rouge2_f1']:.4f} "
                    f"| {r['rougeL_f1']:.4f} "
                    f"| {r['avg_gen_len']:.1f} "
                    f"| {r['avg_compression']:.4f} |\n"
                )
            f.write("\n")

            # Best model callout
            best = max(ds_results, key=lambda x: x["rougeL_f1"])
            f.write(f"> **Best model on {ds_name}**: **{_model_name(best['file'])}** "
                    f"(ROUGE-L F1 = {best['rougeL_f1']:.4f})\n\n")
            f.write("---\n\n")

        # Key takeaways
        f.write("## Key Takeaways\n\n")
        overall_best = max(all_results, key=lambda x: x["rougeL_f1"])
        shortest = min(all_results, key=lambda x: x["avg_gen_len"])
        longest = max(all_results, key=lambda x: x["avg_gen_len"])
        f.write(f"- **Best overall ROUGE-L**: {_model_name(overall_best['file'])} on "
                f"{('CNN/DailyMail' if 'cnn' in overall_best['file'] else 'XSum')} "
                f"({overall_best['rougeL_f1']:.4f})\n")
        f.write(f"- **Shortest summaries**: {_model_name(shortest['file'])} "
                f"(avg {shortest['avg_gen_len']:.1f} tokens)\n")
        f.write(f"- **Longest summaries**: {_model_name(longest['file'])} "
                f"(avg {longest['avg_gen_len']:.1f} tokens)\n")
        f.write(f"- **Note**: ROUGE scores are computed using a lightweight tokenizer (no stemming). "
                f"For publication-quality numbers, run the full evaluator with BERTScore and SummaC.\n")

        # JSON structure explanation
        f.write("\n---\n\n")
        f.write("## Source Data Format\n\n")
        f.write("Each JSONL file in `summaries/` contains one JSON object per line:\n\n")
        f.write("```json\n")
        f.write("{\n")
        f.write('  "news": "The full text of the original news article...",\n')
        f.write('  "reference_summary": "The ground-truth human-written summary...",\n')
        f.write('  "generated_summary": "The output produced by the specific model..."\n')
        f.write("}\n")
        f.write("```\n")

    print(f"✅ Markdown report written to: {path}")


def write_csv(all_results: list[dict], path: Path):
    """Write a CSV spreadsheet with all metrics."""
    fieldnames = [
        "file", "num_samples",
        "avg_news_len", "avg_ref_len", "avg_gen_len", "std_gen_len",
        "avg_ref_sents", "avg_gen_sents", "avg_compression",
        "rouge1_f1", "rouge1_f1_std",
        "rouge2_f1", "rouge2_f1_std",
        "rougeL_f1", "rougeL_f1_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()}
            writer.writerow(row)
    print(f"✅ CSV spreadsheet written to: {path}")


# ===========================================================================
# Main — run lightweight analysis when executed directly
# ===========================================================================

if __name__ == "__main__":
    SUMMARIES_DIR = Path(__file__).parent / "summaries"
    OUTPUT_MD = Path(__file__).parent / "analysis_results.md"
    OUTPUT_CSV = Path(__file__).parent / "analysis_results.csv"

    jsonl_files = sorted(SUMMARIES_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found in summaries/")
        exit(1)

    print(f"Found {len(jsonl_files)} summary files to analyze.\n")
    all_results = []
    for fp in jsonl_files:
        print(f"  Analyzing {fp.name} …")
        result = analyze_file(str(fp))
        all_results.append(result)
        print(f"    → {result['num_samples']} samples, ROUGE-L F1 = {result['rougeL_f1']:.4f}")

    print()
    write_markdown(all_results, OUTPUT_MD)
    write_csv(all_results, OUTPUT_CSV)
    print("\nDone! 🎉")

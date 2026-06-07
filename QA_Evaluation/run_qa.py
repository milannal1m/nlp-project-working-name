import json
from qa_evaluator import QAFactEvaluator

if __name__ == "__main__":
    benchmark_dataset = "newsqasum_gold.jsonl" #choose datasets accordingly

    summary_dataset = "../summaries/Lead-1_cnn_dailymail_summaries.jsonl" #choose datasets accordingly

    print("Initializing QAFactEvaluator...")
    evaluator = QAFactEvaluator(gold_file=benchmark_dataset, max_articles=5)

    print("Starting evaluation...")
    evaluation_results = evaluator.run_qa_evaluation(summary_file=summary_dataset, return_individual_scores=True)

    print("\n--- Final Results ---")
    print(json.dumps(evaluation_results, indent=4))
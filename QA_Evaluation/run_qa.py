import json
from qa_evaluator import QAFactEvaluator

def main():
    # This must match the exact output name from your summaries_merger.py script
    master_dataset = "master_evaluation_dataset.jsonl"
    
    # The lean "gradesheet" file
    output_file = "final_evaluation_results.jsonl"

    print("==================================================")
    print("  Starting QAFactEval Pipeline Execution")
    print("==================================================")
    print(f"Target Dataset: {master_dataset}")

    # 1. Initialize the architecture built in qa_evaluator.py
    print("\n[1/2] Initializing the Evaluator...")
    try:
        evaluator = QAFactEvaluator(master_file=master_dataset)
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to initialize evaluator. Check environment. {e}")
        return

    # 2. Trigger the evaluation
    print("\n[2/2] Executing Factual Consistency Scoring...")
    try:
        final_output_path = evaluator.run_qa_evaluation(output_file=output_file)
    except Exception as e:
        print(f"\nCRITICAL ERROR: Pipeline failed during execution. {e}")
        return

    print("==================================================")
    print(f"PIPELINE COMPLETE.")
    print(f"Scores successfully saved to: {final_output_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
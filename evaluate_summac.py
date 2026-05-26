import glob
import json
from summac.model_summac import SummaCConv

print("Loading SummaC NLI model...")
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", device="cpu")

def evaluate_summac(file_path):
    print(f"Reading data from {file_path}...")
    documents = []
    summaries = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            documents.append(data['news']) 
            summaries.append(data['generated_summary'])

    print(f"Calculating SummaC for {len(summaries)} pairs...")
    results = model_conv.score(documents, summaries)
    
    avg_score = sum(results["scores"]) / len(results["scores"])
    
    print("\n--- SUMMAC RESULTS ---")
    print(f"Average SummaC Score: {avg_score:.4f}")
    print("----------------------\n")


# Evaluation on all the summaary files
file_paths = glob.glob("summaries/*.jsonl")
for path in file_paths:
    evaluate_summac(path)
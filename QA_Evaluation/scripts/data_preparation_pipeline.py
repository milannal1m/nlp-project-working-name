import argparse
import json
import hashlib
import os
import sys
import time
import traceback
from tqdm import tqdm
import torch
from transformers import set_seed

set_seed(42)

# The repo root is two levels up (QA_Evaluation/scripts/ -> QA_Evaluation/ -> repo),
# and it must be the repo root exactly: the imports below resolve the top-level src/
# package, so putting QA_Evaluation/ on the path instead would shadow it.
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.model import SummarizationModel, RunConfig
from src.prompts import PROMPT_CONFIGS
from src.evaluator import Evaluator # importing the cleaner
from src.dataset import strip_dateline  # Importing native cleaner

def get_article_id(text):
    """Generates a unique ID based on source text to prevent mismatching"""
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Run SLM summary generation on NewsQASum.")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path or local path")
    parser.add_argument("--quant", type=str, choices=["None", "8bit", "4bit"], default="None", help="Quantization level")
    parser.add_argument("--prompt_id", type=str, choices=["P1", "P2", "P3"], required=True, help="Prompt variant ID")
    parser.add_argument("--input_file", type=str, default="newsqasum_gold.jsonl", help="Path to input dataset")
    parser.add_argument("--output_dir", type=str, default="Outputs", help="Directory for temp files")
    parser.add_argument("--sample", type=int, default=None, help="Process only N articles for local testing")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Cleaning the model name to use as a file name identifier
    model_clean_name = args.model_path.split("/")[-1].replace("-", "_").lower()
    output_filename = f"{model_clean_name}_{args.quant}_{args.prompt_id}.jsonl"
    output_path = os.path.join(args.output_dir, output_filename)

    # Fault Tolerance: Check for existing progress to allow resuming
    processed_ids = set()
    if os.path.exists(output_path):
        print(f"Found existing output file. Scanning for already processed articles...")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["article_id"])

                    except json.JSONDecodeError:
                        continue
        print(f"Resuming gernation. Skipping {len(processed_ids)} already processed articles.")

    # Initializing the config and model by utilizing methods from models.py
    config = RunConfig(
        model_name_or_path=args.model_path,
        quantization_method=args.quant,
        prompt_template=PROMPT_CONFIGS[args.prompt_id]["template"],
        max_new_tokens=PROMPT_CONFIGS[args.prompt_id]["max_new_tokens"],
    )

    print(f"Loading Model: {args.model_path} | Quant: {args.quant} | Prompt: {args.prompt_id}")
    summarizer = SummarizationModel(config)

    # Pre-load all valid lines to use tqdm for accurate progress estimation
    with open(args.input_file, "r", encoding="utf-8") as infile:
        all_lines = [line for line in infile if line.strip()]
    
    if args.sample:
        all_lines = all_lines[:args.sample]
        print(f"TEST MODE: Limiting run to {args.sample} articles.")

    print(f"Starting generation. Saving progress to: {output_path}")

    run_start = time.time()
    generated_count = 0

    # Process the dataset (Append mode 'a' allows safe resuming)
    with open(output_path, "a", encoding="utf-8") as outfile:
        for line in tqdm(all_lines, desc="Generating Summaries"):
            data = json.loads(line)
            raw_story = data.get("story", "")

            if not raw_story:
                continue
            
            # Clean the artifacts using your team's exact logic for a 1:1 baseline comparison
            story_text = strip_dateline(raw_story)
            
            # Generate the hash based on the CLEANED story
            article_id = get_article_id(story_text)
            
            # Skip if already processed
            if article_id in processed_ids:
                continue
            
            try:
                # Generate summary (token counts are used for throughput logging)
                raw_generated_summary, input_len, generated_len = summarizer.summarize(story_text)

                # CLEAN THE OUTPUT: Strip P3 reasoning and duplicate markers
                clean_summary = Evaluator.extract_summary(raw_generated_summary)

                # Save output mapped to this article ID. raw_summary_text keeps the
                # pre-extraction output so the 'Summary:' marker (esp. P3) is auditable.
                output_row = {
                    "article_id": article_id,
                    "story": story_text,
                    "human_questions": data.get("questions", []),
                    "human_answers": data.get("answers", []),
                    "summary_text": clean_summary,
                    "raw_summary_text": raw_generated_summary,
                    }
                outfile.write(json.dumps(output_row) + "\n")
                outfile.flush()  # Ensure it writes to disk immediately

                generated_count += 1
                if generated_count == 1 or generated_count % 10 == 0:
                    elapsed_min = (time.time() - run_start) / 60
                    print(f"  generated {generated_count} | input_tokens={input_len} "
                          f"| generated_tokens={generated_len} | elapsed={elapsed_min:.2f} min",
                          flush=True)

            except Exception as e:
                print(f"\nError processing article {article_id}: {e}")
                traceback.print_exc()
                # Continue processing other articles despite a single failure
                continue

    total_min = (time.time() - run_start) / 60
    print(f"--- SUCCESS --- Generated {generated_count} summaries in {total_min:.2f} min. "
          f"Saved to {output_path}")

if __name__ == "__main__":
    main()
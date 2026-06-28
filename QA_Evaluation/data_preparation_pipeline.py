import argparse
import json
import hashlib
import os
import sys
import torch
from transformers import set_seed

set_seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.model import SummarizationModel, RunConfig
from src.prompts import PROMPT_CONFIGS

def get_article_id(text):
    """Generates a unique ID based on source text to prevent mismatching"""
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Run SLM summary generation on NewsQASum.")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path or local path")
    parser.add_argument("--quant", type=str, choices=["None", "8bit", "4bit"], default="None", help="Quantization level")
    parser.add_argument("--prompt_id", type=str, choices=["P1", "P2", "P3"], required=True, help="Prompt variant ID")
    parser.add_argument("--input_file", type=str, default="newsqasum_gold.jsonl", help="Path to input dataset")
    parser.add_argument("--output_dir", type=str, default="temp_outputs", help="Directory for temp files")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Cleaning the model name to use as a file name identifier
    model_clean_name = args.model_path.split("/")[-1].replace("-", "_").lower()
    output_filename = f"temp_{model_clean_name}_{args.quant}_{args.prompt_id}.jsonl"
    output_path = os.path.join(args.output_dir, output_filename)

    # Initializing the config and model by utilizing methods from models.py
    config = RunConfig(
        model_name_or_path=args.model_path,
        quantization_method=args.quant,
        prompt_template=PROMPT_CONFIGS[args.prompt_id]["template"]
    )

    summarizer = SummarizationModel(config)

    print(f"starting generation, Saving progress to: {output_path}")

    # Processing the dataset
    with open(args.input_file, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line_idx, line in enumerate(infile):
            if not line.strip():
                continue
            data = json.loads(line)
            story_text = data.get("story", "")

            if not story_text:
                continue
            
            # Assigning ID
            article_id = get_article_id(story_text)
            
            # Calling summarize method from models.py
            generated_summary, _, _ = summarizer.summarize(story_text)

            # Saving the output mapped to this article ID
            output_row = {"article_id": article_id, "summary_text":generated_summary}

            outfile.write(json.dumps(output_row) + "\n")

            if (line_idx + 1) % 50 == 0:
                print(f"Processed {line_idx + 1} articles...")

    print(f"--- SUCCESS --- All summaries saved to {output_path}")


if __name__ == "__main__":
    main()
import json
import os

try:
    from transformers import pipeline
except ImportError:
    print("CRITICAL ERROR: You need to install transformers to run a real model locally.")
    print("Run this in your terminal: pip install transformers torch")
    exit()

def generate_real_data():
    # Using the relative path that successfully found your file in the last run
    gold_dataset_path = "Datasets/newsqasum_gold.jsonl"
    output_path = "mock_master_evaluation_dataset.jsonl"

    print(f"[1/4] Reading exactly one row from {gold_dataset_path}...")
    
    try:
        with open(gold_dataset_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                if not line.strip(): continue
                raw_data = json.loads(line.strip())
                break # Grab exactly 1 row
    except FileNotFoundError:
        print(f"ERROR: Could not find {gold_dataset_path}.")
        return

    story = raw_data.get("story", "")
    questions = raw_data.get("questions", [])
    answers = raw_data.get("answers", [])
    article_id = raw_data.get("id", "test_hash_123") 

    if not story:
        print("ERROR: 'story' is still empty. Check your dataset's exact keys!")
        return

    print(f"[2/4] Successfully loaded article (Length: {len(story)} chars).")
    
    # === THE FIX ===
    # Switched to 'text-generation' and 'gpt2' to perfectly match your Mac's allowed tasks
    print(f"[3/4] Downloading/Loading a REAL tiny SLM (gpt2)...")
    generator = pipeline("text-generation", model="gpt2", device="cpu")

    mock_row = {
        "article_id": article_id,
        "source_article": story,         
        "human_questions": questions,    
        "human_answers": answers         
    }

    models = ["Llama3.2-3B-Ins", "Phi-3-Mini"]
    quants = ["None", "8bit", "4bit"]
    prompts = ["P1", "P2", "P3"]
    
    print("[4/4] Generating 18 real summary variations. This will take a minute on CPU...")
    
    variation_length = 15
    
    # We take just a small chunk of the story to prompt the generator
    prompt_text = f"Summary of the news: {story[:300]}...\n\nShort summary:"
    
    for m in models:
        for q in quants:
            for p in prompts:
                key = f"{m}_{q}_{p}"
                print(f"   -> Generating real text for {key}...")
                
                # Using max_new_tokens and temperature to force 18 unique outputs
                summary_output = generator(
                    prompt_text, 
                    max_new_tokens=variation_length, 
                    do_sample=True, 
                    temperature=0.8,
                    pad_token_id=generator.tokenizer.eos_token_id,
                    return_full_text=False # Only return the generated part, not the prompt
                )
                
                # Clean up the output string and add it to our dictionary
                generated_string = summary_output[0]['generated_text'].strip()
                # If the model hallucinates line breaks, strip them out to keep the JSON clean
                generated_string = generated_string.replace('\n', ' ') 
                
                mock_row[key] = generated_string
                variation_length += 2 # slightly increase token count to guarantee variance
                
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(json.dumps(mock_row) + "\n")
        
    print(f"\nSUCCESS! Created {output_path} with 18 REAL generated texts.")

if __name__ == "__main__":
    generate_real_data()
import os
import json
import hashlib
import glob

def get_article_id(text):
    """Generates the same MD5 hash used in the generation script. """
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

def verify_merge(output_file):
    """Reads the final master file to ensure all columns were stitched properly"""
    print("\n--- Running Verification Check ---")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            first_article = json.loads(f.readline())
        keys = list(first_article())

        # We expect 3 foundation keys + (model * prompt * quant) variations.
        print(f"Total Columns found: {len(keys)}")
        print("Columns mapping preview:")
        for key in keys:
            val_preview = str(first_article[key])[:40].replace('\n', ' ') + "..."
            print(f"   - {key: {val_preview}}")
        
        if len(keys) == 21:
            print("\n Verfication Passed!")
        else:
            print(f"\nWarning: Expected 21 keys but found {len(keys)}. Some cluster jobs may have failed.")
    except Exception as e:
        print(f"Verfication failed to read the file: {e}")


def main():
    gold_file = "newsqasum_gold.jsonl"
    temp_dir = "temp_outputs"
    output_file = "master_evaluation_dataset.jsonl"

    print(f"1. Loading baseline dataset: {gold_file}")
    master_data = {}

    # Loading original articles, questions and answers
    try:
        with open(gold_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                story = data.get('story', '')
                if not story: continue

                article_id = get_article_id(story)

                # Creating the foundation for this article
                master_data[article_id] = {
                    "article_id": article_id,
                    "source_article": story,
                    "human_questions": data.get('questions', []),
                    "human_answers" : data.get('answers', [])
                }
    except FileNotFoundError:
        print(f"Error: Cound not find {gold_file}, make sure you're in the right directory.")
        return
    
    print(f"    Loaded {len(master_data)} baseline articles.")

    #Iterating through all temp summary files
    temp_files = glob.glob(os.path.join(temp_dir, "temp_*.jsonl"))
    print(f"\nFound {len(temp_files)} temporary summary files. Merging...")

    if not temp_files:
        print("Error: No temp files found.")
        return
    
    for file_path in temp_files:
        # Extracting clean filename for the columns for the master file.
        filename = os.path.basename(file_path)
        col_key = filename.replace("temp_", "").replace(".jsonl", "")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                article_id = row.get("article_id")
                summary = row.get("summary_text")

                # Injecting summary into the master dictionary specific to its key.
                if article_id in master_data:
                    master_data[article_id][col_key] = summary
    
    # Writing final master file
    print(f"\n Writing master dataset to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in master_data.values():
            f.write(json.dumps(article) + '\n')
    
    print("--- SUCCESS --- Master dataset compiled successfully!")
    
    # Calling Verfication function
    verify_merge(output_file)

if __name__ == "__main__":
    main()

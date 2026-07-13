import os
import json
import glob

def verify_merge(output_file):
    """Validates the structural integrity of the lean master matrix."""
    print("\n--- Running Verification Check ---")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line:
                print("Warning: Master output file is empty.")
                return
            first_row = json.loads(first_line)
        
        keys = list(first_row.keys())

        print(f"Total columns compiled in master row: {len(keys)}")
        print("Columns mapping preview:")
        for key in keys:
            val_preview = str(first_row[key])[:40].replace('\n', ' ') + "..."
            print(f"   - {key}: {val_preview}")
        
        if len(keys) == 19:
            print("\n[SUCCESS] Verification Passed! Lean matrix contains exactly 1 article_id and 18 variations.")
        else:
            print(f"\n[WARNING] Expected 19 columns, but found {len(keys)}. Some cluster jobs might be missing.")
    except Exception as e:
        print(f"Verification failed to process file: {e}")


def main():
    # Dynamic Path Resolution: Absolute path relative to where this script is saved
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Since merger sits in QA_Evaluation/, test_outputs is right next to it
    variation_dir = os.path.join(SCRIPT_DIR, "Outputs")
    output_file = os.path.join(SCRIPT_DIR, "master_evaluation_dataset.jsonl")

    print(f"Targeting variations directory: {variation_dir}")
    print(f"Targeting output master matrix: {output_file}\n")

    # In-memory dictionary to hold rows: { article_id: { article_id: x, var1: y } }
    master_matrix = {}

    # Gather ALL jsonl files in the target folder
    all_jsonl_files = glob.glob(os.path.join(variation_dir, "*.jsonl"))
    
    # Filter out the master output file if it happens to be in the same folder
    variation_files = [f for f in all_jsonl_files if os.path.basename(f) != "master_evaluation_dataset.jsonl"]
    
    print(f"Found {len(variation_files)} variation datasets to process.")

    if not variation_files:
        print("Error: 0 datasets found. Please check that your files are physically inside the directory shown above.")
        return
    
    for file_path in variation_files:
        filename = os.path.basename(file_path)
        # Clean up column header names cleanly whether they use 'temp_' prefix or not
        col_key = filename.replace("temp_", "").replace(".jsonl", "")

        print(f" -> Extracting columns from: {filename} (Column Key: {col_key})")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                article_id = row.get("article_id")
                summary = row.get("summary_text")

                if not article_id: continue

                # Initialize row structure if it's the first time seeing this article hash
                if article_id not in master_matrix:
                    master_matrix[article_id] = {
                        "article_id": article_id
                    }
                
                # Append this model summary column directly to the row reference
                master_matrix[article_id][col_key] = summary
    
    # Write the compiled matrix payload out to disk
    print(f"\nWriting lean master evaluation matrix to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for row_data in master_matrix.values():
            f.write(json.dumps(row_data) + '\n')
    
    print("\n--- Compilation Complete ---")
    verify_merge(output_file)


if __name__ == "__main__":
    main()
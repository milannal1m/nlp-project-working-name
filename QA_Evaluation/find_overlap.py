import json
import re

def get_fingerprint(text):
    """Strips formatting and skips the first 50 chars to bypass dateline variations like '(CNN)'."""
    clean = re.sub(r'[^a-z0-9]', '', str(text).lower())
    # Grab a chunk of text starting from character 50 to guarantee a safe match
    return clean[50:250] if len(clean) > 250 else clean

def find_overlap(cnn_dm_path, newsqa_path, output_path):
    print("Loading CNN/DailyMail dataset...")
    
    cnn_articles = {}
    with open(cnn_dm_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # CNN dataset uses 'news'
            text = data.get('news', '') 
            if text:
                cnn_articles[get_fingerprint(text)] = data

    print(f"Loaded {len(cnn_articles)} unique articles from CNN/DM.")
    print("Scanning NewsQA for overlapping articles and merging summaries...")

    matches = 0
    with open(newsqa_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            # NewsQA dataset uses 'story'
            text = data.get('story', '') 
            
            if text:
                fingerprint = get_fingerprint(text)
                if fingerprint in cnn_articles:
                    # Pull the generated summary from the CNN dataset and inject it here
                    cnn_data = cnn_articles[fingerprint]
                    data['generated_summary'] = cnn_data.get('generated_summary', '')
                    
                    # Write the merged data to our new file
                    outfile.write(json.dumps(data) + '\n')
                    matches += 1

    print(f"--- SUCCESS ---")
    print(f"Found {matches} perfectly overlapping articles.")
    print(f"Saved merged QA dataset to: {output_path}")

find_overlap(
    cnn_dm_path='summaries/TextRank_cnn_dailymail_summaries.jsonl', 
    newsqa_path='summaries/newsqasum_gold.jsonl',
    output_path='QA_Evaluation/matched_qa_dataset.jsonl'
)
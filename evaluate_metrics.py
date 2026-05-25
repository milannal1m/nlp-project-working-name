import glob
import json
import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

def evaluate_jsonl_file(file_path):
    generated_summaries = []
    news = []

    print(f"\nEvaluating {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            generated_summary = data['generated_summary'] 
            original_text = data['news']
            
            generated_summaries.append(generated_summary)
            news.append(original_text)

    print(f"Calculating metrics for {len(generated_summaries)} summaries...")
    
    bleu_score = bleu.compute(predictions=generated_summaries, references=news)
    rouge_score = rouge.compute(predictions=generated_summaries, references=news)
    meteor_score = meteor.compute(predictions=generated_summaries, references=news)
    
    # I specified language here as it's necessary for BERT Score
    bert_score = bertscore.compute(predictions=generated_summaries, references=news, lang="en")
    avg_bert_f1 = sum(bert_score['f1']) / len(bert_score['f1'])

    print("\n--- RESULTS ---")
    print(f"BLEU:        {bleu_score['bleu']:.4f}")
    print(f"ROUGE-L:     {rouge_score['rougeL']:.4f}")
    print(f"METEOR:      {meteor_score['meteor']:.4f}")
    print(f"BERTScore F1:{avg_bert_f1:.4f}")
    print("---------------\n")

# Evaluation on all the summaary files
file_paths = glob.glob("summaries/*.jsonl")
for path in file_paths:
    evaluate_jsonl_file(path)
import json
import os

from qafacteval import QAFactEval

class QAFactEvaluator:
    def __init__(self, master_file):
        self.master_file = master_file

        print("Initializing QAFactEval Pipeline using the locked cluster environment...")

        kwargs = {
            "cuda_device": -1, # set -1 for cpu mode and 0 for gpu
            "use_lerc_quip": True,
            "verbose": True, # set True for local testing, False for cluster
            "generation_batch_size": 2, # set to 2 for local testing, 32 for cluster
            "answering_batch_size": 2, # set to 2 for local testing, 32 for cluster
            "lerc_batch_size": 2 # set to 2 for local testing, 8 for cluster
        }

        self.metric = QAFactEval(**kwargs)

    def run_qa_evaluation(self, output_file="final_evaluation_results.jsonl"):
        print(f"Reading merged dataset; {self.master_file}")
        
        foundation_keys = {"article_id", "source_article", "human_questions", "human_answers"}
        
        with open(self.master_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:
                
                for idx, line in enumerate(infile):

                    # UNCOMMENT THIS BLOCK FOR LOCAL TESTING: Stop after the 1st article
                    if idx >= 1: 
                        print("Local test complete. Reached 1 article limit.")
                        break
                    ###

                    if not line.strip(): continue

                    article_data = json.loads(line.strip())
                    source_text = article_data.get("source_article", "")
                    article_id = article_data.get("article_id", f"unknown_{idx}")

                    # Extracting gold human questiosn and answers
                    human_questions = article_data.get("human_questions", [])
                    human_answers = article_data.get("human_answers", [])

                    # skip the row if the rows is missing required data.
                    if not source_text or not human_questions or not human_answers: continue

                    # Formatting human questions.
                    qa_pairs_list = [{"question": q, "answer": a} for q, a in zip(human_questions, human_answers)]
                     
                    # Isolating the 18 generated summary variations
                    summary_keys = [k for k in article_data.keys() if k not in foundation_keys]

                    # Formatting the batches, it needs a list of source texts and a list of lists of summaries.
                    sources_batch = [source_text] * len(summary_keys)
                    summaries_batch = [[article_data[k]] for k in summary_keys]

                    print(f"Scoring Article {idx + 1} | ID: {article_id} | Processing {len(summary_keys)} variations")
                    article_scores = {"article_id": article_id}

                    try:
                         # Added qa_pairs_precomputed parameter to inject your human_questions batch.
                         # This forces the pipeline to use dataset questions.
                         score_outputs = self.metric.score_batch_qafacteval(
                              sources_batch,
                              summaries_batch,
                              qa_pairs_precomputed=[qa_pairs_list] * len(summary_keys),
                              return_qa_pairs=False
                         )

                         # Mapping lerc scores back to the specific SLM model/quant/prompt column.
                         for i, key in enumerate(summary_keys):
                              article_scores[f"{key}_lerc_score"] = score_outputs[i][0]['qa-eval']['lerc_quip']

                    except Exception as e:
                         print(f"Error scoring {article_id}: {e}")
                         for key in summary_keys:
                              article_scores[f"{key}_lerc_score"] = None

                    outfile.write(json.dumps(article_scores) + "\n")
                    outfile.flush()

        print(f"\n--- SUCCESS --- Evaluation complete. Results has been saved to{output_file}")
        return output_file
    

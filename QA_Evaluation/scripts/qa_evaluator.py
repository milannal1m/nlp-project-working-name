import json
import os
import sys
import builtins
import torch  # Added to auto-detect hardware

# ==========================================
# 1. DYNAMIC PATH RESOLUTION
# ==========================================
# Anchors to QA_Evaluation/, the parent of this scripts/ directory -- that is where the
# heavy offline model folders live (see README.md, "TODO: Installation of ...").
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_DIR = os.path.dirname(SCRIPT_DIR)
PATH_TO_QAEVAL = os.path.join(CURRENT_DIR, "qaeval")
PATH_TO_QAFACTEVAL = os.path.join(CURRENT_DIR, "QAFactEval")

# Add them to Python's search memory
sys.path.insert(0, PATH_TO_QAEVAL)
sys.path.insert(0, PATH_TO_QAFACTEVAL)

# Sanity check tripwires
if not os.path.exists(PATH_TO_QAFACTEVAL):
    raise FileNotFoundError(f"CRITICAL: Cannot find QAFactEval folder at {PATH_TO_QAFACTEVAL}")
if not os.path.exists(PATH_TO_QAEVAL):
    raise FileNotFoundError(f"CRITICAL: Cannot find qaeval folder at {PATH_TO_QAEVAL}")

# ==========================================
# 2. LEGACY COMPILER BYPASS
# ==========================================
_orig_issubclass = builtins.issubclass
def safe_issubclass(cls, classinfo):
    try:
        return _orig_issubclass(cls, classinfo)
    except TypeError:
        return False
builtins.issubclass = safe_issubclass

from qafacteval import QAFactEval

# ==========================================
# 3. EVALUATOR CLASS
# ==========================================
class QAFactEvaluator:
    def __init__(self, master_file):
        self.master_file = master_file
        model_folder = f"{PATH_TO_QAFACTEVAL}/models"
        
        # ----------------------------------------------------
        # HARDWARE AUTO-DETECTION: Automatically scales to the machine
        # ----------------------------------------------------
        has_gpu = torch.cuda.is_available()
        
        if has_gpu:
            print(">>> GPU Detected: Scaling pipeline for cluster/local GPU execution.")
            c_device = 0
            b_size = 32
            l_size = 8
            is_verbose = False
        else:
            print(">>> No GPU Detected: Downscaling to local CPU testing mode.")
            c_device = -1
            b_size = 2
            l_size = 2
            is_verbose = True

        kwargs = {
            "cuda_device": c_device,
            "use_lerc_quip": True,
            "verbose": is_verbose,
            "generation_batch_size": b_size,
            "answering_batch_size": b_size,
            "lerc_batch_size": l_size
        }

        self.metric = QAFactEval(
            lerc_quip_path=f"{model_folder}/quip-512-mocha",
            generation_model_path=f"{model_folder}/generation/model.tar.gz",
            answering_model_dir=f"{model_folder}/answering",
            lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
            lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
            **kwargs
        )


    # max_articles controls the testing limit. Set to None to run the whole dataset.
    def run_qa_evaluation(self, output_file="final_evaluation_results.jsonl", max_articles=None):
        import traceback 
        print(f"Reading merged dataset: {self.master_file}")
        
        foundation_keys = {"article_id", "source_article", "human_questions", "human_answers"}
        skipped = 0
        scored = 0

        with open(self.master_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
                
                for idx, line in enumerate(infile):
                    # Safely break if testing limit is reached
                    if max_articles is not None and idx >= max_articles: 
                        print(f"Reached execution limit of {max_articles} article(s). Stopping.")
                        break

                    if not line.strip(): continue

                    article_data = json.loads(line.strip())
                    
                    # Defensively clean the source text
                    raw_source = article_data.get("source_article", "")
                    source_text = str(raw_source[0]) if isinstance(raw_source, list) else str(raw_source)
                    article_id = str(article_data.get("article_id", f"unknown_{idx}"))

                    # Defensively clean the QA pairs
                    human_questions = article_data.get("human_questions", [])
                    human_answers = article_data.get("human_answers", [])

                    # Skipping here is silent data loss, so say which field went missing.
                    # An empty results file almost always means the merge dropped these.
                    missing = [name for name, value in (
                        ("source_article", source_text),
                        ("human_questions", human_questions),
                        ("human_answers", human_answers),
                    ) if not value]
                    if missing:
                        skipped += 1
                        print(f"[skip] row {idx + 1} | ID: {article_id} | "
                              f"missing: {', '.join(missing)}", flush=True)
                        continue

                    qa_pairs_list = []
                    for q, a in zip(human_questions, human_answers):
                        clean_q = str(q[0]) if isinstance(q, list) else str(q)
                        clean_a = str(a[0]) if isinstance(a, list) else str(a)
                        
                        qa_pairs_list.append({
                             "question": clean_q, 
                             "answer": clean_a,
                             "answers": [clean_a] 
                        })
                     
                    # Defensively clean the generated summaries
                    summary_keys = [k for k in article_data.keys() if k not in foundation_keys]
                    summaries_batch = []
                    for k in summary_keys:
                        raw_summ = article_data[k]
                        clean_summ = str(raw_summ[0]) if isinstance(raw_summ, list) else str(raw_summ)
                        summaries_batch.append([clean_summ])

                    sources_batch = [source_text] * len(summary_keys)

                    print(f"Scoring Article {idx + 1} | ID: {article_id} | Processing {len(summary_keys)} variations")
                    article_scores = {"article_id": article_id}

                    try:
                         score_outputs = self.metric.score_batch_qafacteval(
                              sources_batch,
                              summaries_batch,
                              qa_pairs_precomputed=[[qa_pairs_list]] * len(summary_keys),
                              return_qa_pairs=True 
                         )

                         # THE SANITY CHECK PRINT (only on the first article actually scored,
                         # which is not necessarily row 0 once rows can be skipped)
                         if scored == 0:
                             print("\n=== PIPELINE PAYLOAD VERIFICATION ===")
                             print(f"SOURCE [0] (First 150 chars): {sources_batch[0][:150]}...")
                             print(f"SUMMARY [0] (First 150 chars): {summaries_batch[0][0][:150]}...")
                             print(f"QA PAIR [0]: {([[qa_pairs_list]] * len(summary_keys))[0][0][0]}")
                             print("=====================================\n")

                         for i, key in enumerate(summary_keys):
                              metrics_dict = score_outputs[i][0]
                              
                              if 'qa-eval' in metrics_dict:
                                  lerc = metrics_dict['qa-eval'].get('lerc_quip')
                              else:
                                  lerc = metrics_dict.get('lerc_quip')
                                  
                              article_scores[f"{key}_lerc_score"] = lerc

                    except Exception as e:
                         print(f"\n[!] CRITICAL TRACEBACK FOR {article_id}:")
                         traceback.print_exc()
                         print("\n")
                         for key in summary_keys:
                              article_scores[f"{key}_lerc_score"] = None

                    outfile.write(json.dumps(article_scores) + "\n")
                    outfile.flush()
                    scored += 1

        print(f"\n--- Evaluation complete --- scored {scored} article(s), "
              f"skipped {skipped}.")
        if scored == 0:
            print("[ERROR] Nothing was scored. Every row was missing at least one of "
                  "source_article / human_questions / human_answers — re-run "
                  "summaries_merger.py, which is what populates them.")
        print(f"Results have been saved to {output_file}")
        return output_file
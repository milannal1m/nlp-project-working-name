import json

class QAFactEvaluator:
    def __init__(self, gold_file, max_articles=None):
        '''Initializes the evaluator and builds the lookup dictionary once.'''
        self.gold_file = gold_file
        self.max_articles = max_articles
        self.gold_lookup = self._build_gold_lookup()

    def _clean_text(self, text):
        '''Helper: Strips all the punctuation and spaces, keeps only lowercase letters/numbers'''
        return "".join(char for char in text if char.isalnum()).lower()

    def _build_gold_lookup(self):
        """
        Reads the gold dataset and builds dictionary for faster matching.
        max_articles: Set to a small number (e.g. 5) for local testing.
        Set to None for the full cluster run.
        """
        gold_lookup = {}
        # Loading the entire gold dataset here so the sliding window has the full 
        # haystack to search through. The limit is safely applied in the evaluation loop below.
        with open(self.gold_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                story_text = data.get("story", "")
                questions_list = data.get("questions", [])

                if story_text and questions_list:
                    clean_story = self._clean_text(story_text)

                    #q_strings to clean up questions as well
                    q_strings = [q['question'] if isinstance(q, dict) else q for q in questions_list]

                    #Saving dictionary with cleaned up articles and questions where clean_story as keys and q_strings as values
                    gold_lookup[clean_story] = q_strings

        print(f"Loaded {len(gold_lookup)} articles into the gold lookup directory.")
        return gold_lookup

    def run_qa_evaluation(self, summary_file, return_individual_scores=False): #Returns a dictionary of mean and stddev of lerc scores
        print(f"\n--- 2. Matching with AI summary: {summary_file} ---")
        matched_news = []
        matched_summaries = []
        matched_questions = []

        try:
            with open(summary_file, "r", encoding = "utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    news = data.get("news", "")
                    gen_summary = data.get("generated_summary", "")

                    clean_news = self._clean_text(news)

                    #Taking a 100 char chunk safely away from the starting noise
                    search_chunk = clean_news[50:150] if len(clean_news) > 150 else clean_news

                    #Sliding Window search
                    for gold_key in self.gold_lookup.keys():
                        if search_chunk in gold_key:
                            matched_news.append(news)
                            matched_summaries.append(gen_summary)
                            matched_questions.append(self.gold_lookup[gold_key])
                            break #Match is found in the article
                    
                    if self.max_articles != None and len(matched_news) >= self.max_articles:
                        break
        except FileNotFoundError:
            print(f"Error: Could not find {summary_file}.")
            return
        
        if len(matched_news) == 0:
            print("No matches found. The gold articles didn't match the AI summaries.")
            return
        
        print(f"Match successful! Found {len(matched_news)} articles to evaluate.")
        print("\n--- 3. Running the AI Evaluation ---")
        from qafacteval import QAFactEval

        kwargs = {"model_folder": "models/qafacteval", "device": "cuda"} #change device accordingly based on where you're running it.
        qa_evaluator = QAFactEval(**kwargs)

        results = qa_evaluator.score_batch(
            inputs=matched_news,
            predictions=matched_summaries,
            questions=matched_questions,
            return_qa_pairs=True
        )

        final_scores = [res[0]['qa-eval']['lerc_quac'] for res in results if res and 'qa-eval' in res[0]]

        if final_scores:
            import statistics #calculating stats instead as individual returning lerc scores would be unreadable for us.
            mean_score = statistics.mean(final_scores)
            std_score = statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0
            output = {
                "qa_eval_mean": mean_score,
                "qa_eval_std": std_score
                    }
            if return_individual_scores:
                output["individual_scores"] = final_scores
            return output

        else:
            print("\nError: Pipeline ran, but no scores were returned.")
            return {
                "qa_eval_mean": None,
                "qa_eval_std": None
                    }
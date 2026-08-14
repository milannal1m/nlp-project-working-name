PROMPT_CONFIGS = {
    "P1": {
        "template": "News: {news}\nSummarize the news in two sentences. "
                    "Output only the summary, beginning with 'Summary:'.",
        "max_new_tokens": 1000,
    },
    "P2": {
        "template": "You are a news summarizer.\nArticle: {news}\n"
                    "Write a concise two-sentence summary covering the key facts and main event. "
                    "Output only the summary, beginning with 'Summary:'.",
        "max_new_tokens": 1000,
    },
    "P3": {
        "template": "Read the following news article.\nArticle: {news}\n"
                    "First, briefly identify the main event, the key people involved, and the outcome. "
                    "Then, on a new line, write a two-sentence summary based on those facts, "
                    "beginning with 'Summary:'.",
        "max_new_tokens": 1000,
    },
    "P4": {
        "template": "News: {example_news}\n"
                    "Summarize the news in two sentences. "
                    "Output only the summary, beginning with 'Summary:'.\n"
                    "Summary: {example_summary}\n\n"
                    "News: {news}\n"
                    "Summarize the news in two sentences. "
                    "Output only the summary, beginning with 'Summary:'.",
        "max_new_tokens": 1000,
    },
}

PROMPT_CONFIGS = {
    "P1": {
        "template": "News: {news}\nSummarize the news in two sentences. Summary:",
        "max_new_tokens": 150,
    },
    "P2": {
        "template": "You are a news summarizer. Read the following article and write a concise two-sentence summary covering the key facts and main event.\nArticle: {news}\nSummary:",
        "max_new_tokens": 150,
    },
    "P3": {
        "template": "Read the following news article. First identify the main event, the key people involved, and the outcome. Then write a two-sentence summary based on those facts.\nArticle: {news}\nSummary:",
        "max_new_tokens": 300,
    },
}

# GitHub Copilot Instructions for NLP Project

## 1. Tech Stack & Coding Preferences
- **Language**: Strictly use Python 3. Always include Type Hints in function signatures (e.g., `def process_text(text: str) -> list[str]:`).
- **NLP Ecosystem**: Prioritize modern APIs from Hugging Face (`transformers`, `datasets`) and `PyTorch`. For traditional NLP pipelines, prefer `spaCy` or `NLTK`.
- **Data Processing**: When handling large-scale text datasets, rely on vectorized operations (Pandas/NumPy) or generators to prevent memory overhead. Avoid inefficient `for` loops.

## 2. Code Style & Standards
- **Documentation**: All core preprocessing functions and model classes must include Google-style docstrings. Explicitly document Tensor shapes and data types in the comments.
- **Naming Conventions**: Use semantic, descriptive names. Avoid abstract variables like `x` or `y`. Use standard NLP terminology (e.g., `input_ids`, `attention_mask`, `labels`, `logits`).
- **Logging & Debugging**: Utilize the `logging` module and `tqdm` for progress bars. Minimize the use of raw `print()` statements, especially inside training loops.

## 3. Academic & Report Formatting
- **Documentation standard**: When generating content for academic reports, documentation, or READMEs, maintain a formal, objective tone. 
- **Mathematics**: Use LaTeX syntax for rendering mathematical formulas, equations, and loss functions.

## 4. Communication & AI Behavior
- Provide concise, direct code snippets. Do not over-explain obvious logic unless specifically requested.
- Proactively identify and warn me about potential GPU Out-Of-Memory (OOM) risks, tensor shape mismatches, or memory leaks in PyTorch routines.
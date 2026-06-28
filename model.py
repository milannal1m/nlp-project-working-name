from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class RunConfig:
    model_name_or_path: str
    quantization_method: str = "None"
    output_dir: str = "./summaries"
    prompt_name: str = "P1"
    prompt_template: str = "News: {news}\nSummarize the news in two sentences."
    max_input_length: int = 2048
    max_new_tokens: int = 150


class SummarizationModel:
    def __init__(self, config: RunConfig):
        quant_config = self._build_quantization_config(config.quantization_method)
        print(f"Loading model with quantization method: {config.quantization_method}")

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=quant_config,
            device_map="auto",
            dtype=torch.float16 if config.quantization_method == "None" else None,
        )
        self.config = config

    def summarize(self, text: str) -> tuple[str, int, int]:
        # Extract prefix and suffix from the prompt template by splitting on {news}.
        parts = self.config.prompt_template.split("{news}")
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        # Estimate template overhead by measuring a minimal message.
        # Apply chat template to a message with minimal content to understand structure.
        minimal_messages = [{"role": "user", "content": f"{prefix}X{suffix}"}]
        minimal_prompt = self.tokenizer.apply_chat_template(
            minimal_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        minimal_tokens = self.tokenizer.encode(minimal_prompt)
        overhead_estimate = len(minimal_tokens) - 1  # Subtract 1 for the "X" placeholder

        # Calculate space available for news (total - overhead)
        available_for_news = max(1, self.config.max_input_length - overhead_estimate)

        # Truncate news text if it exceeds available space, preserving template tokens.
        text_encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if len(text_encoded) > available_for_news:
            text_encoded = text_encoded[:available_for_news]
            text = self.tokenizer.decode(text_encoded)

        # Build messages with truncated text and apply chat template.
        message_content = f"{prefix}{text}{suffix}".strip()
        messages = [{"role": "user", "content": message_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize with truncation as a safety fallback.
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
        ).to(self.model.device)

        input_len = inputs.input_ids.shape[1]

        # Get stop tokens from tokenizer.
        stop_token_ids = self._get_stop_token_ids()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
            )

        generated_len = outputs.shape[1] - input_len
        generated_text = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        return generated_text, input_len, generated_len

    def _get_stop_token_ids(self) -> list[int]:
        # Turn-ending tokens differ by model: Llama uses <|eot_id|>, Phi-3 uses <|end|>.
        stop_tokens = []

        # Base eos (Llama: <|end_of_text|>, Phi-3: <|endoftext|>).
        if self.tokenizer.eos_token_id is not None:
            stop_tokens.append(self.tokenizer.eos_token_id)

        # Instruct turn terminators.
        for tok in ("<|eot_id|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                stop_tokens.append(tid)

        return list(set(stop_tokens))

    @staticmethod
    def _build_quantization_config(method: str) -> Optional[BitsAndBytesConfig]:
        if method == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        if method == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

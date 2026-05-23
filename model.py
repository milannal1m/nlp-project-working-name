from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class RunConfig:
    model_name_or_path: str
    quantization_method: str = "None"
    output_dir: str = "./summaries"
    prompt_template: str = "News: {news}\nSummarize the news in two sentences. Summary:"
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
        prompt = self.config.prompt_template.format(news=text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
        ).to(self.model.device)

        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_len = outputs.shape[1] - input_len
        generated_text = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        return generated_text, input_len, generated_len

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

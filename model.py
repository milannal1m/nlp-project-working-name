from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Settings for one generation run: model id, quantization mode, and output label.
@dataclass
class RunConfig:
    model_name_or_path: str
    quantization_method: str = "None"
    output_dir: str = "./summaries"
    prompt_template: str = "News: {news}\nSummarize the news in two sentences. Summary:"
    max_input_length: int = 2048
    max_new_tokens: int = 150
    model_label: Optional[str] = None

    # Default the output label to Llama_<quant> when none is supplied.
    def __post_init__(self):
        if self.model_label is None:
            self.model_label = f"Llama_{self.quantization_method}"


# Loads a causal LM + tokenizer and turns news articles into summaries.
class SummarizationModel:
    # Load the tokenizer and the (optionally quantized) model onto the device.
    def __init__(self, config: RunConfig):
        quant_config = self._build_quantization_config(config.quantization_method)
        print(f"Loading model with quantization method: {config.quantization_method}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=quant_config,
            device_map="auto",
            dtype=torch.float16 if config.quantization_method == "None" else None,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        self.config = config

    # Summarize one article; returns (summary_text, input_len, generated_len).
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

    # Return a bitsandbytes 4bit/8bit config, or None for full precision (fp16).
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

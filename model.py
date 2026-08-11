import re
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATASET_MAX_NEW_TOKENS: dict[str, int] = {
    "cnn_dailymail": 128,
    "xsum": 64,
}

_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?(?:here(?:'s| is)\s+(?:a\s+)?(?:brief\s+|short\s+|concise\s+)?"
    r"summary(?:\s+of\s+the\s+(?:news|article))?\s*[:\-]?|summary\s*[:\-])\s*",
    re.IGNORECASE,
)
_SENT_END_RE = re.compile(r"[.!?][\"')\]]*\s")


def clean_summary(text: str, truncated: bool) -> str:
    original = text.strip()
    cleaned = _PREAMBLE_RE.sub("", original).strip().strip('"').strip()

    if truncated and cleaned:
        ends = list(_SENT_END_RE.finditer(cleaned + " "))
        if ends:
            cleaned = cleaned[: ends[-1].end()].strip()

    return cleaned or original


def extract_after_marker(text: str, marker: Optional[str]) -> str:
    if not marker or marker not in text:
        return text
    return text.rsplit(marker, 1)[1].strip() or text


@dataclass
class RunConfig:
    model_name_or_path: str
    quantization_method: str = "None"
    output_dir: str = "./summaries"
    prompt_template: str = "News: {news}\nSummarize the news in two sentences. Summary:"
    max_input_length: int = 2048
    max_new_tokens: int = 150
    max_new_tokens_map: dict[str, int] = field(
        default_factory=lambda: dict(DATASET_MAX_NEW_TOKENS)
    )
    min_new_tokens: int = 8
    no_repeat_ngram_size: int = 3
    model_label: Optional[str] = None
    adapter_path: Optional[str] = None
    summary_marker: Optional[str] = None

    def __post_init__(self):
        if self.model_label is None:
            self.model_label = f"Llama_{self.quantization_method}"

    def budget_for(self, dataset: Optional[str]) -> int:
        return self.max_new_tokens_map.get(dataset, self.max_new_tokens)


class SummarizationModel:
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

        if config.adapter_path:
            from peft import PeftModel

            print(f"Attaching LoRA adapter: {config.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, config.adapter_path)
            self.model.eval()

        self.config = config

    def summarize(self, text: str, dataset: Optional[str] = None) -> tuple[str, int, int]:
        if self.config.summary_marker:
            prefix, _, suffix = self.config.prompt_template.partition("{news}")
            overhead = len(self.tokenizer(prefix + suffix, add_special_tokens=True).input_ids)
            budget = max(1, self.config.max_input_length - overhead)
            art_ids = self.tokenizer(text, add_special_tokens=False).input_ids[:budget]
            text = self.tokenizer.decode(art_ids)

        prompt = self.config.prompt_template.replace("{news}", text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
        ).to(self.model.device)

        input_len = inputs.input_ids.shape[1]
        max_new = self.config.budget_for(dataset)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                min_new_tokens=self.config.min_new_tokens,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_len = outputs.shape[1] - input_len
        generated_text = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        generated_text = extract_after_marker(generated_text, self.config.summary_marker)

        truncated = generated_len >= max_new
        return clean_summary(generated_text, truncated), input_len, generated_len

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

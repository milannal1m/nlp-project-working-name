import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk # FIXED: Added missing imports
import time

def getquantization_config(quantization_method):
    if quantization_method == '4bit':
        return BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16
        )
    elif quantization_method == '8bit':
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

def data_loader(path, name=None, split="test"):
    try:
        if name:
            return load_dataset(path, name=name, split=split)
        return load_dataset(path, split=split)
    except Exception as e:
        return load_from_disk(path)
    
def main():
    parser = argparse.ArgumentParser(description="Summarization with LLaMA")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained LLaMA model")
    parser.add_argument("--quantization_method", type=str, choices=['None', '4bit', '8bit'], default='None', help="Quantization method to use")
    # FIXED: Removed the unused and required --input_file argument
    args = parser.parse_args()

    # Load tokenizer and model with quantization
    quantization_config = getquantization_config(args.quantization_method)
    print("Loading model with quantization method:", args.quantization_method)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        quantization_config=quantization_config,
        device_map='auto',
        torch_dtype=torch.float16 if args.quantization_method=="None" else None
    )

    # FIXED: Added [:500] to match the paper's 500-sample limit and save massive amounts of time
    dataset = {
        "cnn_dailymail": data_loader("./cnn_dailymail", name="3.0.0", split="test[:500]"),
        "xsum": data_loader("./xsum", split="test[:500]"),
        "newsroom": data_loader("./newsroom", split="test[:500]"),
        "news-qa-summarization": data_loader("./news-qa-summarization", split="train[:500]"),
    }

    prompts = "News: {news}\nSummarize the news in two sentences. Summary:"
    output_dir = "./summaries"
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, data in dataset.items():
        dataset_start_time = time.time()

        try:
            total_samples = len(data)
        except TypeError:
            total_samples = "unknown"

        print("=" * 80, flush=True)
        print(f"Processing dataset: {dataset_name}", flush=True)
        print(f"Total samples to process: {total_samples}", flush=True)

        output_path = os.path.join(
            output_dir,
            f"Llama_{args.quantization_method}_{dataset_name}_summaries.jsonl"
        )

        print(f"Output file: {output_path}", flush=True)

        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            for idx, item in enumerate(data, start=1):
                sample_start_time = time.time()

                qa_list = None

                if dataset_name == "cnn_dailymail":
                    news_text, ref_summary = item["article"], item["highlights"]

                elif dataset_name == "xsum":
                    news_text, ref_summary = item["document"], item["summary"]

                elif dataset_name == "newsroom":
                    news_text, ref_summary = item["text"], item["summary"]

                elif dataset_name == "news-qa-summarization":
                    news_text, ref_summary = item["story"], item["summary"]

                    questions = item.get("questions", [])
                    qa_list = []
                    if isinstance(questions, list):
                        for qa in questions:
                            if isinstance(qa, dict) and "q" in qa and "a" in qa:
                                qa_list.append({"q": qa["q"], "a": qa["a"]})
                

                else:
                    raise ValueError(f"Unsupported dataset: {dataset_name}")

                prompt = prompts.format(news=news_text)

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(model.device)

                input_token_len = inputs.input_ids.shape[1]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                generated_token_len = outputs.shape[1] - input_token_len

                generated_text = tokenizer.decode(
                    outputs[0][input_token_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                ).strip()

                result = {
                    "news": news_text,
                    "reference_summary": ref_summary,
                    "generated_summary": generated_text
                }

                if dataset_name == "news-qa-summarization":
                    result["qa_pairs"] = qa_list

                f.write(json.dumps(result, ensure_ascii=False) + "\n")

                elapsed = time.time() - sample_start_time

                if idx == 1 or idx % 10 == 0:
                    print(
                        f"[{dataset_name}] "
                        f"sample {idx}/{total_samples} | "
                        f"input_tokens={input_token_len} | "
                        f"generated_tokens={generated_token_len} | "
                        f"time={elapsed:.2f}s",
                        flush=True
                    )

        dataset_elapsed = time.time() - dataset_start_time

        print(
            f"Finished dataset: {dataset_name} | "
            f"time={dataset_elapsed / 60:.2f} min | "
            f"saved to {output_path}",
            flush=True
        )

if __name__ == "__main__":
    main()
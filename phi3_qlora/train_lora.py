import argparse

import glob

import os

import re

import time


import torch

from datasets import Dataset, load_from_disk

from transformers import (

    AutoModelForCausalLM,

    AutoTokenizer,

    BitsAndBytesConfig,

    Trainer,

    TrainingArguments,

)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


from dataset import load_pairs

from model import RunConfig


PROMPT_TEMPLATE = RunConfig.prompt_template


PHI3_TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def local_rank() -> int:


    return int(os.environ.get("LOCAL_RANK", 0))


def is_main() -> bool:


    return int(os.environ.get("RANK", 0)) == 0


def log(msg: str) -> None:

    if is_main():

        print(msg, flush=True)


def allow_own_checkpoint_load() -> None:


    try:

        import transformers.trainer as _trainer


        _trainer.check_torch_load_is_safe = lambda *a, **k: None

    except Exception as e:

        print(f"[warn] could not relax torch.load guard: {e}", flush=True)


def latest_checkpoint(trainer_dir: str):


    cks = glob.glob(os.path.join(trainer_dir, "checkpoint-*"))

    cks = [c for c in cks if os.path.isdir(c) and re.search(r"checkpoint-(\d+)$", c)]

    if not cks:

        return None

    return max(cks, key=lambda c: int(re.search(r"checkpoint-(\d+)$", c).group(1)))


def build_tokenizer(base_model: str):

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)

    if tok.pad_token is None:

        tok.pad_token = tok.eos_token

    tok.padding_side = "right"

    return tok


def make_tokenize_fn(tok, max_seq_length: int, max_summary_tokens: int):


    overhead = len(tok(PROMPT_TEMPLATE.format(news=""), add_special_tokens=True).input_ids)


    art_budget = max_seq_length - overhead - max_summary_tokens - 1


    def tokenize(example):

        art_ids = tok(example["article"], add_special_tokens=False).input_ids[:art_budget]

        prompt = PROMPT_TEMPLATE.format(news=tok.decode(art_ids))

        prompt_ids = tok(prompt, add_special_tokens=True).input_ids


        completion_ids = tok(" " + example["summary"], add_special_tokens=False).input_ids[

            :max_summary_tokens

        ] + [tok.eos_token_id]

        input_ids = prompt_ids + completion_ids

        labels = [-100] * len(prompt_ids) + completion_ids

        return {"input_ids": input_ids, "labels": labels}


    return tokenize


def make_collate_fn(tok):


    pad_id = tok.pad_token_id


    def collate(batch):

        maxlen = max(len(x["input_ids"]) for x in batch)

        maxlen = (maxlen + 7) // 8 * 8

        input_ids, attention_mask, labels = [], [], []

        for x in batch:

            n = len(x["input_ids"])

            pad = maxlen - n

            input_ids.append(x["input_ids"] + [pad_id] * pad)

            attention_mask.append([1] * n + [0] * pad)

            labels.append(x["labels"] + [-100] * pad)

        return {

            "input_ids": torch.tensor(input_ids, dtype=torch.long),

            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),

            "labels": torch.tensor(labels, dtype=torch.long),

        }


    return collate


def build_dataset(tok, datasets: list[str], per_dataset_samples, seed: int,

                  max_seq_length: int, max_summary_tokens: int) -> Dataset:


    articles, summaries = [], []

    for name in datasets:

        pairs = load_pairs(name, split="train", limit=per_dataset_samples, seed=seed)

        print(f"  [{name}] {len(pairs)} train pairs", flush=True)

        for art, summ in pairs:

            articles.append(art)

            summaries.append(summ)

    ds = Dataset.from_dict({"article": articles, "summary": summaries})

    tokenize = make_tokenize_fn(tok, max_seq_length, max_summary_tokens)

    return ds.map(tokenize, remove_columns=["article", "summary"], desc="tokenizing",

                  num_proc=min(8, os.cpu_count() or 1))


def load_quantized_model(base_model: str, compute_dtype):


    bnb = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_use_double_quant=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=compute_dtype,

    )

    model = AutoModelForCausalLM.from_pretrained(

        base_model,

        quantization_config=bnb,


        device_map={"": local_rank()},

        trust_remote_code=False,

        attn_implementation="eager",

    )

    model = prepare_model_for_kbit_training(

        model,

        use_gradient_checkpointing=True,

        gradient_checkpointing_kwargs={"use_reentrant": False},

    )

    if hasattr(model.config, "pretraining_tp"):

        model.config.pretraining_tp = 1

    return model


def parse_args():

    p = argparse.ArgumentParser(description="QLoRA fine-tune Phi-3-mini for summarization")

    p.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct")

    p.add_argument("--datasets", type=str, default="cnn_dailymail,xsum",

                   help="Comma-separated dataset names to combine (balanced per dataset)")

    p.add_argument("--per_dataset_samples", type=int, default=50000,

                   help="Pairs per dataset; 0 means the FULL train split of each")

    p.add_argument("--tokenized_path", type=str, default=None,

                   help="Dir of a pre-tokenized dataset: loaded if present, else built here")

    p.add_argument("--prepare_only", action="store_true",

                   help="Build+tokenize into --tokenized_path and exit (CPU prep job)")

    p.add_argument("--output_dir", type=str, default="adapters/phi3-lora-news")

    p.add_argument("--epochs", type=float, default=1.0)

    p.add_argument("--max_steps", type=int, default=-1, help="Overrides --epochs when > 0")

    p.add_argument("--max_seq_length", type=int, default=1024)

    p.add_argument("--save_steps", type=int, default=250,

                   help="Checkpoint interval; a killed job resumes from the last one")

    p.add_argument("--max_summary_tokens", type=int, default=128)

    p.add_argument("--lora_r", type=int, default=16)

    p.add_argument("--lora_alpha", type=int, default=32)

    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--lr", type=float, default=2e-4)

    p.add_argument("--batch_size", type=int, default=4)

    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--warmup_ratio", type=float, default=0.03)

    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--max_grad_norm", type=float, default=0.3)


    p.add_argument("--optim", type=str, default="adamw_torch",

                   help="Optimizer; paged_* variants cannot resume from checkpoint")

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--force", action="store_true", help="Retrain even if an adapter exists")

    return p.parse_args()


def resolve_dataset(args, tok) -> Dataset:


    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if args.tokenized_path and os.path.isdir(args.tokenized_path):

        ds = load_from_disk(args.tokenized_path)

        log(f"  loaded pre-tokenized dataset from {args.tokenized_path}")

        return ds


    if int(os.environ.get("WORLD_SIZE", 1)) > 1:

        raise SystemExit(

            f"Multi-GPU run requires a pre-tokenized dataset, but "

            f"{args.tokenized_path!r} is missing. Run the prep job first:\n"

            f"  python train_lora.py --prepare_only --per_dataset_samples "

            f"{args.per_dataset_samples} --max_seq_length {args.max_seq_length} "

            f"--tokenized_path {args.tokenized_path}"

        )


    n = args.per_dataset_samples or None

    log(f"=== building train set from {datasets} "

        f"({'FULL split' if n is None else f'{n}/dataset'}) ===")

    ds = build_dataset(tok, datasets, n, args.seed,

                       args.max_seq_length, args.max_summary_tokens)

    if args.tokenized_path and is_main():

        ds.save_to_disk(args.tokenized_path)

        log(f"  saved tokenized dataset -> {args.tokenized_path}")

    return ds


def main():

    args = parse_args()


    adapter_marker = os.path.join(args.output_dir, "adapter_config.json")

    if os.path.exists(adapter_marker) and not args.force and not args.prepare_only:

        log(f"[skip] adapter already exists: {args.output_dir} (use --force to retrain)")

        return


    if args.prepare_only:

        if not args.tokenized_path:

            raise SystemExit("--prepare_only requires --tokenized_path")

        if os.path.isdir(args.tokenized_path):

            log(f"[skip] tokenized dataset already exists: {args.tokenized_path}")

            return

        tok = build_tokenizer(args.base_model)

        ds = resolve_dataset(args, tok)

        log(f"=== prepared {len(ds)} tokenized examples -> {args.tokenized_path} ===")

        return


    if torch.cuda.is_available():

        log(f"Using GPU: {torch.cuda.get_device_name(local_rank())} "

            f"(world_size={os.environ.get('WORLD_SIZE', 1)})")

    else:


        raise SystemExit("No GPU found — QLoRA training requires a CUDA device (run on the cluster).")


    torch.manual_seed(args.seed)

    use_bf16 = torch.cuda.is_bf16_supported()

    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    log(f"=== QLoRA fine-tune {args.base_model} | bf16={use_bf16} ===")


    tok = build_tokenizer(args.base_model)

    train_ds = resolve_dataset(args, tok)

    log(f"  {len(train_ds)} tokenized examples")


    model = load_quantized_model(args.base_model, compute_dtype)

    lora_config = LoraConfig(

        r=args.lora_r,

        lora_alpha=args.lora_alpha,

        lora_dropout=args.lora_dropout,

        bias="none",

        task_type="CAUSAL_LM",

        target_modules=PHI3_TARGET_MODULES,

    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()


    trainer_dir = os.path.join(args.output_dir, "trainer_logs")

    training_args = TrainingArguments(

        output_dir=trainer_dir,

        num_train_epochs=args.epochs,

        max_steps=args.max_steps,

        per_device_train_batch_size=args.batch_size,

        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,

        lr_scheduler_type="cosine",

        warmup_ratio=args.warmup_ratio,

        weight_decay=args.weight_decay,

        max_grad_norm=args.max_grad_norm,

        optim=args.optim,

        bf16=use_bf16,

        fp16=not use_bf16,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={"use_reentrant": False},

        remove_unused_columns=False,

        logging_steps=20,


        save_strategy="steps",

        save_steps=args.save_steps,

        save_total_limit=2,

        ddp_find_unused_parameters=False,

        report_to="none",

        seed=args.seed,

    )


    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_ds,

        data_collator=make_collate_fn(tok),

    )


    resume = latest_checkpoint(trainer_dir)

    if resume:

        allow_own_checkpoint_load()

        log(f"[resume] continuing from {resume}")


    start = time.time()

    trainer.train(resume_from_checkpoint=resume)

    log(f"=== Training done in {(time.time() - start) / 60:.1f} min ===")


    if trainer.is_world_process_zero():

        os.makedirs(args.output_dir, exist_ok=True)

        model.save_pretrained(args.output_dir)

        tok.save_pretrained(args.output_dir)

        print(f"[ckpt] adapter saved -> {args.output_dir}", flush=True)


if __name__ == "__main__":

    main()

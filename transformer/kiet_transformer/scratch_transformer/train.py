import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset import extract_fields
from transformer.kiet_transformer.scratch_transformer.data import (
    build_tokenizer,
    load_pairs,
    load_train_split,
    make_collate_fn,
)
from transformer.kiet_transformer.scratch_transformer.model import Transformer

DATASET_BY_INDEX = ["cnn_dailymail", "xsum"]
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


class PairStream(IterableDataset):

    def __init__(self, name: str, sample, seed: int):
        self.name = name
        self.sample = sample
        self.seed = seed

    def __iter__(self):
        stream = load_train_split(self.name, split="train", sample=self.sample, seed=self.seed)
        for item in stream:
            news, ref, _ = extract_fields(self.name, item)
            if news and ref:
                yield (news, ref)


def iter_batches(loader):
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError("train stream produced no batches — check --dataset / --sample")


def evaluate_val(model, val_pairs, collate, batch_size, criterion, vocab, device) -> float:
    if not val_pairs:
        return float("inf")
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_pairs), batch_size):
            chunk = val_pairs[i:i + batch_size]
            src, trg = collate(chunk)
            src, trg = src.to(device), trg.to(device)
            logits = model(src, trg[:, :-1])
            loss = criterion(logits.reshape(-1, vocab), trg[:, 1:].reshape(-1))
            total += loss.item() * len(chunk)
            n += len(chunk)
    model.train()
    return total / max(1, n)


def save_checkpoint(path, model, arch, args, tok, dataset_name, val_loss, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": arch,
            "max_src_len": args.max_src_len,
            "max_trg_len": args.max_trg_len,
            "vocab_size": len(tok),
            "pad_idx": tok.pad_token_id,
            "bos_idx": tok.bos_token_id,
            "eos_idx": tok.eos_token_id,
            "dataset": dataset_name,
            "val_loss": val_loss,
            "step": step,
        },
        path,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train the from-scratch Transformer summarizer")
    p.add_argument("--dataset_index", type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
                   help="0=cnn_dailymail, 1=xsum (defaults to $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--dataset", type=str, default=None, help="Override dataset name directly")
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_src_len", type=int, default=400)
    p.add_argument("--max_trg_len", type=int, default=64)
    p.add_argument("--embed_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--forward_expansion", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=4000)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--val_size", type=int, default=1000)
    p.add_argument("--val_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--sample", type=int, default=None, help="Cap the train stream (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cpu / cuda (auto if omitted)")
    p.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    p.add_argument("--force", action="store_true", help="Retrain even if a checkpoint exists")
    return p.parse_args()


def main():
    args = parse_args()
    name = args.dataset or DATASET_BY_INDEX[args.dataset_index]
    ckpt_path = os.path.join(args.checkpoint_dir, f"transformer_{name}.pt")

    if os.path.exists(ckpt_path) and not args.force:
        print(f"[skip] checkpoint already exists: {ckpt_path} (use --force to retrain)", flush=True)
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    print(f"=== Training Transformer on {name} | device={device} ===", flush=True)

    tok = build_tokenizer()
    vocab = len(tok)
    pad = tok.pad_token_id
    max_length = max(args.max_src_len, args.max_trg_len)
    arch = {
        "embed_size": args.embed_size,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "forward_expansion": args.forward_expansion,
        "dropout": args.dropout,
        "max_length": max_length,
    }

    model = Transformer(
        src_vocab_size=vocab, trg_vocab_size=vocab,
        src_pad_idx=pad, trg_pad_idx=pad, device=device, **arch,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  vocab={vocab} pad={pad} params={n_params/1e6:.1f}M max_length={max_length}", flush=True)

    collate = make_collate_fn(tok, args.max_src_len, args.max_trg_len)
    loader = DataLoader(
        PairStream(name, args.sample, args.seed),
        batch_size=args.batch_size, collate_fn=collate, num_workers=0,
    )

    print(f"  loading {args.val_size} validation pairs...", flush=True)
    val_pairs = load_pairs(name, split="validation", limit=args.val_size, seed=args.seed)
    print(f"  got {len(val_pairs)} validation pairs", flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=pad, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9,
        weight_decay=args.weight_decay,
    )
    warmup = max(1, args.warmup)

    def lr_lambda(step):
        step = max(1, step)
        return min(step / warmup, (warmup / step) ** 0.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    patience_left = args.patience
    saved = False
    step = 0
    start = time.time()
    model.train()

    batches = iter_batches(loader)
    while step < args.max_steps:
        src, trg = next(batches)
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(src, trg[:, :-1])
            loss = criterion(logits.reshape(-1, vocab), trg[:, 1:].reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

        if step % args.log_every == 0:
            rate = step / (time.time() - start)
            print(f"  step {step}/{args.max_steps} | loss {loss.item():.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {rate:.1f} it/s", flush=True)

        if step % args.val_every == 0 or step == args.max_steps:
            val = evaluate_val(model, val_pairs, collate, args.batch_size, criterion, vocab, device)
            print(f"  [val] step {step} | val_loss {val:.4f} | best {best_val:.4f}", flush=True)
            if val < best_val:
                best_val = val
                save_checkpoint(ckpt_path, model, arch, args, tok, name, best_val, step)
                saved = True
                patience_left = args.patience
                print(f"  [ckpt] improved -> saved {ckpt_path}", flush=True)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  [early-stop] no improvement for {args.patience} evals", flush=True)
                    break

    if not saved:
        final_val = evaluate_val(model, val_pairs, collate, args.batch_size, criterion, vocab, device)
        save_checkpoint(ckpt_path, model, arch, args, tok, name, final_val, step)
        print(f"  [ckpt] saved final weights -> {ckpt_path} (val_loss {final_val:.4f})", flush=True)

    print(f"=== Done {name} in {(time.time() - start)/60:.1f} min | best_val {best_val:.4f} ===", flush=True)


if __name__ == "__main__":
    main()

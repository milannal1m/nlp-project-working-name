"""Train a from-scratch Transformer to summarize XSum + CNN/DailyMail (small subset).

Two model choices (--model):
  * lecture — the encoder-decoder lecture_transformer.py, greedy decoding.
  * milan   — the decoder-only milan_transformer.py, temperature + top-p sampling.
             The article and summary are fed as one stream "article [SEP] summary";
             loss is only counted on the summary tokens.

Trains ONE combined model on TRAIN_N examples of each dataset, generates EVAL_N of each,
and scores them with the repo's full Evaluator (BLEU, ROUGE-L, METEOR, BERTScore, + QA).
Summaries and results are written under this directory (milan_transformer/outputs/).

Why cross-entropy? The model emits logits (batch, len, vocab): at each position it predicts a
distribution over the vocabulary. Summarization, like translation, is conditional next-token
classification, so cross-entropy maximizes the log-likelihood of the gold next token (padding
positions ignored).

Run from the repo root:
    python transformer/milan_transformer/pipeline.py --model milan
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)                             # lecture_transformer.py, milan_transformer.py
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))   # dataset.py, evaluator.py

from lecture_transformer import Transformer as EncoderDecoder
from milan_transformer import Transformer as DecoderOnly
from dataset import extract_fields, load_datasets_streaming
from evaluator import Evaluator

# --- config -----------------------------------------------------------------
DATASETS = ["xsum", "cnn_dailymail"]
TRAIN_N, EVAL_N = 100000, 600
EPOCHS, BATCH, LR = 100, 16, 3e-4
MAX_SRC, MAX_TRG = 400, 64
EMBED, LAYERS, HEADS = 256, 3, 8
TEMPERATURE, TOP_P = 0.7, 0.9        # decoder-only (milan) sampling knobs
TOKENIZER, SEED = "bert-base-uncased", 42
OUT_DIR = os.path.join(HERE, "outputs")


def collate_encdec(batch, pad):
    """Pad the (src, trg) id lists in a batch to their max lengths with `pad`."""
    srcs, trgs = zip(*batch)
    s_max, t_max = max(len(s) for s in srcs), max(len(t) for t in trgs)
    src = torch.full((len(batch), s_max), pad, dtype=torch.long)
    trg = torch.full((len(batch), t_max), pad, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, trgs)):
        src[i, : len(s)] = torch.tensor(s)
        trg[i, : len(t)] = torch.tensor(t)
    return src, trg


def collate_declm(batch, pad):
    """Pad the concatenated sequences to the batch max; carry each prompt length."""
    seqs, prompt_lens = zip(*batch)
    L = max(len(s) for s in seqs)
    seq = torch.full((len(batch), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        seq[i, : len(s)] = torch.tensor(s)
    return seq, list(prompt_lens)


def train(model, loader, step, device):
    """Training loop. `step(model, batch)` returns the loss for one batch."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for batch in loader:
            loss = step(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        print(f"  epoch {epoch + 1:>3}/{EPOCHS}  loss {total / len(loader):.4f}", flush=True)


@torch.no_grad()
def greedy(model, src_ids, sos, eos, device):
    """Encoder-decoder greedy decode (mirrors the lecture demo's argmax loop)."""
    model.eval()
    src = torch.tensor([src_ids], device=device)
    out = [sos]
    for _ in range(MAX_TRG):
        logits = model(src, torch.tensor([out], device=device))
        nxt = logits.argmax(dim=2)[:, -1].item()
        out.append(nxt)
        if nxt == eos:
            break
    return out[1:]                                      # drop leading SOS


def sanity_check_with_gold(files, out_path, n, seed):
    """Local fork of analyze_sanity_check that ALSO prints the gold summary.

    Kept separate so the shared analysis.analyze_sanity_check stays unchanged. Uses the
    same shared record indices so gold/generated line up across files.
    """
    import random
    indices, lines = None, ["# Sanity check — gold vs generated", ""]
    for fp in sorted(files):
        rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        if indices is None and rows:
            indices = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
        lines.append(f"## {os.path.basename(fp)}")
        lines.append("")
        for i in indices or []:
            if i < len(rows):
                lines += [f"### Summary {i}", "",
                          f"**Gold:** {rows[i].get('reference_summary', '').strip()}", "",
                          f"**Generated:** {rows[i].get('generated_summary', '').strip()}", ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}", flush=True)


def build_lecture(train_pairs, tok, device):
    """Encoder-decoder setup. Returns (model, loader, step, generate) for the shared loop.

    step/generate are closures so they can capture this model, its tokenizer and device.
    """
    pad, sos, eos, vocab = tok.pad_token_id, tok.cls_token_id, tok.sep_token_id, tok.vocab_size
    encode = lambda text, n: tok(text, truncation=True, max_length=n)["input_ids"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad)

    rows = [(encode(n, MAX_SRC), encode(r, MAX_TRG)) for n, r in train_pairs]
    loader = DataLoader(rows, batch_size=BATCH, shuffle=True,
                        collate_fn=lambda b: collate_encdec(b, pad))
    model = EncoderDecoder(
        src_vocab_size=vocab, trg_vocab_size=vocab, src_pad_idx=pad, trg_pad_idx=pad,
        embed_size=EMBED, num_layers=LAYERS, heads=HEADS, forward_expansion=4,
        dropout=0.1, device=str(device), max_length=MAX_SRC,
    ).to(device)

    def step(model, batch):                                    # loss for one (src, trg) batch
        src, trg = batch[0].to(device), batch[1].to(device)
        logits = model(src, trg[:, :-1])                       # (N, T-1, vocab)
        return criterion(logits.reshape(-1, vocab), trg[:, 1:].reshape(-1))

    def generate(news):                                        # greedy decode one article
        return tok.decode(greedy(model, encode(news, MAX_SRC), sos, eos, device),
                          skip_special_tokens=True)

    return model, loader, step, generate


def build_milan(train_pairs, tok, device):
    """Decoder-only setup: one concatenated stream "article [SEP] summary".

    Returns (model, loader, step, generate); step masks the loss to the summary tokens.
    """
    pad, eos, vocab = tok.pad_token_id, tok.sep_token_id, tok.vocab_size
    encode = lambda text, n: tok(text, truncation=True, max_length=n)["input_ids"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad)

    def concat(news, ref):
        art = encode(news, MAX_SRC)          # [CLS] article ... [SEP]
        summ = encode(ref, MAX_TRG)[1:]      # drop the summary's own leading [CLS]
        return art + summ, len(art)          # (sequence, prompt length)

    rows = [concat(n, r) for n, r in train_pairs]
    loader = DataLoader(rows, batch_size=BATCH, shuffle=True,
                        collate_fn=lambda b: collate_declm(b, pad))
    model = DecoderOnly(
        vocab_size=vocab, pad_idx=pad, embed_size=EMBED, num_layers=LAYERS, heads=HEADS,
        forward_expansion=4, dropout=0.1, device=str(device), max_length=MAX_SRC + MAX_TRG,
    ).to(device)

    def step(model, batch):                                    # loss for one concatenated batch
        seq, prompt_lens = batch[0].to(device), batch[1]
        logits = model(seq[:, :-1])                            # (N, L-1, vocab)
        labels = seq[:, 1:].clone()
        for i, p_len in enumerate(prompt_lens):                # count loss on summary tokens only
            labels[i, : p_len - 1] = pad
        return criterion(logits.reshape(-1, vocab), labels.reshape(-1))

    def generate(news):                                        # temperature + top-p sample one article
        prompt = torch.tensor([encode(news, MAX_SRC)], device=device)
        out = model.generate(prompt, MAX_TRG, eos, temperature=TEMPERATURE, top_p=TOP_P)
        return tok.decode(out[0, prompt.shape[1]:], skip_special_tokens=True)

    return model, loader, step, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lecture", "milan"], default="lecture",
                        help="lecture = encoder-decoder (greedy); milan = decoder-only (temp+top-p).")
    kind = parser.parse_args().model
    torch.manual_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Model: {kind} | Device: {device}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    # Reuse the repo loader: train on the train split, evaluate on the test split (disjoint).
    train_raw = load_datasets_streaming(sample=TRAIN_N, seed=SEED, names=DATASETS, split="train")
    test_raw = load_datasets_streaming(sample=EVAL_N, seed=SEED, names=DATASETS, split="test")
    train_pairs, eval_pairs = [], {}
    for name in DATASETS:
        train_pairs += [extract_fields(name, it)[:2] for it in train_raw[name]]   # [(news, ref), ...]
        eval_pairs[name] = [extract_fields(name, it)[:2] for it in test_raw[name]]

    # The two architectures need different batch shapes, losses and decoding, so a per-model
    # builder returns the pieces the shared train/eval code below consumes.
    build = build_lecture if kind == "lecture" else build_milan
    model, loader, step, generate = build(train_pairs, tok, device)
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | "
          f"{len(loader.dataset)} train examples", flush=True)
    train(model, loader, step, device)

    # --- evaluate: write summaries + score with the repo's full Evaluator, all under OUT_DIR ---
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, f"{kind}_evaluation.log")
    csv_path = os.path.join(OUT_DIR, f"{kind}_evaluation.csv")
    for p in (log_path, csv_path):                     # fresh results for this run
        if os.path.exists(p):
            os.remove(p)

    evaluator = Evaluator()
    files = []
    print("\nGenerating + evaluating...", flush=True)
    for name in DATASETS:
        out_path = os.path.join(OUT_DIR, f"{kind}_{name}_summaries.jsonl")
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for news, ref in eval_pairs[name]:
                f.write(json.dumps({"news": news, "reference_summary": ref,
                                    "generated_summary": generate(news)}, ensure_ascii=False) + "\n")
        files.append(out_path)
        metrics = evaluator.run_and_log(out_path, log_path, csv_path)
        print(f"  {name}: " + "  ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)), flush=True)

    # --- readable spot-check in Markdown: gold vs generated, side by side ---
    sanity_check_with_gold(files, os.path.join(OUT_DIR, f"{kind}_sanity_check_gold.md"), n=5, seed=SEED)

    print(f"\nSummaries + results saved under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

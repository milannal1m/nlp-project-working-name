"""Transformer-based abstractive summarization baseline.

This module turns the from-scratch encoder-decoder Transformer in this file into
a trainable abstractive summarization baseline that plugs directly into the
existing news-summarization benchmark (``src/``). It keeps the original model
architecture and class hierarchy (``SelfAttention``, ``TransformerBlock``,
``Encoder``, ``DecoderBlock``, ``Decoder``, ``Transformer``) and adds
tokenization, dataset loading, batching, training, checkpointing, greedy
decoding, CLI parsing, and benchmark-compatible result writing around it.

The baseline:
  * trains a separate model per dataset (``cnn_dailymail`` or ``xsum``) from
    scratch on the dataset's official ``train`` split only;
  * generates summaries on the dataset's official ``test`` split only, using the
    exact same sampled test articles as the rest of the benchmark;
  * writes ``Transformer/summaries/Transformer_{dataset}_{sample}_summaries.jsonl``
    using the benchmark's own naming API and JSONL schema. All outputs stay under
    ``Transformer/`` so they never mix with the original benchmark; the schema is
    still identical, so the benchmark evaluator can score them when pointed at this
    directory (``--task evaluate`` handles that automatically).

Example
-------
    python Transformer/Main.py --task all --dataset cnn_dailymail --sample 100
    python Transformer/Main.py --task train --dataset xsum --sample 50
    python Transformer/Main.py --task summarize --dataset xsum --sample 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Paths and benchmark imports
# --------------------------------------------------------------------------- #
# Resolve the repository layout so this script works regardless of the current
# working directory, and so it can reuse the benchmark's dataset/naming logic as
# the single source of truth (without modifying anything under ``src/``).
TRANSFORMER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRANSFORMER_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
# All Transformer-baseline outputs stay under Transformer/ so they never mix
# with the original benchmark's root-level summaries/ and results/ directories.
# VARIANT prefixes every artifact/summary path so ablation runs never collide.
VARIANT = "E0"
CHECKPOINT_DIR = os.path.join(TRANSFORMER_DIR, "checkpoints")
ARTIFACT_DIR = os.path.join(TRANSFORMER_DIR, "artifacts")
DEFAULT_SUMMARY_DIR = os.path.join(TRANSFORMER_DIR, "summaries", VARIANT)
DEFAULT_LOG_PATH = os.path.join(TRANSFORMER_DIR, "results", VARIANT, "evaluation.log")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    # Reuse the benchmark's dataset configuration, field extraction and naming
    # so the Transformer baseline stays consistent with every other experiment.
    from dataset import DATASET_CONFIGS, extract_fields, load_datasets_streaming
    from naming import baseline_filename
except ImportError as exc:  # pragma: no cover - defensive, gives an actionable error.
    raise SystemExit(
        "Could not import the benchmark modules from 'src/'. "
        f"Expected them under: {SRC_DIR}\n"
        "Run this script from inside the repository so 'src/dataset.py' and "
        "'src/naming.py' are importable.\n"
        f"Underlying error: {exc}"
    )

# The Transformer baseline supports exactly the two active abstractive datasets.
SUPPORTED_DATASETS = ["cnn_dailymail", "xsum"]
# 1 is included as a fast smoke-test size to quickly verify the pipeline runs.
SAMPLE_CHOICES = [1, 20, 50, 100, 500]

# Special tokens. Order fixes their integer ids; both vocabularies share them.
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


# =========================================================================== #
# Model architecture (kept intact; only correctness fixes applied)
# =========================================================================== #
class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int) -> None:
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # number of examples in the batch
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        if mask is not None:
            # Positions where the mask is False/0 must not be attended to.
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Scale by sqrt(head_dim) (per-head dimension), not sqrt(embed_size).
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N,
            query_len,
            self.heads * self.head_dim,
        )  # (N, query_len, embed_size)

        out = self.fc_out(out)  # (N, query_len, embed_size)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: int) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        device,
        forward_expansion: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        # Absolute positions 0..seq_length-1 for every row; padded positions get a
        # real embedding but are ignored downstream via the source padding mask.
        positions = torch.arange(0, seq_length, device=self.device).expand(N, seq_length)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: int, device) -> None:
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # Masked self-attention over the target (causal + target padding mask).
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        # Cross-attention over the encoder output (source padding mask).
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        forward_expansion: int,
        dropout: float,
        device,
        max_length: int,
    ) -> None:
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length, device=self.device).expand(N, seq_length)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        src_pad_idx: int,
        trg_pad_idx: int,
        embed_size: int = 256,
        num_layers: int = 6,
        forward_expansion: int = 4,
        heads: int = 8,
        dropout: float = 0.0,
        device="cpu",
        max_length: int = 100,
    ) -> None:
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # (N, 1, 1, src_len) — True where the token is a real (non-pad) token.
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # Combine the causal (no-peeking-ahead) mask with the target padding mask
        # so the decoder attends neither to future tokens nor to pad positions.
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # (N,1,1,trg_len)
        causal_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)
        ).bool()  # (trg_len, trg_len)
        trg_mask = trg_pad_mask & causal_mask  # broadcast -> (N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


# =========================================================================== #
# Reproducibility and device helpers
# =========================================================================== #
def set_seed(seed: int) -> None:
    """Seed Python, PyTorch (CPU) and PyTorch (CUDA) for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def resolve_device(device_arg: Optional[str]) -> torch.device:
    """Return the requested device, or auto-select CUDA when available."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================== #
# Word-level tokenizer and vocabulary
# =========================================================================== #
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Deterministic, lowercase word-level tokenizer for English news text.

    Words and individual punctuation marks become separate tokens. No pretrained
    tokenizer is used, so the vocabulary stays fully reproducible from the data.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def build_vocab(texts: Sequence[str], max_size: int) -> Dict[str, int]:
    """Build a token->id vocabulary from ``texts`` (training portion only).

    The four special tokens always occupy ids 0-3. Remaining slots are filled by
    the most frequent tokens, with ties broken alphabetically so the vocabulary
    is identical across runs on the same data.
    """
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab: Dict[str, int] = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    for token, _count in ordered:
        if len(vocab) >= max_size:
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    """Encode text to token ids (no special tokens), truncated to ``max_len``."""
    ids = [vocab.get(token, UNK_ID) for token in tokenize(text)][:max_len]
    if not ids:
        # Guarantee a non-empty sequence so positional embeddings never see len 0.
        ids = [UNK_ID]
    return ids


def build_target_sequence(text: str, vocab: Dict[str, int], max_content_len: int) -> List[int]:
    """Return ``<sos> <content...> <eos>`` ids for a reference summary."""
    content = [vocab.get(token, UNK_ID) for token in tokenize(text)][:max_content_len]
    return [SOS_ID] + content + [EOS_ID]


def detokenize(tokens: Sequence[str]) -> str:
    """Join word-level tokens into a readable string (light punctuation fixup)."""
    text = " ".join(tokens)
    # Remove the space introduced before standard punctuation marks.
    text = re.sub(r"\s+([.,;:!?%])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def save_vocab(vocab: Dict[str, int], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================== #
# Dataset loading and batching
# =========================================================================== #
def load_train_examples(dataset_name: str, sample: int, seed: int) -> List[Tuple[str, str]]:
    """Load ``sample`` (article, reference) pairs from the official TRAIN split.

    This mirrors ``load_datasets_streaming`` in ``src/dataset.py`` exactly
    (stream -> shuffle(seed) -> take(sample)) but reads the ``train`` split, which
    that helper does not expose. Field extraction is delegated to the benchmark's
    ``extract_fields`` so the article/summary mapping stays authoritative. The
    test split is never touched here.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    if "local_path" in cfg:
        raise SystemExit(
            f"Dataset '{dataset_name}' is a local fixed-sample dataset without a "
            "train split; the Transformer baseline supports only 'cnn_dailymail' "
            "and 'xsum'."
        )

    from datasets import load_dataset  # lazy import: only needed for HuggingFace sources

    kwargs = {"split": "train", "streaming": True}
    if "name" in cfg:
        kwargs["name"] = cfg["name"]
    stream = load_dataset(cfg["path"], **kwargs).shuffle(seed=seed).take(sample)

    examples: List[Tuple[str, str]] = []
    for item in stream:
        news_text, ref_summary, _qa = extract_fields(dataset_name, item)
        examples.append((news_text, ref_summary))
    print(f"  [ready] {dataset_name} (train split, {len(examples)} examples)", flush=True)
    return examples


class SummarizationDataset(Dataset):
    """Encoded (source, target) pairs for teacher-forced training."""

    def __init__(
        self,
        examples: Sequence[Tuple[str, str]],
        src_vocab: Dict[str, int],
        trg_vocab: Dict[str, int],
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.samples: List[Tuple[List[int], List[int]]] = []
        for news_text, ref_summary in examples:
            src_ids = encode(news_text, src_vocab, max_source_length)
            trg_ids = build_target_sequence(ref_summary, trg_vocab, max_target_length)
            self.samples.append((src_ids, trg_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids, trg_ids = self.samples[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)


def _pad_batch(sequences: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    """Pad a list of 1-D LongTensors to the batch's longest sequence."""
    max_len = max(seq.size(0) for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, : seq.size(0)] = seq
    return out


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Padding-aware collate: pad sources and targets independently."""
    src_seqs = [item[0] for item in batch]
    trg_seqs = [item[1] for item in batch]
    return _pad_batch(src_seqs, PAD_ID), _pad_batch(trg_seqs, PAD_ID)


# =========================================================================== #
# Model construction and (de)serialization
# =========================================================================== #
def build_model(config: Dict, device: torch.device) -> Transformer:
    """Instantiate the Transformer from a hyperparameter/config dict."""
    model = Transformer(
        src_vocab_size=config["src_vocab_size"],
        trg_vocab_size=config["trg_vocab_size"],
        src_pad_idx=PAD_ID,
        trg_pad_idx=PAD_ID,
        embed_size=config["embed_size"],
        num_layers=config["num_layers"],
        forward_expansion=config["forward_expansion"],
        heads=config["heads"],
        dropout=config["dropout"],
        device=device,
        max_length=config["pos_max_length"],
    ).to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_path(dataset_name: str, train_sample: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f"Transformer_{VARIANT}_{dataset_name}_{train_sample}.pt")


def src_vocab_path(dataset_name: str, train_sample: int) -> str:
    return os.path.join(ARTIFACT_DIR, f"Transformer_{VARIANT}_{dataset_name}_{train_sample}_src_vocab.json")


def trg_vocab_path(dataset_name: str, train_sample: int) -> str:
    return os.path.join(ARTIFACT_DIR, f"Transformer_{VARIANT}_{dataset_name}_{train_sample}_trg_vocab.json")


def metadata_path(dataset_name: str, train_sample: int) -> str:
    return os.path.join(ARTIFACT_DIR, f"Transformer_{VARIANT}_{dataset_name}_{train_sample}_metadata.json")


# =========================================================================== #
# Training
# =========================================================================== #
def train_model(args: argparse.Namespace) -> Tuple[Transformer, Dict[str, int], Dict[str, int], Dict]:
    """Train the Transformer on the official train split and save a checkpoint.

    Returns the trained model, source/target vocabularies and the config dict so
    the ``all`` task can generate summaries without reloading from disk.
    """
    device = resolve_device(args.device)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"[train] dataset={args.dataset} | train_sample={args.train_sample} | device={device}", flush=True)

    # --- Data: official TRAIN split only ---
    examples = load_train_examples(args.dataset, args.train_sample, args.seed)
    if not examples:
        raise SystemExit("No training examples were loaded; aborting.")

    # --- Vocabulary: built from the training portion only ---
    src_vocab = build_vocab([src for src, _ in examples], args.src_vocab_size)
    trg_vocab = build_vocab([trg for _, trg in examples], args.trg_vocab_size)
    print(f"  vocab sizes: src={len(src_vocab)} | trg={len(trg_vocab)}", flush=True)

    dataset = SummarizationDataset(
        examples, src_vocab, trg_vocab, args.max_source_length, args.max_target_length
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 0 workers keeps training deterministic and Windows-safe.
    )

    # Position-embedding table must cover the longest source/target sequence
    # (the decoder input is <sos> + up to max_target_length content tokens).
    pos_max_length = max(args.max_source_length, args.max_target_length + 2)
    config = {
        "variant": VARIANT,
        "dataset": args.dataset,
        "train_sample": args.train_sample,
        "src_vocab_size": len(src_vocab),
        "trg_vocab_size": len(trg_vocab),
        "embed_size": args.embed_size,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "forward_expansion": args.forward_expansion,
        "dropout": args.dropout,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "pos_max_length": pos_max_length,
        "seed": args.seed,
    }

    model = build_model(config, device)
    print(f"  model parameters: {count_parameters(model):,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)  # padding tokens ignored in the loss
    accum_steps = max(1, args.gradient_accumulation_steps)
    effective_batch_size = args.batch_size * accum_steps

    best_loss = float("inf")
    epoch_losses: List[float] = []
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        epoch_start = time.time()
        optimizer.zero_grad()

        for step, (src, trg) in enumerate(loader):
            src = src.to(device)
            trg = trg.to(device)

            # Teacher forcing: decoder input is <sos>..., labels are ...<eos>.
            trg_input = trg[:, :-1]
            trg_labels = trg[:, 1:]

            logits = model(src, trg_input)  # (N, L, trg_vocab_size)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                trg_labels.reshape(-1),
            )
            (loss / accum_steps).backward()

            is_last_batch = (step + 1) == len(loader)
            if (step + 1) % accum_steps == 0 or is_last_batch:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            num_batches += 1

        mean_loss = running_loss / max(1, num_batches)
        epoch_losses.append(mean_loss)
        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - train_start

        print(
            f"  epoch {epoch}/{args.epochs} | mean_train_loss={mean_loss:.4f} | "
            f"train_examples={len(dataset)} | test_examples={args.sample} | "
            f"device={device} | effective_batch_size={effective_batch_size} | "
            f"epoch_time={epoch_elapsed:.1f}s | elapsed={total_elapsed:.1f}s",
            flush=True,
        )

        # Best checkpoint = lowest TRAINING loss (no validation/test split is used).
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "src_vocab": src_vocab,
                    "trg_vocab": trg_vocab,
                    "best_train_loss": best_loss,
                    "best_epoch": epoch,
                },
                checkpoint_path(args.dataset, args.train_sample),
            )

    total_train_time = time.time() - train_start

    # Persist vocabularies and metadata alongside the checkpoint for reproducible
    # inference (the summarize task loads these files back).
    save_vocab(src_vocab, src_vocab_path(args.dataset, args.train_sample))
    save_vocab(trg_vocab, trg_vocab_path(args.dataset, args.train_sample))

    metadata = {
        "variant": VARIANT,
        "dataset": args.dataset,
        "train_sample": args.train_sample,
        "test_sample": args.sample,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": accum_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "num_train_examples": len(dataset),
        "src_vocab_size": len(src_vocab),
        "trg_vocab_size": len(trg_vocab),
        "epoch_train_losses": epoch_losses,
        "best_train_loss": best_loss,
        "total_train_time_seconds": total_train_time,
        "hyperparameters": config,
        "checkpoint_path": checkpoint_path(args.dataset, args.train_sample),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(metadata_path(args.dataset, args.train_sample), "w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"  training done in {total_train_time:.1f}s | best_train_loss={best_loss:.4f} | "
        f"checkpoint -> {checkpoint_path(args.dataset, args.train_sample)}",
        flush=True,
    )
    return model, src_vocab, trg_vocab, config


# =========================================================================== #
# Greedy decoding and summary generation
# =========================================================================== #
@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src_ids: List[int],
    max_target_length: int,
    device: torch.device,
) -> List[int]:
    """Autoregressively decode one summary with greedy (argmax) selection."""
    model.eval()
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)  # (1, src_len)
    generated = [SOS_ID]

    for _ in range(max_target_length):
        trg_tensor = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model(src_tensor, trg_tensor)  # (1, cur_len, trg_vocab_size)
        next_id = int(logits[0, -1].argmax().item())
        if next_id == EOS_ID:
            break
        generated.append(next_id)

    return generated[1:]  # drop the leading <sos>


def ids_to_summary(ids: Sequence[int], inv_trg_vocab: Dict[int, str]) -> str:
    """Convert generated ids to a clean summary string (specials removed)."""
    special_ids = {PAD_ID, SOS_ID, EOS_ID, UNK_ID}
    tokens = [inv_trg_vocab[i] for i in ids if i not in special_ids and i in inv_trg_vocab]
    return detokenize(tokens)


def run_summarize(
    args: argparse.Namespace,
    model: Optional[Transformer] = None,
    src_vocab: Optional[Dict[str, int]] = None,
    trg_vocab: Optional[Dict[str, int]] = None,
    config: Optional[Dict] = None,
) -> str:
    """Generate summaries on the official TEST split and write benchmark JSONL.

    When ``model``/vocabs/``config`` are provided (the ``all`` task), they are
    reused directly; otherwise they are loaded from the checkpoint and artifact
    files produced by a previous ``train`` run.
    """
    device = resolve_device(args.device)

    if model is None or src_vocab is None or trg_vocab is None or config is None:
        model, src_vocab, trg_vocab, config = load_checkpoint_bundle(
            args.dataset, args.train_sample, device
        )

    inv_trg_vocab = {idx: token for token, idx in trg_vocab.items()}
    max_source_length = config["max_source_length"]
    max_target_length = config["max_target_length"]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir, baseline_filename(f"Transformer_{VARIANT}", args.dataset, args.sample)
    )
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(
            f"Output file already exists: {out_path}\n"
            "Pass --overwrite to regenerate it."
        )

    print("=" * 80, flush=True)
    print(f"[summarize] dataset={args.dataset} | sample={args.sample} | device={device}", flush=True)
    print(f"  writing -> {out_path}", flush=True)

    # Reuse the benchmark's own TEST-split loader so the Transformer summarizes the
    # exact same sampled test articles as every other model/baseline.
    datasets = load_datasets_streaming(sample=args.sample, seed=args.seed, names=[args.dataset])
    data = datasets[args.dataset]

    gen_start = time.time()
    written = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for idx, item in enumerate(data, start=1):
            news_text, ref_summary, qa_pairs = extract_fields(args.dataset, item)
            src_ids = encode(news_text, src_vocab, max_source_length)
            gen_ids = greedy_decode(model, src_ids, max_target_length, device)
            generated_summary = ids_to_summary(gen_ids, inv_trg_vocab)

            record = {
                "news": news_text,
                "reference_summary": ref_summary,
                "generated_summary": generated_summary,
            }
            if qa_pairs is not None:  # preserve QA fields exactly like the benchmark
                record["qa_pairs"] = qa_pairs
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if idx == 1 or idx % 10 == 0:
                print(f"  sample {idx}/{args.sample}", flush=True)

    elapsed = time.time() - gen_start
    print(f"  generated {written} summaries in {elapsed:.1f}s -> {out_path}", flush=True)
    return out_path


def load_checkpoint_bundle(
    dataset_name: str, train_sample: int, device: torch.device
) -> Tuple[Transformer, Dict[str, int], Dict[str, int], Dict]:
    """Load model, vocabularies and config for inference, or fail clearly."""
    ckpt_path = checkpoint_path(dataset_name, train_sample)
    src_path = src_vocab_path(dataset_name, train_sample)
    trg_path = trg_vocab_path(dataset_name, train_sample)

    missing = [p for p in (ckpt_path, src_path, trg_path) if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Cannot summarize: required checkpoint/vocabulary files are missing:\n  "
            + "\n  ".join(missing)
            + f"\nTrain first, e.g.: python Transformer/{VARIANT}.py --task train "
            f"--dataset {dataset_name} --sample {train_sample}"
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]
    src_vocab = load_vocab(src_path)
    trg_vocab = load_vocab(trg_path)

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  loaded checkpoint -> {ckpt_path}", flush=True)
    return model, src_vocab, trg_vocab, config


# =========================================================================== #
# Evaluation (delegates to the original benchmark evaluator)
# =========================================================================== #
def run_evaluate(args: argparse.Namespace) -> None:
    """Invoke the original benchmark evaluator on the summaries directory.

    This shells out to ``src/main.py --task evaluate`` so metric logic is never
    duplicated; the Transformer output is scored exactly like every other run.
    """
    cmd = [
        sys.executable,
        os.path.join(SRC_DIR, "main.py"),
        "--task",
        "evaluate",
        "--output_dir",
        args.output_dir,
        "--log_path",
        args.log_path,
    ]
    print("=" * 80, flush=True)
    print(f"[evaluate] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"The benchmark evaluator exited with code {result.returncode}. "
            "Ensure the evaluation dependencies (evaluate, bert-score, nltk, ...) "
            "are installed in this environment."
        )


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer abstractive summarization baseline for the news "
        "summarization benchmark (train from scratch, generate, evaluate).",
    )

    parser.add_argument(
        "--task",
        choices=["train", "summarize", "all", "evaluate"],
        default="all",
        help="train: train and save a checkpoint. summarize: load a checkpoint and "
        "generate test summaries. all: train then summarize. evaluate: run the "
        "original benchmark evaluator over the summaries directory.",
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="cnn_dailymail",
        help="Which benchmark dataset to use (a separate model per dataset).",
    )

    # Sample sizes (matching the benchmark's sample-based experiment design).
    parser.add_argument(
        "--sample",
        type=int,
        choices=SAMPLE_CHOICES,
        default=20,
        help="Number of official TEST examples to summarize.",
    )
    parser.add_argument(
        "--train_sample",
        type=int,
        choices=SAMPLE_CHOICES,
        default=None,
        help="Number of official TRAIN examples to train on. "
        "Defaults to --sample when omitted.",
    )

    # Training hyperparameters (lightweight defaults for limited resources).
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--max_source_length", type=int, default=256, help="Max article tokens.")
    parser.add_argument("--max_target_length", type=int, default=64, help="Max summary tokens.")
    parser.add_argument("--embed_size", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=2, help="Encoder/decoder layers.")
    parser.add_argument("--forward_expansion", type=int, default=2, help="Feed-forward expansion factor.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before an optimizer step.",
    )

    # Vocabulary limits.
    parser.add_argument("--src_vocab_size", type=int, default=30000, help="Max source vocabulary size.")
    parser.add_argument("--trg_vocab_size", type=int, default=30000, help="Max target vocabulary size.")

    # Reproducibility / device.
    parser.add_argument("--seed", type=int, default=42, help="Random seed (data subset + init).")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. 'cuda' or 'cpu'. Defaults to CUDA if available.",
    )

    # Output / evaluation.
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_SUMMARY_DIR,
        help="Directory for the summary JSONL output "
        "(default: Transformer/summaries, kept separate from the benchmark).",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=DEFAULT_LOG_PATH,
        help="Evaluation log path passed through to the benchmark evaluator "
        "(default: Transformer/results/evaluation.log).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing summary output file instead of failing.",
    )
    parser.add_argument(
        "--evaluate_after_generation",
        action="store_true",
        help="After writing summaries, invoke the original benchmark evaluator.",
    )

    args = parser.parse_args()
    if args.train_sample is None:
        args.train_sample = args.sample
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Running on CPU (no CUDA device selected/available).", flush=True)

    if args.task == "train":
        train_model(args)
    elif args.task == "summarize":
        run_summarize(args)
        if args.evaluate_after_generation:
            run_evaluate(args)
    elif args.task == "all":
        model, src_vocab, trg_vocab, config = train_model(args)
        run_summarize(args, model=model, src_vocab=src_vocab, trg_vocab=trg_vocab, config=config)
        if args.evaluate_after_generation:
            run_evaluate(args)
    elif args.task == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()

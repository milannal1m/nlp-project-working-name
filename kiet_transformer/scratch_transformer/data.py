import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets import load_dataset
from transformers import AutoTokenizer

from dataset import DATASET_CONFIGS, extract_fields

TOKENIZER_NAME = "gpt2"
SPECIAL_TOKENS = {"pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>"}


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tok.add_special_tokens(SPECIAL_TOKENS)
    return tok


def encode_seq(tok, text: str, max_len: int) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False)[: max_len - 2]
    return [tok.bos_token_id] + ids + [tok.eos_token_id]


def _pad_to(seqs: list[list[int]], pad_id: int) -> list[list[int]]:
    width = max(len(s) for s in seqs)
    return [s + [pad_id] * (width - len(s)) for s in seqs]


def make_collate_fn(tok, max_src_len: int, max_trg_len: int):
    pad_id = tok.pad_token_id

    def collate(batch):
        src = _pad_to([encode_seq(tok, a, max_src_len) for a, _ in batch], pad_id)
        trg = _pad_to([encode_seq(tok, s, max_trg_len) for _, s in batch], pad_id)
        return torch.tensor(src, dtype=torch.long), torch.tensor(trg, dtype=torch.long)

    return collate


def load_train_split(name: str, split: str = "train", sample: int | None = None,
                     seed: int = 42, buffer_size: int = 10000):
    cfg = DATASET_CONFIGS[name]
    kwargs = {"split": split, "streaming": True}
    if "name" in cfg:
        kwargs["name"] = cfg["name"]
    ds = load_dataset(cfg["path"], **kwargs)
    if split == "train":
        ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    if sample is not None:
        ds = ds.take(sample)
    return ds


def load_pairs(name: str, split: str, limit: int, seed: int = 42) -> list[tuple[str, str]]:
    stream = load_train_split(name, split=split, sample=limit, seed=seed)
    pairs = []
    for item in stream:
        news, ref, _ = extract_fields(name, item)
        if news and ref:
            pairs.append((news, ref))
    return pairs

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scratch_transformer.data import build_tokenizer
from scratch_transformer.model import Transformer


class ScratchSummarizer:

    def __init__(self, checkpoint_path: str, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        self.tok = build_tokenizer()
        if len(self.tok) != ckpt["vocab_size"]:
            raise ValueError(
                f"Tokenizer vocab ({len(self.tok)}) != checkpoint vocab "
                f"({ckpt['vocab_size']}) — special-token setup changed."
            )

        self.max_src_len = ckpt["max_src_len"]
        self.max_trg_len = ckpt["max_trg_len"]
        self.bos = ckpt["bos_idx"]
        self.eos = ckpt["eos_idx"]

        self.model = Transformer(
            src_vocab_size=ckpt["vocab_size"],
            trg_vocab_size=ckpt["vocab_size"],
            src_pad_idx=ckpt["pad_idx"],
            trg_pad_idx=ckpt["pad_idx"],
            device=self.device,
            **ckpt["arch"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def summarize(self, text: str) -> tuple[str, int, int]:
        ids = self.tok.encode(text, add_special_tokens=False)[: self.max_src_len - 2]
        src_ids = [self.bos] + ids + [self.eos]
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        input_len = src.shape[1]

        out = [self.bos]
        for _ in range(self.max_trg_len - 1):
            trg = torch.tensor([out], dtype=torch.long, device=self.device)
            logits = self.model(src, trg)
            next_id = logits[:, -1, :].argmax(dim=-1).item()
            out.append(next_id)
            if next_id == self.eos:
                break

        gen_ids = out[1:]
        if gen_ids and gen_ids[-1] == self.eos:
            gen_ids = gen_ids[:-1]
        summary = self.tok.decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        return summary, input_len, len(gen_ids)

"""AdityaTransformer: Abstractive Summarization with Weight Tying"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, sys, time, traceback
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# =========================================================================== #
# PATH FIXES 
# =========================================================================== #
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(TRANSFORMER_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")

VARIANT = "Aditya"
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, "checkpoints")
ARTIFACT_DIR = os.path.join(CURRENT_DIR, "artifacts")
DEFAULT_SUMMARY_DIR = os.path.join(CURRENT_DIR, "summaries", VARIANT)
DEFAULT_LOG_PATH = os.path.join(CURRENT_DIR, "results", VARIANT, "evaluation.log")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

SUPPORTED_DATASETS = ["cnn_dailymail", "xsum"]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

# =========================================================================== #
# MODEL ARCHITECTURE (With 2-Way Weight Tying)
# =========================================================================== #
class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int) -> None:
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        v_l, k_l, q_l = values.shape[1], keys.shape[1], query.shape[1]
        values = self.values(values.reshape(N, v_l, self.heads, self.head_dim))
        keys = self.keys(keys.reshape(N, k_l, self.heads, self.head_dim))
        queries = self.queries(query.reshape(N, q_l, self.heads, self.head_dim))
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None: energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, q_l, self.heads * self.head_dim)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: int) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion * embed_size), nn.ReLU(), nn.Linear(forward_expansion * embed_size, embed_size))
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        return self.dropout(self.norm2(self.feed_forward(x) + x))

class Encoder(nn.Module):
    def __init__(self, src_vocab_size: int, embed_size: int, num_layers: int, heads: int, device, forward_expansion: int, dropout: float, max_length: int) -> None:
        super(Encoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length, device=self.device).expand(N, seq_length)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers: out = layer(out, out, out, mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: int, device) -> None:
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        return self.transformer_block(value, key, query, src_mask)

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size: int, embed_size: int, num_layers: int, heads: int, forward_expansion: int, dropout: float, device, max_length: int) -> None:
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([DecoderBlock(embed_size, heads, dropout, forward_expansion, device) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length, device=self.device).expand(N, seq_length)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers: x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        return self.fc_out(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, trg_vocab_size: int, src_pad_idx: int, trg_pad_idx: int, embed_size: int = 256, num_layers: int = 6, forward_expansion: int = 4, heads: int = 8, dropout: float = 0.0, device="cpu", max_length: int = 100) -> None:
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        # --- ADITYA'S MODIFICATION: 2-WAY WEIGHT TYING ---
        self.decoder.fc_out.weight = self.decoder.word_embedding.weight

    def make_src_mask(self, src): return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        return (trg_pad_mask & causal_mask).to(self.device)
    def forward(self, src, trg):
        src_mask, trg_mask = self.make_src_mask(src), self.make_trg_mask(trg)
        return self.decoder(trg, self.encoder(src, src_mask), src_mask, trg_mask)

# =========================================================================== #
# UTILS & VOCABULARY
# =========================================================================== #
def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg: return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
def tokenize(text: str) -> List[str]: return _TOKEN_RE.findall(text.lower()) if text else []

def build_vocab(texts: Sequence[str], max_size: int) -> Dict[str, int]:
    counter = Counter()
    for text in texts: counter.update(tokenize(text))
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(vocab) >= max_size: break
        if token not in vocab: vocab[token] = len(vocab)
    return vocab

def encode(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(token, UNK_ID) for token in tokenize(text)][:max_len]
    return ids if ids else [UNK_ID]

def build_target_sequence(text: str, vocab: Dict[str, int], max_content_len: int) -> List[int]:
    return [SOS_ID] + [vocab.get(token, UNK_ID) for token in tokenize(text)][:max_content_len] + [EOS_ID]

def detokenize(tokens: Sequence[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,;:!?%])", r"\1", text)
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", text)).strip()

def save_vocab(vocab: Dict[str, int], path: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f: json.dump(vocab, f, ensure_ascii=False, indent=2)

def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

# =========================================================================== #
# DATASET LOADING (FIXED FOR STANDARD JSON ARRAY)
# =========================================================================== #
def _load_local_split(dataset_name, split, sample, seed):
    """CUSTOM LOADER TO USE YOUR EXACT JSON FILES."""
    if dataset_name == "cnn_dailymail":
        filename = "cnndm_sample_500_0k5_1k5_qwen_summary.jsonl"
    else:
        filename = "xsum_sample_500_0k5_1k5_qwen_summary.jsonl"
        
    file_path = os.path.join(REPO_ROOT, "xu_et_all_datasets", filename)
    print(f"----> INTERCEPTED: Loading dataset from {file_path}", flush=True)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CRITICAL: Could not find dataset at {file_path}")
    
    class DummyDataset(list):
        def shuffle(self, seed):
            rng = random.Random(seed); rng.shuffle(self); return self
        def select(self, indices):
            return DummyDataset([self[i] for i in indices])
            
    # THE FIX: Read the whole file at once because it's a JSON array!
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    ds = DummyDataset(data)
    if sample is not None: ds = ds.shuffle(seed).select(range(min(sample, len(ds))))
    return ds

def load_train_examples(dataset_name: str, sample: int, seed: int) -> List[Tuple[str, str]]:
    ds = _load_local_split(dataset_name, "train", sample, seed)
    examples = []
    for item in ds:
        # THE FIX: Using exact keys!
        news_text = item.get("article", "")
        ref_summary = item.get("qwen_reference_summary", item.get("original_reference_summary", ""))
        examples.append((news_text, ref_summary))
    print(f"  [ready] {dataset_name} ({len(examples)} examples)", flush=True)
    return examples

class SummarizationDataset(Dataset):
    def __init__(self, examples: Sequence[Tuple[str, str]], src_vocab: Dict[str, int], trg_vocab: Dict[str, int], max_source_length: int, max_target_length: int) -> None:
        self.samples = [(encode(n, src_vocab, max_source_length), build_target_sequence(r, trg_vocab, max_target_length)) for n, r in examples]
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.samples[idx][0], dtype=torch.long), torch.tensor(self.samples[idx][1], dtype=torch.long)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    def pad(seqs):
        m = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), m), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs): out[i, :s.size(0)] = s
        return out
    return pad([i[0] for i in batch]), pad([i[1] for i in batch])

# =========================================================================== #
# TRAINING & EVALUATION PIPELINE
# =========================================================================== #
def build_model(config: Dict, device: torch.device) -> Transformer:
    return Transformer(config["src_vocab_size"], config["trg_vocab_size"], PAD_ID, PAD_ID, config["embed_size"], config["num_layers"], config["forward_expansion"], config["heads"], config["dropout"], device, config["pos_max_length"]).to(device)

def train_model(args) -> Tuple[Transformer, Dict[str, int], Dict[str, int], Dict]:
    device = resolve_device(args.device)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"[train] dataset={args.dataset} | device={device}", flush=True)

    examples = load_train_examples(args.dataset, args.train_sample, args.seed)
    src_vocab = build_vocab([src for src, _ in examples], args.src_vocab_size)
    trg_vocab = build_vocab([trg for _, trg in examples], args.trg_vocab_size)
    
    loader = DataLoader(SummarizationDataset(examples, src_vocab, trg_vocab, args.max_source_length, args.max_target_length), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    config = {
        "variant": VARIANT, "dataset": args.dataset, "train_sample": args.train_sample,
        "src_vocab_size": len(src_vocab), "trg_vocab_size": len(trg_vocab), "embed_size": args.embed_size,
        "num_layers": args.num_layers, "heads": args.heads, "forward_expansion": args.forward_expansion,
        "dropout": args.dropout, "max_source_length": args.max_source_length, "max_target_length": args.max_target_length,
        "pos_max_length": max(args.max_source_length, args.max_target_length + 2), "seed": args.seed,
    }

    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    
    best_loss = float("inf")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (src, trg) in enumerate(loader):
            src, trg = src.to(device), trg.to(device)
            logits = model(src, trg[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), trg[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            
        mean_loss = running_loss / len(loader)
        print(f"  epoch {epoch}/{args.epochs} | loss={mean_loss:.4f} | time={time.time() - train_start:.1f}s", flush=True)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{VARIANT}_{args.dataset}_ckpt.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config}, ckpt_path)
    save_vocab(src_vocab, os.path.join(ARTIFACT_DIR, f"{VARIANT}_{args.dataset}_src.json"))
    save_vocab(trg_vocab, os.path.join(ARTIFACT_DIR, f"{VARIANT}_{args.dataset}_trg.json"))
    return model, src_vocab, trg_vocab, config

@torch.no_grad()
def greedy_decode(model: Transformer, src_ids: List[int], max_target_length: int, device: torch.device) -> List[int]:
    model.eval()
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    generated = [SOS_ID]
    for _ in range(max_target_length):
        logits = model(src_tensor, torch.tensor([generated], dtype=torch.long, device=device))
        next_id = int(logits[0, -1].argmax().item())
        if next_id == EOS_ID: break
        generated.append(next_id)
    return generated[1:]

def run_summarize(args, model, src_vocab, trg_vocab, config) -> str:
    device = resolve_device(args.device)
    inv_trg_vocab = {idx: token for token, idx in trg_vocab.items()}
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{VARIANT}_{args.dataset}_summaries.jsonl")
    
    print(f"[summarize] writing -> {out_path}", flush=True)
    data = _load_local_split(args.dataset, "test", args.sample, args.seed)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for idx, item in enumerate(data, start=1):
            news_text = item.get("article", "")
            ref_summary = item.get("qwen_reference_summary", item.get("original_reference_summary", ""))
            src_ids = encode(news_text, src_vocab, config["max_source_length"])
            gen_ids = greedy_decode(model, src_ids, config["max_target_length"], device)
            f.write(json.dumps({"news": news_text, "reference_summary": ref_summary, "generated_summary": detokenize([inv_trg_vocab[i] for i in gen_ids if i not in {PAD_ID, SOS_ID, EOS_ID, UNK_ID} and i in inv_trg_vocab])}, ensure_ascii=False) + "\n")
    return out_path

def run_evaluate(args) -> None:
    cmd = [sys.executable, os.path.join(SRC_DIR, "main.py"), "--task", "evaluate", "--output_dir", args.output_dir, "--log_path", args.log_path]
    print(f"[evaluate] running benchmark evaluator...", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT)

def main() -> None:
    try:
        class Args: pass
        args = Args()
        args.task = "all"
        
        # ---> CHANGE THIS TO THE SPECIFIC DATASET NAME <---
        args.dataset = "xsum"  
        
        args.sample = 500
        args.train_sample = 500
        args.epochs = 5
        args.batch_size = 2
        args.learning_rate = 3e-4
        args.max_source_length = 256
        args.max_target_length = 64
        args.embed_size = 128
        args.heads = 4
        args.num_layers = 2
        args.forward_expansion = 2
        args.dropout = 0.1
        args.gradient_accumulation_steps = 1
        args.src_vocab_size = 30000
        args.trg_vocab_size = 30000
        args.seed = 42
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.output_dir = DEFAULT_SUMMARY_DIR
        args.log_path = DEFAULT_LOG_PATH

        set_seed(args.seed)
        model, src_vocab, trg_vocab, config = train_model(args)
        run_summarize(args, model, src_vocab, trg_vocab, config)
        run_evaluate(args)
        
        print("ALL TASKS COMPLETED SUCCESSFULLY!", flush=True)
    except Exception as e:
        print("\nCRITICAL CRASH DETECTED:\n")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
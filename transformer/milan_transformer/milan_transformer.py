import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
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
        N = query.shape[0] #number of examples in the batch
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
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N,
            query_len,
            self.heads * self.head_dim
        )  # after einsum (N,query_len, head, head_dim) then flatten  the last two dimensions (N, query_len, embed_size)

        out = self.fc_out(out)  # (N, query_len, embed_size)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.pad_idx = pad_idx
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_mask(self, x):
        # Combine the padding mask with a causal (lower-triangular) mask so each
        # position attends only to real, past tokens. (N, 1, seq_len, seq_len)
        N, seq_len = x.shape
        pad_mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, seq_len)
        causal = torch.tril(torch.ones((seq_len, seq_len), device=self.device)).bool()
        return (pad_mask & causal).to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        mask = self.make_mask(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = self.fc_out(out)

        return out

    def sample_top_p(self, probs, top_p):
        # Nucleus sampling: keep the smallest set of tokens whose cumulative
        # probability exceeds top_p (always at least one), renormalise, then sample.
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs > top_p  # keep the token that crosses the threshold
        sorted_probs[remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, choice)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_idx, temperature=1.0, top_p=0.9):
        # idx: (1, prompt_len) token ids of the prompt (e.g. "[CLS] article [SEP]").
        # Autoregressively append sampled tokens until <eos> or max_new_tokens.
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.position_embedding.num_embeddings:]  # never exceed max_length
            logits = self.forward(idx_cond)[:, -1, :]        # (1, vocab): last position only
            probs = torch.softmax(logits / temperature, dim=-1)  # temperature softmax
            next_idx = self.sample_top_p(probs, top_p)       # (1, 1)
            idx = torch.cat([idx, next_idx], dim=1)
            if next_idx.item() == eos_idx:
                break
        return idx


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Toy check: a tiny vocab, one random prompt, decode a few tokens.
    vocab_size, pad_idx = 20, 0
    model = Transformer(vocab_size, pad_idx, embed_size=32, num_layers=2, heads=4,
                        device=device, max_length=50).to(device)
    model.eval()

    prompt = torch.tensor([[1, 5, 6, 7, 2]]).to(device)  # e.g. <sos> a b c <sep>
    print("Logits shape:", model(prompt).shape)          # (1, 5, vocab_size)

    out = model.generate(prompt, max_new_tokens=10, eos_idx=2, temperature=1.0, top_p=0.9)
    print("Generated token ids:", out[0].tolist())

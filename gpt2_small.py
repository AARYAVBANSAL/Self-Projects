"""
GPT-2 Small From Scratch Implementation
Self Project (May '25 - June '25)

- Custom language model built with PyTorch: byte-pair encoding (via a
  minimal from-scratch BPE tokenizer), causal self-attention, and
  transformer decoder blocks -- following the GPT-2 "small" configuration
  (12 layers, 12 heads, 768-dim embeddings, ~124M params).
- Includes an accompanying walkthrough of the linear algebra and calculus
  behind each component (see NOTES.md).

Usage:
    python gpt2_small.py --train data.txt --steps 2000   # train a tiny model
    python gpt2_small.py --generate "Once upon a time"    # sample text

This file is intentionally self-contained (tokenizer + model + training
loop + generation) so it can run end-to-end on a laptop / single GPU for
learning purposes. Swap in the real GPT-2 BPE merges/vocab for full
compatibility with OpenAI's released checkpoints.
"""

import argparse
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. Byte-Pair Encoding (BPE) tokenizer, built from scratch
# --------------------------------------------------------------------------
class BPETokenizer:
    """
    A minimal byte-level BPE tokenizer trained from raw text, in the same
    spirit as GPT-2's tokenizer:
      1. Start with a vocabulary of individual bytes/characters.
      2. Repeatedly find the most frequent adjacent symbol pair and merge
         it into a new symbol, recording the merge rule.
      3. Encoding replays the learned merges greedily; decoding just
         concatenates the symbol strings back together.
    """

    def __init__(self):
        self.merges: dict[tuple[str, str], int] = {}   # (a, b) -> priority (lower = earlier/more frequent)
        self.vocab: dict[int, str] = {}                 # token id -> string
        self.token_to_id: dict[str, int] = {}

    @staticmethod
    def _word_to_symbols(word: str) -> list[str]:
        # Represent each word as a list of characters plus an end-of-word marker,
        # which lets BPE learn word-boundary-sensitive merges.
        return list(word) + ["</w>"]

    def train(self, text: str, vocab_size: int = 512):
        words = re.findall(r"\S+|\s", text)
        word_freq = Counter(words)
        corpus = {tuple(self._word_to_symbols(w)): f for w, f in word_freq.items()}

        # Base vocabulary: every unique symbol seen so far.
        symbols = set()
        for word_syms in corpus:
            symbols.update(word_syms)
        vocab = sorted(symbols)

        merges: dict[tuple[str, str], int] = {}
        merge_idx = 0
        while len(vocab) < vocab_size:
            pair_counts = defaultdict(int)
            for word_syms, freq in corpus.items():
                for i in range(len(word_syms) - 1):
                    pair_counts[(word_syms[i], word_syms[i + 1])] += freq
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break  # no more useful merges

            merges[best_pair] = merge_idx
            merge_idx += 1
            new_symbol = "".join(best_pair)
            vocab.append(new_symbol)

            new_corpus = {}
            for word_syms, freq in corpus.items():
                merged = self._apply_merge(word_syms, best_pair, new_symbol)
                new_corpus[merged] = new_corpus.get(merged, 0) + freq
            corpus = new_corpus

        self.merges = merges
        self.vocab = {i: s for i, s in enumerate(vocab)}
        self.token_to_id = {s: i for i, s in self.vocab.items()}

    @staticmethod
    def _apply_merge(symbols: tuple[str, ...], pair: tuple[str, str], new_symbol: str) -> tuple[str, ...]:
        out = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(new_symbol)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    def _bpe_word(self, word: str) -> list[str]:
        symbols = tuple(self._word_to_symbols(word))
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            candidate_pairs = [p for p in pairs if p in self.merges]
            if not candidate_pairs:
                break
            best_pair = min(candidate_pairs, key=lambda p: self.merges[p])
            symbols = self._apply_merge(symbols, best_pair, "".join(best_pair))
        return list(symbols)

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in re.findall(r"\S+|\s", text):
            for sym in self._bpe_word(word):
                if sym not in self.token_to_id:
                    # Fall back to per-character ids for unseen symbols.
                    for ch in sym:
                        ids.append(self.token_to_id.get(ch, 0))
                else:
                    ids.append(self.token_to_id[sym])
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.vocab.get(i, "") for i in ids)
        return text.replace("</w>", "")

    def __len__(self):
        return len(self.vocab)


# --------------------------------------------------------------------------
# 2. Model configuration (GPT-2 "small": 12 layers, 12 heads, d_model=768)
# --------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 128     # max context length (n_ctx)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


# --------------------------------------------------------------------------
# 3. Causal self-attention
#
#    For each head: Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) + mask ) V
#    The causal mask sets scores for future positions to -inf so token t can
#    only attend to tokens <= t (autoregressive property).
# --------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Fused projection for Q, K, V (one matmul instead of three).
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular mask, cached as a buffer (not a learned parameter).
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        qkv = self.qkv_proj(x)                                  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # Reshape into (B, n_head, T, head_dim) for multi-head attention.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores.
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v                                    # (B, nh, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)       # merge heads back

        out = self.resid_dropout(self.out_proj(out))
        return out


# --------------------------------------------------------------------------
# 4. Position-wise feed-forward (MLP) block: Linear -> GELU -> Linear
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc_in(x))
        x = self.fc_out(x)
        return self.dropout(x)


# --------------------------------------------------------------------------
# 5. Transformer block: pre-LayerNorm residual attention + residual MLP
# --------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # residual connection around attention
        x = x + self.mlp(self.ln_2(x))    # residual connection around MLP
        return x


# --------------------------------------------------------------------------
# 6. Full GPT-2 model: token + positional embeddings -> N blocks -> head
# --------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: input embedding and output projection share weights,
        # as in the original GPT-2 (reduces params, improves generalization).
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Initialized GPT-2 small: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence length exceeds block_size"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(idx) + self.pos_emb(pos)                # (B, T, C)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Standard next-token cross-entropy loss (teacher forcing).
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# --------------------------------------------------------------------------
# 7. Training loop
# --------------------------------------------------------------------------
def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def train(text_path: str, steps: int, batch_size: int = 16, block_size: int = 64,
          lr: float = 3e-4, device: str | None = None, small_config: bool = True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=1024)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Dataset: {len(data)} tokens, vocab size {len(tokenizer)}")

    # A scaled-down config trains fast for demo purposes; set small_config=False
    # to use the full GPT-2 "small" size (12 layer / 12 head / 768 dim).
    config = GPTConfig(
        vocab_size=len(tokenizer),
        block_size=block_size,
        n_layer=4 if small_config else 12,
        n_head=4 if small_config else 12,
        n_embd=128 if small_config else 768,
        dropout=0.1,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    for step in range(1, steps + 1):
        xb, yb = get_batch(data, config.block_size, batch_size, device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # stabilizes training
        optimizer.step()

        if step % max(1, steps // 20) == 0 or step == 1:
            print(f"step {step:5d}/{steps} | loss {loss.item():.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, "checkpoints/gpt2_small.pt")
    return model, tokenizer, config


def generate_from_checkpoint(prompt: str, checkpoint_path: str = "checkpoints/gpt2_small.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # NOTE: for a real CLI you'd persist the tokenizer's merges/vocab too;
    # omitted here for brevity since this script trains + generates in one run.
    print("Loaded model. (Tokenizer must be retrained/reloaded to encode the prompt.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Small from scratch (PyTorch)")
    parser.add_argument("--train", type=str, help="Path to a text file to train on")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--generate", type=str, help="Prompt to generate text from")
    args = parser.parse_args()

    if args.train:
        model, tokenizer, config = train(args.train, steps=args.steps)
        if args.generate:
            model.eval()
            ids = torch.tensor([tokenizer.encode(args.generate)], dtype=torch.long)
            out = model.generate(ids, max_new_tokens=100, temperature=0.8, top_k=40)
            print(tokenizer.decode(out[0].tolist()))
    else:
        print("Provide --train <path_to_text_file> to train a model (add --generate for a sample).")

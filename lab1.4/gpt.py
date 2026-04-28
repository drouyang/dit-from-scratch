# Decoder-only transformer (GPT), built by stacking lab 1.3's MultiHeadAttention.
#
# This file defines WHAT the model computes.
# train.py defines HOW it is trained on TinyShakespeare.
# sample.py loads a checkpoint and generates text.
#
# The pieces:
#   MLP(embed_dim)               — position-wise feed-forward sublayer
#   Block(embed_dim, num_heads)  — pre-norm: LN → MHA(causal) → +res, LN → MLP → +res
#   GPT(...)                     — embed + N Blocks + final LN + linear head
#
# Why this matters for DiT:
#   A DiT block is structurally identical to a GPT block — multi-head attention
#   plus an MLP, both wrapped in pre-norm + residual. The only DiT-specific
#   thing is AdaLN-Zero modulation of the LayerNorms from a conditioning
#   vector. Get the GPT block right here; DiT is "this block, but with image
#   patches instead of text tokens, and a modulated LN."

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the MultiHeadAttention you wrote (and verified bit-for-bit against
# torch.nn.MultiheadAttention) in lab 1.3. The directory name `lab1.3` has a
# dot in it, which would break a normal `import lab1.3.attention` — but we're
# adding the directory to sys.path and importing the file by its module name
# (`attention.py` -> `import attention`), which works fine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lab1.3"))
from attention import MultiHeadAttention, causal_mask  # noqa: E402


class MLP(nn.Module):
    """Position-wise feed-forward: Linear → GELU → Linear → Dropout.

    Two linears with a 4× hidden expansion (the standard transformer ratio),
    applied independently at every sequence position. Attention itself is a
    *linear* weighted average — without an MLP between attention layers, the
    whole stack would collapse to one big linear op (linear ∘ linear = linear).
    The MLP's GELU is what makes the network actually deep.
    """

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """One transformer block: pre-norm attention + pre-norm MLP, both residual.

        x ──► LN ──► MHA(causal) ──┐
        │                          ▼
        └─────────► + ◄────────────┘
                    │
                    ├──► LN ──► MLP ──┐
                    │                 ▼
                    └────► + ◄────────┘  ──► out

    Pre-norm (LN before each sublayer, residual added after) is the modern
    standard. Residuals carry an unnormalized identity path through every
    block, so gradients flow back through dozens of layers without vanishing
    — post-norm needs careful warmup to do the same.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer. Predicts the next token from the past.

    Trained with cross-entropy on shifted sequences:
        input  = tokens[0..L-1]
        target = tokens[1..L]
    The causal mask ensures position i never sees i+1 during training, so
    each prediction depends only on the past — exactly the constraint at
    inference time. No train/test mismatch for autoregression.
    """

    def __init__(self, vocab_size, block_size, n_layer=6, n_head=6,
                 n_embd=384, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout=dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: input embedding and output head share parameters.
        # Saves vocab_size * n_embd parameters and gives the head's row for
        # token t the same direction as the embedding of token t — which is
        # what "predicting t" geometrically should mean.
        self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)
        # GPT-2 trick: scale residual-projection inits by 1/sqrt(2*n_layer).
        # Each block adds two residual contributions (attn out_proj, mlp fc2);
        # without this rescaling, activation variance grows linearly with
        # depth and deep stacks become unstable.
        for p_name, p in self.named_parameters():
            if p_name.endswith("out_proj.weight") or p_name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, L = idx.shape
        assert L <= self.block_size, f"sequence length {L} exceeds block_size {self.block_size}"

        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.drop(x)

        # One causal mask, reused across all blocks.
        mask = causal_mask(L, device=idx.device)
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Sample tokens one at a time from the model's predictive distribution.

        idx: (B, L) prompt tokens. Returns idx extended by max_new_tokens.

        Each step runs the full model on the (truncated to block_size) context,
        takes logits at the last position, optionally rescales by temperature,
        truncates to top-k, and samples. O(L²) per step because no K/V cache —
        fine for a teaching impl; production code would cache.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

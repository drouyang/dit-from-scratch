# Decoder-only transformer (GPT), using PyTorch's built-in attention kernel.
#
# This file defines WHAT the model computes.
# train.py defines HOW it is trained on TinyShakespeare.
# sample.py loads a checkpoint and generates text.
#
# The pieces:
#   CausalSelfAttention          — multi-head self-attention with causal mask,
#                                  via F.scaled_dot_product_attention
#   MLP(embed_dim)               — position-wise feed-forward sublayer
#   Block(embed_dim, num_heads)  — pre-norm: LN → attn → +res, LN → MLP → +res
#   GPT(...)                     — embed + N Blocks + final LN + linear head

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention via F.scaled_dot_product_attention.

    Layout follows the standard nanoGPT style: one packed QKV linear
    (`c_attn`, shape `(embed_dim, 3*embed_dim)`) and one output projection
    (`c_proj`, shape `(embed_dim, embed_dim)`). The causal mask is applied
    inside the kernel via `is_causal=True` — no need to materialize an
    `(L, L)` boolean tensor.

    The attention math is exactly what you wrote in lab 1.3:
        attn = softmax(Q Kᵀ / sqrt(d_k))   (with causal mask)
        out  = attn @ V
    PyTorch just runs it as a single fused kernel.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Packed Q/K/V projection — one matmul instead of three.
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection — mixes heads back into the residual stream.
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(self, x):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # 1) One packed QKV projection; split into three (B, L, D) tensors.
        q, k, v = self.c_attn(x).split(D, dim=-1)

        # 2) Reshape for multi-head: (B, L, D) -> (B, H, L, Dh).
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        # 3) Fused attention. is_causal=True applies the upper-triangular
        #    mask internally; dropout_p is applied to attention weights.
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        # 4) Merge heads back: (B, H, L, Dh) -> (B, L, D).
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # 5) Output projection + residual dropout.
        return self.resid_dropout(self.c_proj(out))


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
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout=dropout)

    def forward(self, x):
        # Causal mask lives inside CausalSelfAttention (is_causal=True),
        # so the block doesn't need to know about it.
        x = x + self.attn(self.ln1(x))
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
        # Each block adds two residual contributions (attn c_proj, mlp fc2);
        # without this rescaling, activation variance grows linearly with
        # depth and deep stacks become unstable.
        for p_name, p in self.named_parameters():
            if p_name.endswith("c_proj.weight") or p_name.endswith("fc2.weight"):
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

        # Causal masking is handled inside each block's attention sublayer
        # (F.scaled_dot_product_attention with is_causal=True).
        for block in self.blocks:
            x = block(x)
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

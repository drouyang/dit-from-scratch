# Multi-head attention, from scratch.
#
# This file defines WHAT attention computes.
# train.py defines HOW it is trained on the reverse task.
# verify.py checks numerical parity with torch.nn.MultiheadAttention.
#
# Two pieces:
#   scaled_dot_product_attention(q, k, v, mask) — the core operation
#   MultiHeadAttention(embed_dim, num_heads)    — splits into H heads, runs attention in parallel, merges
#
# Why this matters for DiT:
#   Every DiT block is (AdaLN-modulated) self-attention + MLP. The attention
#   here is the same operation used inside DiT — only the conditioning and the
#   residual structure around it change. Get the kernel right once; re-use forever.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, attn_mask=None):
    """The core attention kernel.

    q, k, v: (..., L_q, D_head), (..., L_k, D_head), (..., L_k, D_head)
             The leading dims (...) can be (B,) or (B, H) — broadcasting handles it.
    attn_mask: optional (..., L_q, L_k)
             - bool: True = this position is NOT allowed to attend (matches PyTorch)
             - float: added to scores before softmax (use -inf to block, 0 to keep)

    Returns:
        out:  (..., L_q, D_head) — attended values
        attn: (..., L_q, L_k)    — attention weights (each row sums to 1)

    The mechanical story in one line:
        attn = softmax( Q Kᵀ / sqrt(d_k) )          # "who attends to whom"
        out  = attn @ V                              # weighted mix of values

    Why the sqrt(d_k) scaling? As d_k grows, the variance of q·k grows too,
    pushing softmax into a near-one-hot regime where gradients vanish. Dividing
    by sqrt(d_k) keeps the logits roughly unit-variance at init.
    """
    d_k = q.size(-1)

    # Q Kᵀ: for each query position, a similarity score against every key.
    # Shapes: (..., L_q, d_k) @ (..., d_k, L_k) -> (..., L_q, L_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Optional mask: either bool ("True = block") or float ("additive").
    # For causal LM you'd pass an upper-triangular bool mask so position i
    # cannot see positions > i. For the reverse task we use no mask at all.
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    # Softmax over the KEY dimension: each query's attention weights sum to 1.
    attn = F.softmax(scores, dim=-1)

    # Weighted sum of values. Each output position is a convex combination of
    # all value vectors, with weights decided by the query-key similarities.
    out = torch.matmul(attn, v)
    return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention.

    Splits the model dimension into `num_heads` independent heads of width
    `head_dim = embed_dim // num_heads`, runs scaled dot-product attention in
    each head in parallel, then concatenates and projects back to `embed_dim`.

    Parameter layout deliberately matches torch.nn.MultiheadAttention so we can
    copy weights 1-to-1 in verify.py and prove our kernel is correct:
        in_proj_weight : (3*embed_dim, embed_dim)   Q, K, V packed into one matrix
        in_proj_bias   : (3*embed_dim,)
        out_proj       : nn.Linear(embed_dim, embed_dim)

    Why pack Q, K, V? One big matmul is faster than three small ones, and the
    concat of three independent Linear(embed_dim, embed_dim) layers is
    mathematically identical.

    Why multi-head? A single head has to use its d_k dimensions to represent
    ALL the relationships between tokens. With H heads, each head can
    specialize (one on syntax, one on copy patterns, one on position, ...).
    The total parameter count is unchanged — we just reshape the same weights.
    """

    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Single packed Q/K/V projection: (B, L, D) -> (B, L, 3D).
        # We expose it as a raw Parameter (not nn.Linear) to match PyTorch's
        # naming exactly — makes weight transfer in verify.py trivial.
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None

        # Output projection: mixes information across heads after concat.
        # Without this, each head's output would occupy its own fixed slice of
        # the embedding — out_proj lets heads contribute jointly to every feature.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init on the packed Q/K/V matches torch.nn.MultiheadAttention's default.
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, attn_mask=None, return_attn=False):
        """
        x: (B, L, D)
        attn_mask: (L, L) or (B, L, L) — see scaled_dot_product_attention docstring.
        return_attn: if True, also return attention weights averaged across heads.

        Returns:
            out:  (B, L, D)
            attn: (B, H, L, L) or None
        """
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # 1) Project to Q, K, V in one matmul.
        # F.linear(x, W, b) computes x @ Wᵀ + b. Packed projection gives (B, L, 3D).
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # three tensors of (B, L, D)

        # 2) Reshape for multi-head: split D into (H, Dh) and move H forward
        #    so each head becomes an independent batch dim for the attention op.
        # (B, L, D) -> (B, L, H, Dh) -> (B, H, L, Dh)
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        # 3) Broadcast the mask over heads if the user gave us one per batch.
        if attn_mask is not None and attn_mask.dim() == 3:
            # (B, L, L) -> (B, 1, L, L), broadcasts over the H dim.
            attn_mask = attn_mask.unsqueeze(1)

        # 4) Run attention on each head in parallel — the H dim is just a batch
        #    dim as far as scaled_dot_product_attention is concerned.
        out, attn = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # out: (B, H, L, Dh)   attn: (B, H, L, L)

        # 5) Merge heads: (B, H, L, Dh) -> (B, L, H, Dh) -> (B, L, D).
        # .contiguous() is needed because .view() requires contiguous memory,
        # and .transpose() only swaps strides.
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # 6) Final linear mix across heads.
        out = self.out_proj(out)

        return (out, attn) if return_attn else (out, None)


def causal_mask(seq_len, device=None):
    """Upper-triangular True-above-diagonal mask: position i cannot see j > i.

    Shape: (seq_len, seq_len). True = BLOCKED (matches our convention).
    Use this for autoregressive language modeling (lab1.4 / DiT-style class-conditional
    generation with masked attention). The reverse task in train.py does NOT use it —
    each output position needs to see the full input to find its mirror partner.
    """
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


class AttentionModel(nn.Module):
    """Tiny model that exists only to train attention on the reverse task.

    Architecture:
        token_embed + pos_embed -> MultiHeadAttention -> Linear head -> logits

    No MLP, no residual, no LayerNorm — we want to see attention do the work
    by itself. That's what makes the attention heatmap so clean: the
    anti-diagonal pattern is the ONLY way this model can solve reverse.
    """

    def __init__(self, vocab_size, seq_len, embed_dim=64, num_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        # Learned (not sinusoidal) positional embeddings — simpler, and for
        # fixed-length inputs there's no benefit to the sinusoidal form.
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, return_attn=False):
        # x: (B, L) integer tokens   ->   logits: (B, L, vocab)
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.token_embed(x) + self.pos_embed(pos)          # (B, L, D)
        h, attn = self.attn(h, return_attn=return_attn)         # (B, L, D)
        logits = self.head(h)                                   # (B, L, vocab)
        return (logits, attn) if return_attn else logits

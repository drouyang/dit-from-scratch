"""DiT for text-to-image, extending lab 3.1's class-conditional DiT.

Architecture is identical to lab 3.1 except for two changes that turn class
conditioning into text conditioning:

1. **Pooled text → AdaLN.** Lab 3.1 had `c = t_embed(t) + class_embed(y)`.
   Here `c = t_embed(t) + text_proj(pooled_text)` — the same conditioning
   vector that drives every AdaLN-Zero modulation, just sourced from CLIP's
   pooled output instead of an embedding lookup.

2. **Per-token text → cross-attention.** Each DiT block adds a cross-attention
   sublayer between self-attention and MLP. The image tokens (queries) attend
   to the text tokens (keys, values), so the model can route information from
   *individual words* in the prompt — not just a prompt-level summary.

The block becomes:

    x = x + gate * self_attn(AdaLN(x, c))     ← image self-attention (lab 3.1)
    x = x + cross_attn(LN(x), text_tokens)    ← NEW: image attends to text
    x = x + gate * mlp(AdaLN(x, c))           ← MLP (lab 3.1)

This is the standard SD3 / FLUX / PixArt recipe at small scale.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Helpers (same as lab 3.1) ──────────────────────────────────────────────


def modulate(x, shift, scale):
    """Apply AdaLN's per-c shift+scale: x' = (1 + scale) * x + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sinusoidal_time_embed(t, dim):
    """Sinusoidal embedding for continuous t in [0, 1]. (B,) → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal-embed t, then run through a small MLP to (B, hidden)."""

    def __init__(self, hidden, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t):
        return self.mlp(sinusoidal_time_embed(t, self.time_dim))


# ─── Text conditioning (NEW relative to lab 3.1) ────────────────────────────


class TextProjector(nn.Module):
    """Project CLIP's pooled text vector to the DiT's conditioning dim.

    Replaces lab 3.1's `LabelEmbedder`. Same role: produce a single per-sample
    vector that gets added into `c` and drives AdaLN-Zero in every block.
    """

    def __init__(self, text_dim, hidden, p_uncond=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Learned "null" embedding for CFG label-dropout. Same idea as lab 3.1's
        # null class slot in nn.Embedding(num_classes + 1, ...).
        self.null_embed = nn.Parameter(torch.zeros(1, hidden))
        self.p_uncond = p_uncond

    def forward(self, pooled, force_null=None):
        """Args:
            pooled: (B, text_dim) pooled CLIP output
            force_null: optional bool tensor (B,) — True forces null, False forces
                conditional. If None, drops with p_uncond at training time only.
        """
        emb = self.proj(pooled)
        if force_null is None:
            if self.training and self.p_uncond > 0:
                drop = (torch.rand(emb.size(0), device=emb.device) < self.p_uncond)
                emb = torch.where(drop[:, None], self.null_embed, emb)
        else:
            emb = torch.where(force_null[:, None], self.null_embed, emb)
        return emb


class TextTokenProjector(nn.Module):
    """Project per-token CLIP outputs to the DiT's hidden dim, for cross-attention.

    Cross-attention works most cleanly when keys/values share the model's hidden
    dim. CLIP-base outputs 512-dim per-token vectors; we project to whatever
    hidden the DiT uses (e.g., 256 or 384).
    """

    def __init__(self, text_dim, hidden):
        super().__init__()
        self.proj = nn.Linear(text_dim, hidden)
        # Learned null sequence for CFG: when conditioning is dropped, replace
        # the entire token sequence with this learned tensor.
        self.null_tokens = nn.Parameter(torch.zeros(1, 77, hidden))

    def forward(self, tokens, force_null=None):
        """Args:
            tokens: (B, L, text_dim) per-token CLIP outputs
            force_null: optional bool (B,) — True forces null tokens
        """
        proj = self.proj(tokens)  # (B, L, hidden)
        if force_null is not None:
            null = self.null_tokens.expand(proj.size(0), -1, -1)
            proj = torch.where(force_null[:, None, None], null, proj)
        return proj


# ─── RoPE-2D (same as lab 3.1) ──────────────────────────────────────────────


def rope_freqs(head_dim, h, w, theta=10000.0, device="cpu"):
    """Compute 2-D RoPE rotation frequencies for an h × w spatial grid.

    Splits head_dim in half: first half encodes height position, second half
    encodes width. Returns (cos, sin) of shape (h*w, head_dim) ready to be
    multiplied into Q and K.
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    quarter = head_dim // 4
    freqs_h = 1.0 / (theta ** (torch.arange(0, quarter, device=device).float() / quarter))
    freqs_w = 1.0 / (theta ** (torch.arange(0, quarter, device=device).float() / quarter))

    pos_h = torch.arange(h, device=device).float()
    pos_w = torch.arange(w, device=device).float()
    angles_h = torch.outer(pos_h, freqs_h)  # (h, quarter)
    angles_w = torch.outer(pos_w, freqs_w)  # (w, quarter)

    # Broadcast to (h, w, quarter)
    angles_h = angles_h[:, None, :].expand(h, w, quarter).reshape(h * w, quarter)
    angles_w = angles_w[None, :, :].expand(h, w, quarter).reshape(h * w, quarter)

    # Concat: first half-of-head encodes h, second encodes w
    cos = torch.cat([torch.cos(angles_h), torch.cos(angles_h),
                     torch.cos(angles_w), torch.cos(angles_w)], dim=-1)
    sin = torch.cat([torch.sin(angles_h), -torch.sin(angles_h),
                     torch.sin(angles_w), -torch.sin(angles_w)], dim=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply 2D RoPE to x: pair adjacent dims and rotate each pair.

    x: (B, H, L, head_dim)
    cos, sin: (L, head_dim)
    """
    # Pair adjacent dims (..., 2k) and (..., 2k+1) and rotate.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos1 = cos[..., 0::2]
    sin1 = sin[..., 0::2]
    cos2 = cos[..., 1::2]
    sin2 = sin[..., 1::2]
    rot1 = x1 * cos1 - x2 * sin1
    rot2 = x1 * sin2 + x2 * cos2
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


# ─── Attention modules ──────────────────────────────────────────────────────


class SelfAttention(nn.Module):
    """Multi-head self-attention over image tokens, with RoPE on Q/K."""

    def __init__(self, hidden, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        assert self.head_dim * num_heads == hidden
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.dropout = dropout

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        # RoPE on Q and K.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: image tokens (Q) attend to text tokens (K, V).

    NEW relative to lab 3.1. This is the mechanism that lets the model route
    information from individual words in the prompt to specific spatial
    positions in the image.

    No RoPE on text K (text doesn't have a 2D spatial position; the attention
    learns whatever positional structure it needs from CLIP's own position
    encoding, which is already baked into the text embeddings).
    """

    def __init__(self, hidden, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        assert self.head_dim * num_heads == hidden
        self.q = nn.Linear(hidden, hidden)
        self.kv = nn.Linear(hidden, 2 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.dropout = dropout

    def forward(self, x, text_tokens, attn_mask=None):
        """Args:
            x:           (B, L_img, hidden)  image tokens (queries)
            text_tokens: (B, L_txt, hidden)  text tokens (keys, values)
            attn_mask:   (B, L_txt) bool — True = attend, False = ignore (padding)
        """
        B, L_img, D = x.shape
        L_txt = text_tokens.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q = self.q(x).view(B, L_img, H, Dh).transpose(1, 2)
        k, v = self.kv(text_tokens).split(D, dim=-1)
        k = k.view(B, L_txt, H, Dh).transpose(1, 2)
        v = v.view(B, L_txt, H, Dh).transpose(1, 2)

        # Convert padding mask to additive bias.
        if attn_mask is not None:
            # attn_mask: (B, L_txt), True = real token. Need shape
            # (B, 1, 1, L_txt) for broadcasting against (B, H, L_img, L_txt).
            bias = torch.where(
                attn_mask[:, None, None, :].bool(),
                torch.zeros_like(attn_mask, dtype=q.dtype)[:, None, None, :],
                torch.full_like(attn_mask, -float("inf"), dtype=q.dtype)[:, None, None, :],
            )
        else:
            bias = None

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, L_img, D)
        return self.proj(out)


# ─── MLP, DiTBlock, FinalLayer ──────────────────────────────────────────────


class MLP(nn.Module):
    def __init__(self, hidden, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * mlp_ratio)
        self.fc2 = nn.Linear(hidden * mlp_ratio, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class DiTBlock(nn.Module):
    """DiT block with three sublayers: self-attn, cross-attn (text), MLP.

    AdaLN-Zero modulation from `c` controls self-attn and MLP. Cross-attn
    uses a plain LayerNorm with no modulation — the conditioning information
    is already coming in through the text tokens themselves.
    """

    def __init__(self, hidden, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.self_attn = SelfAttention(hidden, num_heads, dropout=dropout)

        self.norm_ca = nn.LayerNorm(hidden, elementwise_affine=False)
        self.cross_attn = CrossAttention(hidden, num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = MLP(hidden, mlp_ratio=mlp_ratio)

        # AdaLN-Zero: 6 modulation parameters (shift/scale/gate × 2 sublayers
        # that get modulated — self-attn and MLP). Cross-attn is unmodulated.
        self.ada_proj = nn.Linear(hidden, 6 * hidden)
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)

    def forward(self, x, c, text_tokens, text_mask, cos, sin):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.ada_proj(c).chunk(6, dim=-1)
        )
        x = x + gate_a.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_a, scale_a), cos, sin
        )
        x = x + self.cross_attn(self.norm_ca(x), text_tokens, attn_mask=text_mask)
        x = x + gate_m.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_m, scale_m)
        )
        return x


class FinalLayer(nn.Module):
    """Project tokens back to per-patch pixels with one final AdaLN modulation."""

    def __init__(self, hidden, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.linear = nn.Linear(hidden, patch_size * patch_size * out_channels)
        self.ada_proj = nn.Linear(hidden, 2 * hidden)
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.ada_proj(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ─── DiT main class ─────────────────────────────────────────────────────────


class DiT(nn.Module):
    """Latent text-to-image DiT.

    Inputs (per forward call):
        x:           (B, C_lat, H_lat, W_lat)  noisy latent
        t:           (B,)                       time in [0, 1]
        text_tokens: (B, L_txt, D_text)         per-token CLIP outputs
        text_pooled: (B, D_text)                pooled CLIP output
        text_mask:   (B, L_txt) bool            attention mask for text padding
        force_null:  optional (B,) bool         CFG: force unconditional path
    Output:
        v: (B, C_lat, H_lat, W_lat) predicted velocity
    """

    def __init__(self, latent_size=8, latent_channels=4, patch_size=2,
                 hidden=384, num_heads=6, num_blocks=8, mlp_ratio=4,
                 text_dim=512, p_uncond=0.1, dropout=0.0):
        super().__init__()
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.hidden = hidden

        self.patch_embed = nn.Conv2d(
            latent_channels, hidden, patch_size, stride=patch_size,
        )

        self.t_embed = TimestepEmbedder(hidden)
        self.text_pooled_proj = TextProjector(text_dim, hidden, p_uncond=p_uncond)
        self.text_tokens_proj = TextTokenProjector(text_dim, hidden)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.final = FinalLayer(hidden, patch_size, latent_channels)

        # Pre-compute RoPE frequencies for the spatial grid.
        n_patches = latent_size // patch_size
        head_dim = hidden // num_heads
        cos, sin = rope_freqs(head_dim, n_patches, n_patches)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def patchify(self, x):
        # (B, C, H, W) -> (B, L, hidden) with L = (H/p)*(W/p)
        x = self.patch_embed(x)
        return x.flatten(2).transpose(1, 2)

    def unpatchify(self, x):
        # (B, L, P*P*C) -> (B, C, H, W)
        B, L, _ = x.shape
        p = self.patch_size
        c = self.latent_channels
        h_p = w_p = self.latent_size // p
        x = x.view(B, h_p, w_p, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, c, h_p * p, w_p * p)

    def forward(self, x, t, text_tokens, text_pooled, text_mask, force_null=None):
        x = self.patchify(x)
        c = self.t_embed(t) + self.text_pooled_proj(text_pooled, force_null=force_null)
        text_kv = self.text_tokens_proj(text_tokens, force_null=force_null)

        for block in self.blocks:
            x = block(x, c, text_kv, text_mask, self.rope_cos, self.rope_sin)
        x = self.final(x, c)
        return self.unpatchify(x)

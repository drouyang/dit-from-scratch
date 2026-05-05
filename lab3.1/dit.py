"""DiT — Diffusion Transformer.

A transformer that operates on flattened image patches and is conditioned on
(timestep, class label) via AdaLN-Zero. Trained with flow matching (lab 2.2)
to predict velocity in pixel space.

The architecture is a small set of edits to lab 1.4's GPT block:

    GPT block  (lab 1.4)              DiT block  (here)
    ───────────────────────────       ─────────────────────────────────────
    LN(x)        → MHA → +res         AdaLN(x | c)  → MHA → gate(c) → +res
    LN(x)        → MLP → +res         AdaLN(x | c)  → MLP → gate(c) → +res
    causal mask                       no mask  (image gen is non-autoregressive)
    learned absolute pos embed        RoPE-2D  (applied to Q/K inside attention)

`c = t_emb + class_emb` is the conditioning vector that drives every
LayerNorm in every block. Everything else — pre-norm + residual structure,
multi-head attention with `F.scaled_dot_product_attention`, position-wise
GELU MLP — is verbatim from lab 1.4.

Pieces:
    sinusoidal_time_embed   — same as lab 2.2
    TimestepEmbedder        — sinusoidal time + 2-layer MLP
    LabelEmbedder           — class embedding with a null slot for CFG
    rope_freqs / apply_rope — 2-D Rotary Positional Embedding
    Attention               — MHA + RoPE-2D, no mask
    MLP                     — same shape as lab 1.4
    DiTBlock                — AdaLN-Zero around attention and MLP
    FinalLayer              — AdaLN + Linear projection back to patch pixels
    DiT                     — patchify → N × DiTBlock → final → unpatchify
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    """AdaLN's per-feature affine, applied to a sequence.

        x:     (B, L, D)
        shift: (B, D)
        scale: (B, D)

    Same shift/scale is broadcast across every token in the sequence — every
    patch gets the same conditioning, but each *feature dim* is shifted /
    scaled independently. The `1 + scale` form means scale=0 ⇒ identity,
    which is what AdaLN-Zero relies on at init.
    """
    return x * (1 + scale[:, None]) + shift[:, None]


def sinusoidal_time_embed(t, dim):
    """Sinusoidal embedding for continuous t. (B,) -> (B, dim).

    Identical to the helper in lab 2.2 (and the time-embedding pattern from
    lab 1.1). DiT uses this for the timestep input.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal time -> 2-layer MLP, output dim = hidden."""

    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t):
        return self.mlp(sinusoidal_time_embed(t, self.freq_dim))


class LabelEmbedder(nn.Module):
    """Class embedding with a learnable 'null' slot for CFG label-dropout.

    Same trick as lab 2.2: `nn.Embedding(num_classes + 1, hidden)` where the
    extra row at index `num_classes` is the unconditional class. Train with
    label-dropout, sample with CFG extrapolation between conditional and
    unconditional.
    """

    def __init__(self, num_classes, hidden):
        super().__init__()
        self.num_classes = num_classes
        self.null_class = num_classes
        self.embed = nn.Embedding(num_classes + 1, hidden)

    def forward(self, c):
        return self.embed(c)


# ─────────────────────────────────────────────────────────────────────────
# 2-D Rotary Positional Embedding (RoPE-2D)
# ─────────────────────────────────────────────────────────────────────────
#
# Pure attention is permutation-equivariant: shuffle the patch tokens and
# the outputs shuffle with them. To break that symmetry the model needs to
# know *where* each patch sits in the image. Lab 1.4's GPT solved this with
# a learned absolute position table (`pos_embed[i]` added at the input).
# Modern DiT-family models (SD3, FLUX, Lumina-T2X) use RoPE instead.
#
# 1-D RoPE in one paragraph. For each consecutive pair of dims (2i, 2i+1)
# of Q (and K), at position p, rotate that pair by angle p · θ_i, where
# θ_i = 1 / 10000^(2i/d). Because the rotation is identical on Q and K, the
# dot product Q·K becomes a function of (q_pos − k_pos) — purely *relative*
# position, baked directly into the attention logits with zero added params.
#
# 2-D extension. Image patches have a (h, w) coordinate, not a scalar
# position. Split each head's d_head into halves: rotate the first half by
# the row index h, the second half by the column index w. After the dot
# product, the "y-half" depends on (h_q − h_k), the "x-half" on (w_q − w_k).
# The model gets relative-y and relative-x simultaneously, which is what
# image attention naturally needs.
#
# Constraint: head_dim must be divisible by 4 (each axis half is divisible
# by 2 for the pair-wise rotation).


def rope_freqs(head_dim, h, w, theta=10000.0, device="cpu"):
    """Precompute (cos, sin) tables of shape (h*w, head_dim) for RoPE-2D.

    Layout: the first head_dim/2 columns rotate by row index h; the second
    half rotate by column index w. Generated once and reused across forward
    passes (cached as a buffer in `DiT`).
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for RoPE-2D"
    d_half = head_dim // 2  # per-axis dimensionality
    # One frequency per (2i, 2i+1) dim pair, geometric with base 10000.
    freqs = 1.0 / (theta ** (torch.arange(0, d_half, 2, device=device).float() / d_half))

    pos_h = torch.arange(h, device=device).float()
    pos_w = torch.arange(w, device=device).float()
    angles_h = pos_h[:, None] * freqs[None, :]   # (h, d_half/2)
    angles_w = pos_w[:, None] * freqs[None, :]   # (w, d_half/2)

    # Repeat each angle so both slots of a (2i, 2i+1) pair share one angle.
    angles_h = angles_h.repeat_interleave(2, dim=-1)   # (h, d_half)
    angles_w = angles_w.repeat_interleave(2, dim=-1)   # (w, d_half)

    # Broadcast to the full (h, w) grid.
    H = angles_h[:, None, :].expand(h, w, d_half)      # y-angle, varies in h
    W = angles_w[None, :, :].expand(h, w, d_half)      # x-angle, varies in w
    angles = torch.cat([H, W], dim=-1)                 # (h, w, head_dim)
    angles = angles.reshape(h * w, head_dim)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate consecutive dim pairs of x by the precomputed angles.

        x:   (B, H, L, head_dim)
        cos: (L, head_dim)
        sin: (L, head_dim)

    For each pair (2i, 2i+1):
        x'_{2i}   =  x_{2i}   · cos − x_{2i+1} · sin
        x'_{2i+1} =  x_{2i}   · sin + x_{2i+1} · cos
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos1 = cos[..., 0::2]
    sin1 = sin[..., 0::2]
    rot1 = x1 * cos1 - x2 * sin1
    rot2 = x1 * sin1 + x2 * cos1
    # Re-interleave the pair: stack along a new last dim, then flatten it.
    return torch.stack([rot1, rot2], dim=-1).flatten(-2)


# ─────────────────────────────────────────────────────────────────────────
# Attention + MLP (the GPT block, with two changes)
# ─────────────────────────────────────────────────────────────────────────


class Attention(nn.Module):
    """Multi-head self-attention with RoPE-2D applied to Q and K.

    Two differences from lab 1.4's `CausalSelfAttention`:
        1. No mask. Image generation is non-autoregressive — every patch can
           attend to every other patch.
        2. RoPE-2D rotates Q and K (not V) before the dot product, replacing
           the absolute learned position table that lab 1.4 added at input.

    The kernel itself is unchanged: F.scaled_dot_product_attention, dispatched
    to Flash Attention on supported hardware.
    """

    def __init__(self, hidden, num_heads):
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        # Packed Q/K/V projection — same layout as lab 1.4.
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x, rope_cos, rope_sin):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, H, Dh).transpose(1, 2)   # (B, H, L, Dh)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)
        # RoPE rotates positional information into Q and K. V is left alone:
        # positions matter for *who attends to whom*, not for the content
        # being passed through.
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        out = F.scaled_dot_product_attention(q, k, v)   # no mask
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class MLP(nn.Module):
    """Position-wise feed-forward: Linear → GELU → Linear.

    Identical to lab 1.4's `MLP`, minus dropout (DiT trains long enough that
    dropout isn't usually needed).
    """

    def __init__(self, hidden, mlp_ratio=4.0):
        super().__init__()
        inner = int(hidden * mlp_ratio)
        self.fc1 = nn.Linear(hidden, inner)
        self.fc2 = nn.Linear(inner, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ─────────────────────────────────────────────────────────────────────────
# DiT block — pre-norm transformer with AdaLN-Zero conditioning
# ─────────────────────────────────────────────────────────────────────────


class DiTBlock(nn.Module):
    """One DiT block: AdaLN-Zero modulation around attention and MLP.

    The AdaLN-Zero pattern (DiT paper, Peebles & Xie 2022):

        For each sublayer (attn, mlp):
            x'  =  norm(x)                     ← elementwise-affine OFF
            x'  =  x' * (1 + scale) + shift    ← per-feature affine from c
            y   =  sublayer(x')
            x   =  x + gate * y                ← residual, GATED by c

        Six per-block parameters per token:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        all produced from c by a single Linear (the "adaLN_modulation" head).

    Why "Zero"? The final Linear in `adaLN_modulation` is initialized to
    zero, so at init `gate_msa = gate_mlp = 0` (and `scale = shift = 0`).
    Each block is then *exactly* the identity at step 0 — a pure residual
    stack. Training learns each block's contribution from zero, which is a
    much friendlier optimization landscape than starting with random
    sublayer outputs added to the residual stream. This is the single
    biggest stability tweak DiT made over earlier conditioning recipes
    (in-context, cross-attention, plain AdaLN).
    """

    def __init__(self, hidden, num_heads, mlp_ratio=4.0):
        super().__init__()
        # elementwise_affine=False: AdaLN provides the per-feature scale/shift,
        # so vanilla LN's own affine is redundant (and would just fight the
        # conditioning).
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden, num_heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden, mlp_ratio)
        # One Linear produces all six modulation tensors. SiLU first because
        # the conditioning vector c is itself the output of a linear (in the
        # time/label embedders), so an activation here gives this projection
        # something nonlinear to consume.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, 6 * hidden),
        )

    def forward(self, x, c, rope_cos, rope_sin):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # Pre-norm sublayer pattern from lab 1.4, with two additions per
        # sublayer:
        #   1. modulate(...) replaces vanilla LN's affine with a per-(t, c)
        #      shift+scale. The model's normalization itself is conditioned
        #      on time and class.
        #   2. gate * y rescales the sublayer's residual contribution. At
        #      init gate=0, so each block is the identity; learned values
        #      let the model dial each sublayer in independently.
        x = x + gate_msa[:, None] * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope_cos, rope_sin
        )
        x = x + gate_mlp[:, None] * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """AdaLN + Linear projection back to patch-pixel space.

    Mirrors a DiT block's first half — AdaLN modulation of the residual
    stream — and then projects each token's hidden vector to a flat patch
    of `patch_size * patch_size * out_channels` numbers, which `unpatchify`
    re-tiles into an image.
    """

    def __init__(self, hidden, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden),   # only shift + scale, no gate
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────────────
# DiT model
# ─────────────────────────────────────────────────────────────────────────


class DiT(nn.Module):
    """Diffusion Transformer.

    Forward:
        1. patchify     Conv2d (kernel=stride=P) → (B, hidden, H/P, W/P) → (B, L, hidden)
        2. condition    c = t_embed(t) + y_embed(y)             (B, hidden)
        3. N × DiTBlock each modulated by c via AdaLN-Zero
        4. final layer  AdaLN + Linear → (B, L, P*P*C_out)
        5. unpatchify   reshape back to (B, C_out, H, W)        ← predicted velocity

    The output has the same shape as the input — the model predicts a
    velocity tensor for the flow-matching loss.
    """

    def __init__(self, image_size, in_channels, patch_size,
                 hidden, depth, num_heads, num_classes, mlp_ratio=4.0):
        super().__init__()
        assert image_size % patch_size == 0, \
            f"image_size {image_size} must be divisible by patch_size {patch_size}"
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.grid = image_size // patch_size  # tokens per side
        self.num_classes = num_classes
        self.null_class = num_classes
        self.hidden = hidden
        self.num_heads = num_heads

        # Patchify via strided conv. Kernel=stride=P means each non-overlapping
        # patch of P×P pixels is linearly projected to a `hidden`-dim vector —
        # mathematically identical to flattening each patch and applying a
        # Linear, but expressed as a conv so PyTorch can fuse it.
        #
        # (B, C, H, W)  →  Conv2d(C, hidden, kernel=P, stride=P)  →  (B, hidden, H/P, W/P)
        self.patch_embed = nn.Conv2d(in_channels, hidden, patch_size, stride=patch_size)
        self.t_embed = TimestepEmbedder(hidden)
        self.y_embed = LabelEmbedder(num_classes, hidden)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final = FinalLayer(hidden, patch_size, in_channels)

        # Precompute the RoPE-2D table once for the fixed (grid, grid) layout.
        # Stored as a non-persistent buffer so it moves with .to(device) but
        # isn't written into the checkpoint (we recompute on load).
        head_dim = hidden // num_heads
        cos, sin = rope_freqs(head_dim, self.grid, self.grid)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        # Generic init for linears.
        def basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(basic_init)

        # Patch conv: treat as a linear projection over flattened patch pixels.
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))
        nn.init.zeros_(self.patch_embed.bias)

        # Time/label embeddings: small std keeps `c` well-scaled at init.
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embed.embed.weight, std=0.02)

        # AdaLN-Zero. Zero out the *final* Linear in every adaLN_modulation
        # so that at init shift = scale = gate = 0 in every block:
        #   modulate(norm(x), 0, 0)  =  norm(x)        (vanilla LN)
        #   x + 0 * sublayer(norm(x)) =  x             (identity block)
        # The model starts as a pure pass-through and learns each block's
        # contribution from zero. This is what "Zero" in AdaLN-Zero means.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final.adaLN_modulation[-1].bias)

        # Final output projection — also zeroed, so the untrained model
        # predicts v = 0 (no flow). A reasonable, non-destructive starting
        # point: at t=1 a zero-velocity step keeps `x` at noise; the loss
        # is then exactly `||v_target||²` and the model only has to learn
        # to *increase* its output magnitude in useful directions.
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, x):
        """(B, L, P*P*C) → (B, C, H, W). Inverse of `patch_embed`."""
        B = x.size(0)
        P, G, C = self.patch_size, self.grid, self.in_channels
        x = x.view(B, G, G, P, P, C)
        # (B, gridH, gridW, P_h, P_w, C) → (B, C, gridH, P_h, gridW, P_w)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, C, G * P, G * P)

    def forward(self, x, t, y):
        """
        Args:
            x: (B, C, H, W) noised image at time t
            t: (B,)         time in [0, 1]
            y: (B,)         class index in [0, num_classes]   (num_classes = null)

        Returns:
            (B, C, H, W) predicted velocity
        """
        x = self.patch_embed(x)               # (B, hidden, G, G)
        x = x.flatten(2).transpose(1, 2)      # (B, L, hidden),  L = G*G
        c = self.t_embed(t) + self.y_embed(y) # (B, hidden) — the conditioning vector
        for block in self.blocks:
            x = block(x, c, self.rope_cos, self.rope_sin)
        x = self.final(x, c)                  # (B, L, P*P*C_in)
        return self.unpatchify(x)             # (B, C_in, H, W)

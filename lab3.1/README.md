# Module 3.1 — DiT architecture (patchify, AdaLN-Zero, RoPE-2D)

**Goal**: assemble the **Diffusion Transformer** — the architecture every modern image / video generator (SD3, FLUX, Lumina-T2X, WAN, LTX-Video) is built around — and train it on MNIST with the flow-matching loss from lab 2.2. By the end you have a class-conditional MNIST generator: pass in `c=7` and out comes a recognizable `7`. Lab 3.2 is the synthesis step that wraps a VAE around this same DiT and swaps the class label for a CLIP/T5 text embedding — the full production text-to-image stack at small scale. Neither change touches the architecture you build here.

**Why this matters for DiT**: this *is* DiT. After lab 1.4 you had a transformer block (LN → attn → +res, LN → MLP → +res); after lab 2.2 you had the flow-matching training loop *and* end-to-end class conditioning with CFG. The three pieces this lab adds — **patchify**, **AdaLN-Zero conditioning**, and **RoPE-2D** — are exactly what turns "a transformer" into "a Diffusion Transformer". Every line that's new here is one of those three; class conditioning, the time embedding, CFG, the loss, and the sampler all carry over from labs 1.4 / 2.2 unchanged.

**The continuity from lab 1.4's GPT block to a DiT block** in one diff:

| | GPT block (lab 1.4) | DiT block (here) |
|---|---|---|
| sublayer 1 | `LN → MHA → +res` | `AdaLN(c) → MHA → gate(c) → +res` |
| sublayer 2 | `LN → MLP → +res` | `AdaLN(c) → MLP → gate(c) → +res` |
| attention mask | causal (`is_causal=True`) | none (image gen is non-autoregressive) |
| position info | learned absolute `pos_embed` added at input | RoPE-2D applied to Q/K *inside* attention |
| inputs | tokens `(B, L)` | flat patches `(B, L, hidden)` from patchify |
| target | next-token logits | velocity tensor `(B, C, H, W)` |
| loss | cross-entropy | flow-matching MSE (lab 2.2) |

Where `c = t_embed(t) + class_embed(y)` is the **conditioning vector** that drives every LayerNorm in every block.

## Why MNIST and why pixel space

**MNIST.** Same dataset as lab 1.1 (classifier) and lab 2.1 (VAE). 60k 28×28 grayscale digits, 10 classes, fits in memory, trains in minutes on a laptop. Ten classes give CFG something to do (you get 10 distinct conditional distributions instead of 8 toy Gaussians from lab 2.2). And because you've already seen MNIST through a classifier and a VAE, the only new variable is the architecture — exactly what this lab is meant to isolate.

**Pixel space.** Production DiT models operate on **VAE latents** because real images are too big to do diffusion on directly: a 256×256×3 image has 196,608 pixels, and the attention pass over a patchified version of that would dominate compute. Compressing to a `(4, 32, 32)` latent first cuts the token count ~64×.

At MNIST's 28×28×1 = 784 pixels, that constraint doesn't apply — pixel-space DiT works directly. So this lab keeps things in pixel space, not as a debugging shortcut, but because *MNIST is small enough that the production motivation for using a VAE doesn't exist yet*. Lab 3.2 moves to a bigger dataset where the VAE becomes necessary, and that's where it gets introduced.

```
this lab (3.1):       image (1, 28, 28)   →  DiT  →  velocity (1, 28, 28)
lab 3.2 (bigger):     image  →  VAE.encode  →  z (C, H', W')  →  DiT  →  v  →  VAE.decode
                                ─────────────                   ─────────
                                spatial VAE                      same DiT
                                (drop-in: SD-VAE,                architecture
                                 or a small custom one)           as here
```

The DiT itself doesn't know whether its inputs are pixels or latents — it just sees a `(B, C, H, W)` tensor. That's why the architecture you write here works unchanged in lab 3.2.

## What the model learns

The model learns a **velocity field over the space of 28×28 images**, conditioned on the class:

```
input:  x ∈ R^(1,28,28)    — current (noisy) image
        t ∈ [0, 1]          — time
        c ∈ {0..10}          — class label (10 = null/unconditional, for CFG)
output: v ∈ R^(1,28,28)    — predicted velocity (same shape as input)
```

Lab 2.2 had a 2-D analogue of this — a velocity field over the plane. Here the field lives in 784-dimensional pixel space, and the "clusters" are the regions of pixel space where digits of each class actually exist. CFG concentrates samples toward those regions, just like it concentrated 2-D points around their cluster centers in lab 2.2.

## Files

| File | What it is |
| --- | --- |
| `data.py` | MNIST loader normalized to `[-1, 1]` (matches the unit-variance noise prior) |
| `dit.py` | `DiT`, `DiTBlock`, `Attention` (MHA + RoPE-2D), `MLP`, `FinalLayer`, `TimestepEmbedder`, `LabelEmbedder`, `rope_freqs` / `apply_rope`, `modulate` |
| `flow.py` | `fm_q_sample` + `fm_euler_sample` — same as `lab2.2/flow.py`, generalized to image-shape inputs |
| `train.py` | flow-matching training loop with `--label-dropout` for CFG, AdamW |
| `sample.py` | sampling CLI; saves a grid of generated digits |
| `visualize.py` | three figures: `samples.png`, `cfg.png`, `steps.png` |

## Instructions

### 1. Set up

Python 3.9+. From `lab3.1/`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training auto-selects device: CUDA → Apple MPS → CPU. MNIST downloads to `./data/` on first run (~12 MB).

### 2. Train

Default: flow matching with CFG label-dropout = 0.1, 20 epochs, batch 128.

```bash
python train.py
```

~15 minutes on M3 MPS. Saves `dit.pt`.

Per-step log (one line per `--log-every` steps):

```
device: mps
model: 1.87M params  (patch=4, hidden=128, depth=6, heads=4)
step    200  |   45.3s  |  epoch   1  |  loss 0.7124
step    400  |   90.1s  |  epoch   1  |  loss 0.5891
...
step   9000  |  830.5s  |  epoch  20  |  loss 0.3812
saved dit.pt
```

The flow-matching MSE on MNIST plateaus around `0.35–0.40` — that's the floor; getting it lower needs a much bigger model or longer training and won't visibly improve sample quality at this scale.

Useful knobs:

```bash
python train.py --epochs 40                 # longer training, sharper digits
python train.py --hidden 192 --depth 8      # ~5M params, better but slower
python train.py --label-dropout 0.0         # disable CFG (samples ignore class)
python train.py --patch-size 2              # finer tokens, 196 tokens, 4× slower
```

### 3. Sample

After training:

```bash
python sample.py                             # 8 samples/class, cfg=4
python sample.py --cfg-scale 2.0 --steps 20  # weaker CFG, fewer steps
python sample.py --class-id 7 --n-per-class 16
```

Saves a grid to `samples.png`. Rows are classes (0 → 9), columns are independent samples.

### 4. Visualize

```bash
python visualize.py --mode all
```

Three figures:

- **`samples.png`** — `n_per_class` samples for each digit, one row per class. With default settings (`steps=50, cfg=4.0`) you should see recognizable digits with reasonable diversity. Some failure modes are expected at this model size — occasional malformed strokes, ambiguous 4/9 or 3/8 confusions.

- **`cfg.png`** — same starting noise across `cfg ∈ {0, 1, 2, 4, 7}`. At `cfg=0` the model samples unconditionally — the class label is ignored, so you see digits that don't necessarily match the row label. At `cfg=1` you get the model's natural conditional distribution (correct class, lots of diversity). At `cfg=4` the conditioning is sharp (correct class, less diversity). At `cfg=7` samples often look over-saturated — the same trade-off you saw in lab 2.2.

- **`steps.png`** — same noise across `n_steps ∈ {1, 2, 4, 8, 16, 50}`. As in lab 2.2, the headline result: very few Euler steps suffice for a flow-matching model. `N=4–8` already produces plausible digits; `N=50` is barely distinguishable from `N=16`. This is the "few-step sampling" advantage that makes diffusion deployable in real-time settings.

## The three DiT-specific ideas

Read `dit.py` top to bottom; it walks through these in order. The three ideas:

### 1. Patchify

A transformer operates on a *sequence of vectors*. To feed it an image, slice the image into non-overlapping `P × P` patches and project each patch to a `hidden`-dim vector — that's the "tokenization" step for images, exactly analogous to how lab 1.4's GPT turned characters into 384-dim embeddings.

```
input image          patches              tokens
(B, 1, 28, 28)  →    (B, 1, 7×7 patches  →   (B, 49, hidden)
                          of 4×4)
```

Implementation is one strided convolution:

```python
self.patch_embed = nn.Conv2d(in_channels, hidden, patch_size, stride=patch_size)
```

Kernel size = stride = patch size means each `P × P` patch is independently linearly projected to a `hidden`-dim vector. Mathematically identical to `Flatten` + `Linear` per patch, but expressed as a conv so PyTorch can fuse it. After the conv, flatten the spatial dims to get `(B, L, hidden)` with `L = (H/P) × (W/P) = 49` for our 28×28 → 4-patch case.

**Why this is the same as ViT.** Vision Transformer (Dosovitskiy 2020) introduced this trick — *tokens are image patches* — and DiT inherited it directly. The hidden dim acts as the per-patch feature width; deeper / wider DiTs just use more patches and bigger hidden dims. SD3 uses 2×2 patches on 64×64 latents.

**Unpatchify is the inverse.** After the transformer stack, the final layer projects each token back to `P × P × C` numbers, and `unpatchify` re-tiles the patches into a `(B, C, H, W)` image. The model output has the same shape as the input — exactly what flow matching's MSE expects.

**Patchify vs VAE encode (worth distinguishing).** Both are "image ↔ vectors" round-trips, but they do completely different jobs:

| | Patchify / unpatchify | VAE encode / decode |
|---|---|---|
| Lossy? | No — pure linear, mathematically invertible | Yes (small reconstruction error even at perfect convergence) |
| Compresses? | No — usually *expands* per-position (e.g., 4×4×3 = 48 pixels → 384 hidden) | Yes — SD-VAE compresses 256×256×3 ≈ 200k floats → 32×32×4 ≈ 4k floats |
| Trained how? | The single linear is trained jointly with DiT; no special objective | Separately, with reconstruction loss + KL prior |
| Purpose | **Format conversion**: image grid → sequence of tokens (so a transformer can consume it) | **Information bottleneck**: compress + regularize the space the DiT operates in |

A clean way to think about it: patchify is to images what `nn.Embedding` is to token IDs in lab 1.4's GPT — *format conversion*, not compression. The VAE (introduced in lab 2.1) is the actual encoder/decoder. In lab 3.2 you'll see both compositions stacked: VAE compresses image → latent, patchify reshapes latent → token sequence, transformer processes, unpatchify reshapes back, VAE decodes to image.

### 2. AdaLN-Zero — the DiT paper's main contribution

Lab 1.4's block was **pre-norm**:

```python
x = x + sublayer(LayerNorm(x))
```

DiT replaces vanilla LayerNorm with an **adaptive LayerNorm conditioned on `c`**, and adds a **gate** to the residual:

```python
x = x + gate_msa * attn(modulate(LN(x), shift_msa, scale_msa))
x = x + gate_mlp * mlp (modulate(LN(x), shift_mlp, scale_mlp))
```

**Where does `c` come from?** Two components reused from lab 2.2 — sinusoidal time embedding and class embedding with a null slot for CFG — folded into one shared vector:

```python
c = t_embed(t) + class_embed(y)    # (B, hidden)
```

- **`t_embed`** — sinusoidal time embedding (same helper as lab 2.2 / lab 1.1) followed by a 2-layer MLP. Output shape `(B, hidden)`.
- **`class_embed`** — `nn.Embedding(num_classes + 1, hidden)`. The `+1` row is the **null class** for CFG label-dropout, exactly like lab 2.2's `TimeMLP`.

The DiT-specific bit is the *folding* — lab 2.2's MLP concatenated `x_emb`, `t_emb`, `c_emb` separately; DiT sums time and class into one shared `c` that then drives every block. This is what makes the conditioning trivial to extend in lab 3.2 — there `c = t_embed(t) + text_embed(prompt)` and nothing else in the architecture changes. CFG itself (label-dropout in training, `v_uncond + s·(v_cond − v_uncond)` at sampling) carries over from lab 2.2 unchanged.

**The modulation parameters.** `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)` are six per-block tensors of shape `(B, hidden)`, all produced from `c` by a single linear projection per block (`adaLN_modulation`). `modulate` does the per-feature affine:

```python
def modulate(x, shift, scale):
    return x * (1 + scale) + shift
```

**What "Zero" means.** The final `Linear` in every block's `adaLN_modulation` is initialized to **zero**. So at init:

- `shift = scale = 0` → `modulate(LN(x), 0, 0) = LN(x)` (vanilla LN)
- `gate = 0` → `x + 0 * sublayer(...) = x` (the block is the identity)

The model starts as a *pure* residual stack — every block passes its input through unchanged, and the final-layer linear (also zero-initialized) outputs `v = 0`. Training then learns each block's contribution from zero. This is much friendlier than the standard "small random init" because:

1. **No fight at init.** Random sublayer outputs added to the residual stream at depth 12 sum to large random perturbations; AdaLN-Zero has zero perturbation at depth N.
2. **Each gate is a learned switch.** A block that isn't pulling its weight can keep its gate near zero and the residual flows around it cleanly. Useful capacity emerges layer by layer.
3. **The output is a learned signal, not a coincidence.** At step 0 the model predicts `v = 0` — a meaningful prior ("don't move"). The loss is then exactly `||v_target||²` and gradients tell the model what *direction* to push outputs in. No "lucky" early predictions to chase.

The DiT paper compares four conditioning mechanisms (in-context, cross-attention, plain AdaLN, AdaLN-Zero) and finds AdaLN-Zero wins by a wide margin in FID at every scale they tried. Empirically this single trick is the largest reason DiT trains stably; everything else (patchify, attention shape, flow matching) is shared with non-DiT generators.

**`adaLN_modulation` is just an MLP.** Per block:

```python
self.adaLN_modulation = nn.Sequential(
    nn.SiLU(),
    nn.Linear(hidden, 6 * hidden),    # ← initialized to ZERO
)
```

The lab 1.1 pattern (sinusoidal time → MLP) shows up here too: the conditioning vector `c` is an MLP output, and another linear inside each block decodes it into modulation parameters. Counting MLPs in DiT: one for the time embedder, one for `adaLN_modulation` per block, and the FFN inside each block. Every one of them is the pattern from lab 1.1.

### 3. RoPE-2D — relative position, applied inside attention

A pure attention layer is **permutation-equivariant**: shuffle the input tokens and the outputs shuffle the same way. To break that symmetry the model needs position info. Lab 1.4's GPT solved this with a learned absolute position table — `pos_embed[i]` added to the input before any attention. The original DiT paper used a fixed sin-cos 2-D position embedding the same way.

Modern DiT-family models (SD3, FLUX, Lumina-T2X) use **RoPE** instead, and that's what `dit.py` implements.

**1-D RoPE in one paragraph.** For each consecutive pair of dims `(2i, 2i+1)` of `Q` (and `K`), at position `p`, rotate that pair by angle `p · θ_i` where `θ_i = 1 / 10000^(2i/d)`. Because the rotation is identical on `Q` and `K`, the dot product `Q · K` becomes a function of `(q_pos − k_pos)` — purely *relative* position, baked directly into the attention logits with **zero added parameters**. (This is one of the cleanest mechanism-level ideas in the post-Transformer literature; the math is in `apply_rope`.)

**2-D extension.** Image patches have `(h, w)` coordinates, not a scalar position. Split each head's `head_dim` into halves: rotate the first half by the row index `h`, the second half by the column index `w`. After the dot product, the y-half depends on `(h_q − h_k)` and the x-half on `(w_q − w_k)`. The model gets relative-y *and* relative-x for free.

```
head_dim = 32
   ┌────────────────┬────────────────┐
   │  16 dims:      │  16 dims:      │
   │  rotate by h   │  rotate by w   │
   └────────────────┴────────────────┘
       y-RoPE              x-RoPE
```

Implementation lives in `rope_freqs` (precompute the cos/sin tables once for the fixed `7×7` grid) and `apply_rope` (rotate dim pairs of Q and K). RoPE is *not* applied to V — positions matter for **who attends to whom**, not for the content being routed.

**Why RoPE over learned absolute embeddings.** Three reasons production migrated:

1. **Relative is the right inductive bias for images.** What matters for attention is "this patch is two to the left of that patch", not "this patch is at absolute index 17." RoPE encodes the relative offset directly into the dot product.
2. **Generalizes across resolutions.** A learned table of size `7 × 7` doesn't extend to a `14 × 14` grid; RoPE just takes a different `(h, w)` index and recomputes the rotation. Production text-to-image models (SD3, FLUX) train on multiple resolutions with the same weights — RoPE makes that trivial.
3. **Zero parameters.** Learned position embeddings add `L · D` parameters; RoPE adds zero.

The trade-off RoPE imposes is `head_dim % 4 == 0` (so each axis half is divisible by 2 for the pair-wise rotation). With `hidden=128, num_heads=4` we have `head_dim=32` ✓.

## Putting it together

```
input image (B, 1, 28, 28)
   │
   ▼
patchify (Conv2d k=4, s=4) ─────────► tokens (B, 49, 128)
   │
   │   c = t_embed(t) + class_embed(y)       (B, 128)
   │       └── sinusoidal-time + MLP    └── nn.Embedding(11, 128)
   │
   ▼
DiTBlock × 6  (each one):
   │     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(c)
   │     x = x + gate_msa · MHA-with-RoPE-2D ( modulate(LN(x), shift_msa, scale_msa) )
   │     x = x + gate_mlp · MLP              ( modulate(LN(x), shift_mlp, scale_mlp) )
   ▼
final layer:
   │     shift, scale = adaLN_modulation(c)
   │     x = modulate(LN(x), shift, scale)
   │     x = Linear(hidden, P*P*C_out)(x)        # (B, 49, 16)
   ▼
unpatchify  ───────────────────────► velocity (B, 1, 28, 28)
```

At default config (`patch=4, hidden=128, depth=6, heads=4`) this is **~1.9M parameters** — small enough to train on M3 MPS in ~15 minutes.

## Discussion

### What's identical to lab 1.4

- **The block recipe.** `pre-norm + residual` around two sublayers (attention, MLP), in that order.
- **The MLP shape.** `Linear(D, 4D) → GELU → Linear(4D, D)`, position-wise.
- **The attention kernel.** `F.scaled_dot_product_attention` with multi-head Q/K/V projections, exactly as you used in lab 1.4 once you'd verified it bit-for-bit against PyTorch's built-in in lab 1.3.
- **Pre-norm + residual structure.** LayerNorm before each sublayer, residual added after — gradients reach early layers without vanishing, blocks can learn to do nothing.

### What's identical to lab 2.2

- **The training loop.** Sample t uniformly, compute `x_t` with the closed-form forward process, predict velocity, MSE on the supervision target. ~15 lines.
- **The sampler.** Euler integration of `dx/dt = v(x, t, c)` from t=1 to t=0. Same `fm_euler_sample`, generalized to image shapes.
- **CFG.** Train with label-dropout to a null class; sample with `v_uncond + s · (v_cond - v_uncond)`. Identical mechanism, paradigm-agnostic.
- **The conditioning idea.** A single learnable vector per class (with a null slot for CFG) added into the model. The 8-class toy stand-in for text in lab 2.2 → 10-class digit conditioning here → text-embedding conditioning in lab 3.2. Same shape of mechanism, different sources for the embedding.

### What's new in this lab

Three things, all in `dit.py`:

1. **Patchify**, the ViT/DiT trick that turns an image into a sequence of tokens.
2. **AdaLN-Zero** — replaces vanilla LN's affine with a per-`c` shift+scale (where `c = t_embed(t) + class_embed(y)` folds time and class into one shared vector), gates each sublayer's residual contribution by `c`, zero-initializes the final modulation linear so blocks start as identity.
3. **RoPE-2D**, relative-position-info-as-rotation, applied to Q and K inside attention. Replaces lab 1.4's learned absolute position table.

That's the entire architectural delta from lab 1.4 to a modern image DiT. Every other line is shared — and the flow-matching training stack is shared with lab 2.2.

### What changes for lab 3.2

**Lab 3.2 — Latent text-to-image DiT.** Two changes from this lab, both small:

1. **Latents instead of pixels.** Plug lab 2.1's VAE in front of and behind the DiT:

   ```
      image  ──VAE.encode──►  z (B, C_lat, H_lat, W_lat)  ──DiT──►  v
                                                                     │
      image' ◄─VAE.decode─── z + integrate(v) ◄────────── (sampler) ─┘
   ```

   The DiT now operates on *latents* (4 channels, 8× smaller spatial dims) instead of pixels. Encode the batch with the frozen VAE before running flow matching; everything else (`fm_q_sample`, the loss, the sampler) is unchanged. The DiT *itself* doesn't change at all — only its `in_channels`, `image_size`, and `patch_size` are reconfigured for the latent shape.

2. **Text instead of class labels.** Swap class labels for text embeddings:

   ```
      "a photo of a 7"  ──CLIP/T5──►  text_emb  ──┐
                                                   ▼
                                        c = t_embed + text_proj(text_emb)
   ```

   `LabelEmbedder` (an `nn.Embedding`) is replaced by a small `Linear` that projects the frozen text encoder's output to `hidden`-dim, and AdaLN-Zero is augmented with **cross-attention** to the per-token text embeddings (so the model can attend to *individual words* in the prompt, not just the prompt-level summary). The flow-matching stack and the DiT block recipe are otherwise unchanged.

The combination of (1) and (2) is the SD3 / FLUX recipe at small scale. Lab 3.2 trains it end-to-end on a tiny text-image dataset.

### Where to go deeper

- Original DiT paper (Peebles & Xie 2022) — read sections 3 (architecture) and 4.1 (the AdaLN-Zero ablation). The four-way conditioning comparison is the empirical core of the paper.
- SD3 / MMDiT (Esser et al. 2024) — DiT + latent diffusion + flow matching + multimodal text/image attention, all in one. Reading this paper after labs 2.1, 2.2, 3.1, and 3.2 should feel like a tour of mechanisms you've already implemented.
- HuggingFace `diffusers`'s `DiTTransformer2DModel` — production reference implementation; structurally near-identical to `dit.py`, just with more knobs (resolution, channel counts, conditioning sources).

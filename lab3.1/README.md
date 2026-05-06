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

Lab 1.4's GPT block was:

```python
x = x + attn(LayerNorm(x))                    # sublayer 1: multi-head self-attention
x = x + mlp (LayerNorm(x))                    # sublayer 2: position-wise MLP (FFN)
```

A DiT block adds `c`-driven modulation around each sublayer's LN, gates each sublayer's residual contribution, and rotates Q/K inside MHA via RoPE:

```python
x = x + gate_msa * attn(modulate(LN(x), shift_msa, scale_msa))
x = x + gate_mlp * mlp (modulate(LN(x), shift_mlp, scale_mlp))
```

Each sublayer is two steps — **predict** the modulation parameters from `c`, then **apply** them to the activations:

```python
shift, scale, gate, ... = adaLN_modulation(c).chunk(6, dim=-1)   # predict
y = x + gate * sublayer(modulate(LN(x), shift, scale))           # apply
```

Same backbone (pre-norm + residual around two sublayers), three new things:

- **`modulate(shift, scale)` inside the LN path** — vanilla LN's affine replaced by per-`c` shift+scale.
- **`× gate` on the residual** — each sublayer's contribution is gated by a per-`c` scalar; zero-init means blocks start as identity.
- **`MHA+RoPE`** — Q and K get rotated by their (h, w) position before the dot product, so attention sees relative position with zero added parameters.

That's the entire architectural delta from "transformer" to "Diffusion Transformer."

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

**Unpatchify is the inverse.** After the transformer stack, the final layer projects each token back to `P × P × C` numbers, and `unpatchify` re-tiles the patches into a `(B, C, H, W)` image. The model output has the same shape as the input — exactly what flow matching's MSE expects.

**Patchify vs VAE encode (worth distinguishing).** Both are "image ↔ vectors" round-trips, but they do completely different jobs:

| | Patchify / unpatchify | VAE encode / decode |
|---|---|---|
| Lossy? | No — pure linear, mathematically invertible | Yes (small reconstruction error even at perfect convergence) |
| Compresses? | No — usually *expands* per-position (e.g., 4×4×3 = 48 pixels → 384 hidden) | Yes — SD-VAE compresses 256×256×3 ≈ 200k floats → 32×32×4 ≈ 4k floats |
| Trained how? | The single linear is trained jointly with DiT; no special objective | Separately, with reconstruction loss + KL prior |
| Purpose | **Format conversion**: image grid → sequence of tokens (so a transformer can consume it) | **Information bottleneck**: compress + regularize the space the DiT operates in |

If you removed the VAE, the DiT would still patchify and unpatchify pixels directly. If you removed patchify, the DiT couldn't run at all — its attention layers expect a sequence, not a 2D feature map.

In lab 3.2 you'll see both compositions stacked: VAE compresses image → latent, patchify reshapes latent → token sequence, transformer processes, unpatchify reshapes back, VAE decodes to image.

### 2. AdaLN-Zero — the DiT paper's main contribution

Lab 1.4's block was **pre-norm** — LayerNorm runs on the input *before* the sublayer, and the residual is added afterward, so the residual stream itself is never normalized:

```python
x = x + sublayer(LayerNorm(x))                # ← pre-norm: LN before sublayer, residual untouched
```

`sublayer` here is a placeholder for either of the two sublayers inside a transformer block — **attention** or **MLP**. Each block applies the pre-norm pattern twice, once per sublayer:

```python
x = x + attn(LayerNorm(x))                    # sublayer 1: multi-head self-attention
x = x + mlp (LayerNorm(x))                    # sublayer 2: position-wise MLP (FFN)
```

(Compare the original 2017 Transformer's **post-norm** — `LayerNorm(x + sublayer(x))` — which sits LN *on* the residual path. Post-norm is harder to train stably past ~12 stacked blocks and needs careful learning-rate warmup; pre-norm leaves a clean unmolested residual, so gradients from deep layers reach early layers without rescaling. Lab 1.4 has the full discussion. GPT-2, LLaMA, ViT, BERT-large, and DiT all use pre-norm.)

DiT replaces vanilla LayerNorm with an **adaptive LayerNorm conditioned on `c`**, and adds a **gate** to the residual:

```python
x = x + gate_msa * attn(modulate(LN(x), shift_msa, scale_msa))
x = x + gate_mlp * mlp (modulate(LN(x), shift_mlp, scale_mlp))
```

```python
c = t_embed(t) + class_embed(y)    # (B, hidden)
```

- **`t_embed`** — sinusoidal time embedding (same helper as lab 2.2 / lab 1.1) followed by a 2-layer MLP. Output shape `(B, hidden)`.
- **`class_embed`** — `nn.Embedding(num_classes + 1, hidden)`. The `+1` row is the **null class** for CFG label-dropout, exactly like lab 2.2's `TimeMLP`.

The DiT-specific bit is the *folding* — lab 2.2's MLP concatenated `x_emb`, `t_emb`, `c_emb` separately; DiT sums time and class into one shared `c` that then drives every block. This is what makes the conditioning trivial to extend in lab 3.2 — there `c = t_embed(t) + text_embed(prompt)` and nothing else in the architecture changes. CFG itself (label-dropout in training, `v_uncond + s·(v_cond − v_uncond)` at sampling) carries over from lab 2.2 unchanged.

**The modulation parameters.** `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)` are six per-block tensors of shape `(B, hidden)`, all produced from `c` by a single linear projection per block (`adaLN_modulation`). `modulate` does the per-feature affine (`y = A·x + b`):

```python
def modulate(x, shift, scale):
    return x * (1 + scale) + shift
```

**What the "Zero" in AdaLN-Zero means.** The final `Linear` in every block's `adaLN_modulation` is initialized to **zero**. So at init:

- `shift = scale = 0` → `modulate(LN(x), 0, 0) = LN(x)` (vanilla LN)
- `gate = 0` → `x + 0 * sublayer(...) = x` (the block is the identity)

The model starts as a *pure* residual stack — every block passes its input through unchanged, and the final-layer linear (also zero-initialized) outputs `v = 0`. Training then learns each block's contribution from zero.

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

**Worked example on a 4×4 image grid.** Take a Q at position `(2, 3)` on this patch grid:

```
   col→  0      1      2      3
 row↓  ┌──────┬──────┬──────┬──────┐
   0   │      │      │      │      │
       ├──────┼──────┼──────┼──────┤
   1   │      │      │      │      │
       ├──────┼──────┼──────┼──────┤
   2   │      │      │      │  Q   │  ← query at (2, 3)
       ├──────┼──────┼──────┼──────┤
   3   │      │      │      │      │
       └──────┴──────┴──────┴──────┘
```

**Q and K carry two ingredients.** Each is a learned **content vector** (what the patch *is* — cat fur, wheel edge, sky) coming from `W_q`, `W_k`. RoPE then **rotates** that content vector by an angle that depends on the patch's position. So Q at position `(h_q, w_q)` is "Q's content, rotated by row `h_q` and column `w_q`."

**The half-split.** For 2-D RoPE, split `head_dim` in half. The **first half** of the content vector is rotated by an angle proportional to the **row index** (`θ·h_q` for Q, `θ·h_k` for K). The **second half** by the **column index** (`θ·w_q`, `θ·w_k`). With one frequency per axis (using `θ = π/4` for clean arithmetic; real RoPE uses much smaller values), the rotated dot product expands into:

```
logit  ≈  cos((h_q − h_k) · π/4)  ·  ⟨q_h, k_h⟩       (row term)
        + cos((w_q − w_k) · π/4)  ·  ⟨q_w, k_w⟩       (col term)
        + (smaller sin cross-terms)
```

Each term factors cleanly into two pieces:

- **Position factor**: `cos(Δh · π/4)`, `cos(Δw · π/4)` — depends *only* on the offset along that axis. Pure geometry, identical for every Q/K pair at the same offset, no learning involved.
- **Content alignment**: `⟨q_h, k_h⟩`, `⟨q_w, k_w⟩` — the unrotated content halves dotted together. Depends *only* on what Q and K contain, not where they sit. This is what attention would compute if there were no position info at all. Learned via `W_q`, `W_k`.

The position halves cancel in the dot product because RoPE rotates Q by `+h_q` and K by `+h_k`, so the rotation that survives in the cosine is the difference `h_q − h_k`. That's why each cosine sees only `Δh` or `Δw`, never absolute positions.

**Position factors at Q = (2, 3) vs four candidate keys** (`row cos` ≡ `cos(Δh · π/4)`, `col cos` ≡ `cos(Δw · π/4)`):

| K position | Δh = h_q − h_k | Δw = w_q − w_k | row cos | col cos |
|---|---|---|---|---|
| **(2, 3)** — same patch | 0 | 0 | cos(0) = **1.00** | cos(0) = **1.00** |
| **(1, 3)** — directly above | +1 | 0 | cos(π/4) ≈ 0.71 | 1.00 |
| **(2, 2)** — directly left | 0 | +1 | 1.00 | 0.71 |
| **(0, 0)** — far up-and-left | +2 | +3 | cos(π/2) = **0.00** | cos(3π/4) ≈ −0.71 |

Example: for K at (1, 3) directly above Q, `logit = 0.71·⟨q_h, k_h⟩ + 1.00·⟨q_w, k_w⟩`. If that K's content matches Q's strongly (both are cat-edge features), the inner products are large and Q attends to it. If the content doesn't match (one is cat, one is sky), the inner products are small or negative and Q ignores it — *despite* the favorable position.

**Thought experiment — what RoPE alone contributes.** Suppose for a moment that all keys had the *same* content alignment with Q (`⟨q_h, k_h⟩ = ⟨q_w, k_w⟩ = 1` everywhere). Then content can't pick winners, and the logit reduces to `row_cos + col_cos` — read straight off the table. The pattern peaks at Q's own position (2.00), gently falls off to neighbors (1.71), and goes negative far away (−0.71). That's a **cos-shaped soft locality bias** that costs zero parameters — the same inductive bias CNNs hard-code via small kernels, but emerging here from the rotation geometry alone.

In real training content is *not* uniform — it does most of the work in deciding who attends to whom. The cosines just modulate it. By shaping Q and K's content vectors during training, different attention heads can override the locality default and learn different relative-offset preferences:

- **Local head**: content vectors uniform → cos modulation gives a soft locality bias (attend to neighbors).
- **Directional head**: Q/K shaped to peak at a specific `(Δh, Δw)` — e.g., "always look up-and-to-the-left."
- **Global head**: small content magnitude → cos modulation is a small ripple; attention mostly content-based, position-independent.

These are all *learnable* in the same model. Production DiTs end up with heads specializing this way automatically, all from the same RoPE-2D mechanism with zero added parameters.

**Why different heads learn different things at all** (true in any transformer, not RoPE-specific):

1. **Independent parameters.** Each head has its own `W_q`, `W_k`, `W_v` projections. They start from different random initializations.
2. **Gradient descent + capacity pressure.** If two heads converged to the same function, the model is wasting half its capacity. The optimizer gets a stronger gradient signal by spreading work across heads — heads that find a "niche" reduce loss more than redundant heads.
3. **Empirical observation.** In every trained transformer (text, vision, audio), probing reveals heads specialize: some attend to syntactic neighbors, some to long-range coreference, some to semantic similarity, etc. It emerges, no one programs it in.

**What RoPE-2D adds**: because position is now part of the dot product geometry, the space of features a head can specialize in includes spatial patterns. So the "feature" a head learns can be:

- **Content-only**: "match cat-edge features wherever they are" — content alignment dominates, position cosine averaged out.
- **Position-biased**: "match patches that are one step above me" — content shaped to peak when `Δh = +1, Δw = 0`.
- **Locality**: "trust my neighbors" — content roughly uniform, the bare RoPE cosine carries the day.

So different heads learn different features (general fact), and features can include spatial patterns because position is in the geometry (RoPE's contribution) — together, heads naturally specialize into different relative-offset preferences with no extra parameters or supervision. The DiT gets this for free.

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


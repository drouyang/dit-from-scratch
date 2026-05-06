# Module 3.2 — Latent text-to-image DiT

**Goal**: train a real text-to-image DiT — pretrained CLIP text encoder + pretrained SD-VAE + the lab 3.1 DiT architecture (with cross-attention added) — on a tiny subset of **MS-COCO** (Common Objects in Context: 330K natural photographs, each with five human-written captions like *"a man riding a bike on a city street"* — the canonical image-text benchmark). 5K image-caption pairs is tiny by production standards but enough that the text encoder is doing real work over real natural language, not class labels. End-to-end "type a prompt, get an image." This is the SD3 / FLUX recipe at small scale.

**Why this matters for DiT**: this is the first lab where everything is *production-shape*. Pretrained text encoder, pretrained VAE, learned DiT in the middle, flow matching as the training paradigm, CFG for conditioning strength — exactly the stack that runs in production text-to-image. Lab 3.1 verified the DiT architecture in isolation; this lab puts it inside the actual production pipeline.

## What changed from lab 3.1

Two architectural changes; everything else carries over.

**1. Pixels become latents.** SD-VAE's encoder turns a 64×64 RGB image into a (4, 8, 8) spatial latent. The DiT operates on that latent shape. SD-VAE's decoder turns the generated latent back into an image at sampling time. The VAE is **pretrained** — we use OpenAI / Stability's checkpoint directly, no training.

```
training:    image (3, 64, 64) ──VAE.encode──► z (4, 8, 8) ──flow matching on z──
sampling:    z (4, 8, 8) ──VAE.decode──► image (3, 64, 64)
```

The DiT itself doesn't know it's operating on latents — it just sees a `(B, 4, 8, 8)` tensor. Same architecture as lab 3.1, just reconfigured: `latent_channels=4`, `latent_size=8`, `patch_size=2`.

**2. Class labels become text embeddings.** Two encoding granularities are needed, because each handles a job the other can't:

- **AdaLN-Zero needs a single per-prompt vector.** We use CLIP's **pooled** output (B, 512) in the modulation: `c = t_embed(t) + text_proj(pooled)`.
- **Attribute binding needs word-level structure.** Pooled collapses too much: "a red car" and "a blue car" produce nearly identical pooled vectors, so the model can't reliably tie "red" to the car. CLIP's **per-token** outputs (B, 77, 512) preserve the structure. We feed them as keys/values into a new **cross-attention** sublayer in each DiT block, where image patches (queries) decide locally which words to listen to.

CLIP gives us both forms in one forward pass, but the *reason* we use both is that neither alone is enough — pooled-only loses attribute binding; per-token-only loses the AdaLN-Zero modulation channel.

So each DiT block now has three sublayers instead of two:

```
x = x + gate * self_attn(AdaLN(x, c))           ← image self-attention (lab 3.1)
x = x + cross_attn(LN(x), text_tokens)          ← NEW: image attends to text
x = x + gate * mlp(AdaLN(x, c))                 ← MLP (lab 3.1)
```

Cross-attention is unmodulated (no AdaLN gating around it) — the text tokens already carry conditioning information through their values.

## Files

| File | What it is |
| --- | --- |
| `vae.py` | Pretrained SD-VAE (`stabilityai/sd-vae-ft-mse`) wrapper: `.encode(image) → latent`, `.decode(latent) → image`. Includes the standard 0.18215 scale factor. |
| `text_encoder.py` | Pretrained CLIP text encoder (`openai/clip-vit-base-patch32`) wrapper. Returns per-token outputs (for cross-attention), pooled output (for AdaLN), and attention mask. |
| `data.py` | Tiny COCO subset (~5K image-caption pairs) loaded via HuggingFace `datasets`, center-cropped + resized to 64×64. |
| `dit.py` | DiT with `SelfAttention` + `CrossAttention` + `MLP` per block. `TextProjector` (replaces `LabelEmbedder`) and `TextTokenProjector` for the two text-conditioning paths. |
| `flow.py` | Flow matching `fm_q_sample` and `fm_euler_sample`, signature adapted for text inputs. |
| `train.py` | Training loop. Encodes images via VAE and captions via CLIP at each step (pretrained, no_grad), trains DiT on the velocity-prediction objective. |
| `sample.py` | Text → image CLI. Loads everything, runs Euler ODE in latent space, decodes through VAE. Supports `--prompts` for grid generation. |

## Setup

Python 3.9+. From `lab3.2/`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads three models from HuggingFace (cached at `~/.cache/huggingface/`):
- **SD-VAE**: ~335 MB
- **CLIP text encoder**: ~250 MB
- **COCO captions subset**: streamed; ~50 MB for 5K samples

## Train

```bash
python train.py
```

Defaults: 5K COCO pairs at 64×64, 20K training steps, batch size 32, ~2–4 hours on M3 MPS. Saves `model.pt`.

The DiT is small: 8 blocks × 384 hidden × 6 heads ≈ 10M parameters. The pretrained CLIP and VAE add ~120M and ~84M parameters respectively, but those don't get gradient updates — they're forward-only pretrained features.

## Sample

```bash
# Single prompt, 4 generations:
python sample.py --prompt "a cat sitting on a chair"

# Multiple prompts, grid:
python sample.py --prompts "a red car" "a blue truck" "a yellow taxi" \
                 --cfg-scale 4.0

# Production-like settings:
python sample.py --prompt "a dog running on a beach" \
                 --steps 50 --cfg-scale 7.0
```

Output: `generated.png` — a grid of decoded images, rows = prompts, cols = `--n-per-prompt`.

CFG scale ~3–7 is the production sweet spot; the same range used by SD3 / FLUX. `cfg-scale 1.0` is the model's natural conditional behavior; `0.0` ignores text entirely (samples from the marginal image distribution).

## What to expect at this scale

This is a demonstrator, not a good text-to-image model. With 10M DiT params, 5K image-caption pairs, and ~1 hour of M3 training, you should expect:

- **Recognizable objects** for common prompts (`"a cat"`, `"a car"`, `"a tree"`).
- **Color follows captions** somewhat (`"a red car"` → reddish-ish).
- **Composition is poor** — multi-object scenes ("a dog and a cat") often produce a single ambiguous blob.
- **Text legibility is zero** — even production models at our scale struggle.
- **Photographic detail is poor** — 64×64 + tiny model = blurry, posterized output.

That's fine. The lab's goal is to demonstrate the *mechanism* — typing different prompts produces different images, and CFG amplifies the conditioning. The qualitative gap to SD3 / FLUX is purely scale (model size × dataset size × compute), not architecture or recipe.

## Discussion

**Why pretrained components.** Production training pipelines:

```
   Text encoder     SD-VAE              DiT
   ─────────────    ─────────────       ─────────────────
   pretrained,      pretrained,         trained from scratch on
   used as-is       used as-is          (image-latent, text-embedding) pairs
```

Same setup we use here. The DiT learns to navigate the *fixed* latent space defined by SD-VAE, conditioned on the *fixed* text embedding produced by CLIP. Decoupling the modalities is what makes scale-up tractable — you don't have to retrain CLIP or the VAE every time you scale the DiT.

**Why cross-attention.** Pooling the entire prompt to a single vector loses too much information. "A red car" and "a car" produce nearly identical pooled vectors after CLIP's pooler, but they should produce visibly different images. Per-token cross-attention preserves the word-level structure: the model can route the "red" token's information to color decisions and the "car" token's information to shape decisions independently.

This is why every modern text-to-image model uses some form of per-token text conditioning (cross-attention in SD1.x/SDXL, MMDiT joint attention in SD3 / FLUX). Pooled-only conditioning would be a strict downgrade.

**What changes for video (Part 4).** Same recipe. Three pieces extend dimensionally:

- VAE → 3D causal VAE: latent shape becomes `(C, T, H, W)`.
- Patchify → 3D patchify: patch shape `(p_t, p_h, p_w)` instead of `(p_h, p_w)`.
- RoPE → 3D RoPE: positions are `(t, h, w)` instead of `(h, w)`.

Cross-attention to text, AdaLN-Zero, flow matching, CFG — all unchanged. Part 4 reads WAN's code where these dimensional extensions are visible in production.

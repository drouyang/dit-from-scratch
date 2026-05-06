# Module 3.2 — Latent text-to-image DiT

**Goal**: train a real text-to-image DiT — pretrained CLIP text encoder + pretrained SD-VAE + the lab 3.1 DiT architecture (with cross-attention added) — on **`lambdalabs/pokemon-blip-captions`**: 833 Pokemon images with BLIP-generated captions like *"a drawing of a green pokemon with red eyes"*. 833 pairs is tiny by production standards, but the captions are real natural language — diverse color, shape, and feature words across hundreds of Pokemon designs — so the text encoder is doing real work, not class lookup. End-to-end "type a prompt, get an image." This is the SD3 / FLUX recipe at small scale, applied to a laptop-friendly dataset. *(MS-COCO would be the canonical choice, but the popular HF COCO-captions datasets all use script-based loaders that the current `datasets` library refuses; pokemon-blip-captions uses native parquet and works once you authenticate — see the Setup section.)*

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

So each DiT block now has three sublayers instead of two:

```
x = x + gate * self_attn(AdaLN(x, c))           ← image self-attention (lab 3.1)
x = x + cross_attn(LN(x), text_tokens)          ← NEW: image attends to text
x = x + gate * mlp(AdaLN(x, c))                 ← MLP (lab 3.1)
```

Cross-attention is unmodulated (no AdaLN gating around it) — the text tokens already carry conditioning information through their values.

**Why cross-attention.** Pooling the entire prompt to a single vector loses too much information. "A red car" and "a car" produce nearly identical pooled vectors after CLIP's pooler, but they should produce visibly different images. Per-token cross-attention preserves the word-level structure: the model can route the "red" token's information to color decisions and the "car" token's information to shape decisions independently.

This is why every modern text-to-image model uses some form of per-token text conditioning (cross-attention in SD1.x/SDXL, MMDiT joint attention in SD3 / FLUX). Pooled-only conditioning would be a strict downgrade.

### Why frozen pretrained VAE + text encoder

**Latent space is task-agnostic.** SD-VAE's latent encodes natural images; CLIP's text encoder maps natural language into a vision-aligned space (CLIP was trained jointly with an image encoder, so its text vectors already carry visual semantics — that's why it's preferred over a pure language model like BERT for text-to-image). Neither is specific to our text-to-image task — they're general-purpose pretrained backbones, the same reuse pattern vision and NLP have relied on for years. SD-VAE was trained on a curated LAION subset (hundreds of millions of images, perceptual + adversarial losses); CLIP saw 400M (text, image) pairs.

**Decoupling makes scale-up tractable.**

```
   Text encoder     SD-VAE              DiT
   ─────────────    ─────────────       ─────────────────
   pretrained       pretrained          trained from scratch on
   FROZEN           FROZEN              (image-latent, text-embedding) pairs
```

**FROZEN** means the VAE and text encoder run *only* forward — no gradient flows back, no optimizer state, no parameter updates. Their weights stay identical from the first step to the last.

The DiT learns to navigate the fixed latent space defined by SD-VAE, conditioned on the fixed text representation from CLIP. If you trained them jointly, every DiT experiment would also have to retrain the VAE and text encoder — that kills iteration speed. By freezing, every change is local to the DiT.

**Production reality.** SD1.x, SDXL, SD3, FLUX all freeze their VAE and text encoder during DiT training. We're matching the actual recipe.

### Lab vs production scale

Same architecture, very different parameter counts and data:

| | Lab 3.2 (image) | SD3 / FLUX (image) | WAN 2.2 (video) |
|---|---|---|---|
| VAE | SD-VAE-ft-mse — **84M** params, 4-channel image latent | Same SD-VAE family — 84M (sometimes upgraded to 16-channel) | Wan-VAE — 3D causal, **~250M** params, 16- to 48-channel video latent (4× temporal + 8–16× spatial compression) |
| Text encoder | CLIP-base — **120M** params, 512-dim, 77 tokens | CLIP-L + CLIP-G + T5-XXL — together **~12B** params, longer context | umT5-XXL — **~11B** params, multilingual T5-XXL variant |
| DiT | **10M** (8 blocks × 384 hidden) | **2B** (SD3) / **12B** (FLUX) | **5B** (TI2V-5B dense) up to **27B-total / 14B-activated** (A14B MoE) |
| Training data | **833** Pokemon image-caption pairs | hundreds of millions of (image, caption) pairs | hundreds of millions of (video clip, caption) pairs |
| Training compute | ~hours on a laptop | thousands of A100/H100-hours | tens of thousands of H100-hours |

The VAE actually *hasn't* scaled much — 84M params is enough to reconstruct natural images cleanly, so image production keeps it small. Almost all the image-scale-up went into the DiT itself and into a much larger text encoder stack (T5-XXL alone is ~11B params, dwarfing everything else combined). For video, the VAE picks up a temporal dimension (Wan-VAE compresses along time as well as space) and grows ~3×, but the dominant cost is still the DiT — WAN 2.2's flagship A14B is an MoE with 27B total parameters.



## Files

| File | What it is |
| --- | --- |
| `vae.py` | Pretrained SD-VAE (`stabilityai/sd-vae-ft-mse`) wrapper: `.encode(image) → latent`, `.decode(latent) → image`. Includes the standard 0.18215 scale factor. |
| `text_encoder.py` | Pretrained CLIP text encoder (`openai/clip-vit-base-patch32`) wrapper. Returns per-token outputs (for cross-attention), pooled output (for AdaLN), and attention mask. |
| `data.py` | `lambdalabs/pokemon-blip-captions` (833 image-caption pairs) loaded via HuggingFace `datasets`, center-cropped + resized to 64×64. |
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
- **Pokemon BLIP captions** (`lambdalabs/pokemon-blip-captions`): ~85 MB for 833 image-caption pairs (**gated**; see auth steps below)

### HuggingFace authentication

`lambdalabs/pokemon-blip-captions` is a gated dataset on the HuggingFace Hub — the first download requires you to authenticate and accept the dataset's terms. One-time setup:

1. **Get a token.** Create a read-scope token at https://huggingface.co/settings/tokens. Copy the `hf_...` string.
2. **Accept the dataset terms.** Visit https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions while logged in and click "Agree and access repository." Without this step the token alone won't work — Hub gates are per-dataset.
3. **Authenticate locally.** Two equivalent options:

   ```bash
   # Option A — interactive (writes the token to ~/.cache/huggingface/token):
   pip install huggingface_hub
   huggingface-cli login          # paste the token when prompted

   # Option B — env var (good for one-off shells / CI):
   export HF_TOKEN=hf_...         # the token from step 1
   ```

After that, `python train.py` downloads the dataset normally. The two pretrained model checkpoints (SD-VAE, CLIP) aren't gated — they download without auth.

## Train

```bash
python train.py
```

Defaults: 833 Pokemon image-caption pairs at 64×64, 20K training steps, batch size 32, ~2–4 hours on M3 MPS. Saves `model.pt`.

The DiT is small: 8 blocks × 384 hidden × 6 heads ≈ 10M parameters. The pretrained CLIP and VAE add ~120M and ~84M parameters respectively, but those don't get gradient updates — they're forward-only pretrained features.

## Sample

```bash
# Single prompt, 4 generations:
python sample.py --prompt "a green pokemon with red eyes"

# Multiple prompts, grid:
python sample.py --prompts "a fire pokemon" "a water pokemon" "a flying pokemon" \
                 --cfg-scale 4.0

# Production-like settings:
python sample.py --prompt "a blue dragon pokemon with horns" \
                 --steps 50 --cfg-scale 7.0
```

Output: `generated.png` — a grid of decoded images, rows = prompts, cols = `--n-per-prompt`.

CFG scale ~3–7 is the production sweet spot; the same range used by SD3 / FLUX. `cfg-scale 1.0` is the model's natural conditional behavior; `0.0` ignores text entirely (samples from the marginal image distribution).

## What to expect at this scale

This is a demonstrator, not a good text-to-image model. With 10M DiT params, 833 image-caption pairs, and a few hours of M3 training, you should expect:

- **Pokemon-shaped outputs** regardless of prompt — that's all the model has ever seen, so even unrelated prompts get drawn as pokemon-like blobs.
- **Color follows captions** somewhat (`"a red pokemon"` → reddish-ish, `"a blue pokemon"` → blueish).
- **Feature words map roughly** — "with horns", "with wings", "with red eyes" can shift the silhouette in the expected direction at high CFG.
- **Composition is poor** — multi-feature prompts ("a green dragon with red eyes and horns") often blur together.
- **Photographic detail is poor** — 64×64 + tiny model = blurry, posterized output.

That's fine. The lab's goal is to demonstrate the *mechanism* — typing different prompts produces different images, and CFG amplifies the conditioning. The qualitative gap to SD3 / FLUX is purely scale (model size × dataset size × compute), not architecture or recipe.

## Discussion

**What changes for video (Part 4).** Same recipe. Three pieces extend dimensionally:

- VAE → 3D causal VAE: latent shape becomes `(C, T, H, W)`.
- Patchify → 3D patchify: patch shape `(p_t, p_h, p_w)` instead of `(p_h, p_w)`.
- RoPE → 3D RoPE: positions are `(t, h, w)` instead of `(h, w)`.

Cross-attention to text, AdaLN-Zero, flow matching, CFG — all unchanged. Part 4 reads WAN's code where these dimensional extensions are visible in production.

# Module 4.3 — Post-training — LoRA

**Goal**: take WAN-2.1 T2V-1.3B (the smallest variant of the Wan-Video team's open production text-to-video DiT family) and *post-train* it. Hands-on: LoRA fine-tune it on a tiny custom video-caption set so the model adapts to a new style or concept. Full SFT — every parameter trainable — is its own hands-on lab (lab 4.4).

**Why this matters for DiT**: post-training is what *every* production model goes through after the initial pretraining run. Pretraining gets you "knows how to make video"; post-training gets you "makes video the way *you* want." For a small team or solo developer, post-training is also the *only* place you have leverage — pretraining a 14B-parameter video model is a $1M+ cluster job, but a LoRA fine-tune of a 1.3B variant fits in $20 of rented H100 time. This lab teaches the cheap, ubiquitous version (LoRA) end-to-end and orients you on the rest of the landscape.

**Why WAN**: Wan-2.1 / Wan-2.2 is the strongest open text-to-video DiT family at the time of writing (matches or beats closed Sora-class models on several benchmarks, fully open weights, supported in `diffusers`). T2V-1.3B is the smallest variant — 1.3 B parameters, fits at bf16 on a single 24 GB consumer GPU for inference and on a 40–80 GB H100 for LoRA training.

## The post-training landscape

Pretraining sets up the model's broad capabilities (compresses billions of (caption, video) pairs into a velocity field over latent space). Post-training adapts it. The four flavors you'll run into:

```
                        what changes      base weights      typical compute
                        per step          modified?         budget
─────────────────       ──────────────    ───────────────   ───────────────
LoRA fine-tune          ~10 M LoRA params  no, frozen        $10–50
Full SFT                ~1.3 B base params yes, all          $50–500 (1.3B on 4090)
                                                             $1k–10k+ (production scale)
```

This lab does LoRA hands-on; lab 4.4 does full SFT hands-on.

## What LoRA actually is

LoRA (Low-Rank Adaptation, Hu et al. 2021) is the dominant parameter-efficient fine-tuning method. The mechanic, in one paragraph:

For every linear layer `y = W x` in the base model that you want to adapt, *don't* modify `W`. Instead, add a parallel learned correction:

```
y  =  W x  +  (α / r) · B A x
            └── frozen ──┘  └── trainable ──┘

where  W ∈ R^(d_out × d_in)   ← original weight, frozen
       A ∈ R^(r × d_in)        ← "down" projection
       B ∈ R^(d_out × r)        ← "up" projection
       r  = rank, e.g. 8 or 16  ← much smaller than d_out, d_in
       α  = scaling factor
```

`B A` is at most rank-`r`, so the correction lives on a tiny low-rank subspace. The whole adapter contributes `r · (d_in + d_out)` new parameters per linear, vs `d_in · d_out` for full fine-tuning — typically **0.1–1% of the base model's parameter count**. At bf16, a rank-16 LoRA on Wan-2.1 T2V-1.3B is ~50–150 MB; full bf16 weights are ~2.6 GB. That's the entire pitch.

Why it works:

1. **Empirically, fine-tuning has low intrinsic rank.** When you fully fine-tune a pretrained model on a downstream task, the weight delta `ΔW = W_finetuned − W_pretrained` turns out to be approximately low-rank — most directions of change matter, but not many. LoRA pre-commits to "use only the top r singular directions" and you lose surprisingly little quality.
2. **Stackable / swappable.** Because the base is untouched, you can keep multiple LoRAs around (one per character, one per style, one per camera-angle preference) and merge them at inference. Civitai is essentially a marketplace of these.
3. **Cheap to ship.** A 100 MB file vs a 2.6 GB checkpoint changes how distribution works. Authors share LoRAs the way they used to share image filters.

### Where LoRAs go in WAN's transformer

WAN's transformer block looks like (simplified):

```
x ─┬─► AdaLN(c) ─► attention(q,k,v,o) ─► gate(c) ─┐
   │                                              ▼
   └────────────────────────────────────────────► ⊕ ─┐
                                                     ▼
   ┌────────────────────────────────────────────► ⊕ ◄── gate(c) ◄── ffn(gate, up, down)
   │                                              ▲
   ▲ ─── AdaLN(c) ─────────────────────────────── ┘
```

The canonical LoRA targets (matching `LORA_TARGET_MODULES` in `train_lora.py`):

- **Attention's `to_q`, `to_k`, `to_v`, `to_out.0`** — these are the routing matrices. Adapting them changes *what attends to what*. Most LoRAs concentrate here because attention dominates how the model uses spatial / temporal context.
- **MLP's `ffn.net.0.proj` (gate+up) and `ffn.net.2` (down)** — adapting these changes the per-token nonlinear transform. Usually included for completeness.

Things you don't typically wrap with LoRA:
- LayerNorm / AdaLN — small, already very task-specific in their gain/bias.
- VAE / text-encoder weights — frozen by design (lab 3.2 covers why).
- The 3D RoPE frequency tables — fixed positional construction, no params.

## Hands-on: train a LoRA on Wan-2.1 T2V-1.3B

### Hardware

Part 4's compute target is a **4× 4090 server**. The per-GPU config is sized to fit a single 4090 (24 GB) at 256 × 256 × 17 frames with gradient checkpointing — DDP across the 4 cards just divides wall clock.

| GPU | What fits | Wall clock @ 2000 steps |
|---|---|---|
| **4× 4090 24 GB** (Part 4 default, DDP) | 256×256 × 17 frames per GPU, grad-accum 8, GC on; effective batch = 32 | **~1.5–2 hours** |
| **1× 4090 24 GB** (single-card fallback) | same per-GPU config; effective batch = 8 | ~6–8 hours |
| A100 40 GB (1×) | 384×384 × 17 frames, batch 1, grad-accum 8 | ~3–4 hours |
| A100 80 GB (1×) | 480p × 25 frames, batch 1, grad-accum 4, can drop GC | ~2–3 hours |
| H100 80 GB (1×) | 480p × 33 frames, batch 1, grad-accum 4, can drop GC | ~1.5–2.5 hours |
| MacBook M3 | inference only — *don't* try to LoRA-train a 1.3B video DiT on MPS | — |

**Default config rationale.** 17 = 4·4 + 1 lines up with Wan-VAE's 4× temporal compression (it expects an "anchor frame" plus multiples of 4). 256×256 lines up with WAN's 8× spatial compression. Gradient checkpointing trades a ~30% wall-clock hit for the activation memory needed to fit the 24GB budget.

**Going multi-GPU.** `accelerate launch` automatically uses every visible GPU via DDP — no code change needed. On 4× 4090, effective batch size = `1 (batch) × 8 (grad-accum) × 4 (GPUs) = 32`. Wall-clock divides almost linearly.

**Easiest path**: rent a 4× 4090 server from Lambda / Vast / RunPod (~$1.50–3/hr for the full server) for an afternoon, or fall back to a single 4090 (~$0.40–0.80/hr) overnight if 4× isn't available.

### Setup

```bash
cd lab4.3/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First run will download Wan-2.1 T2V-1.3B from HuggingFace (~5 GB total: VAE + text encoder + transformer). The repo is public — no auth required.

### Prepare your data

Pick a *concept* you want WAN to learn. Common LoRA targets:

- **Style** — *"in the style of Studio Ghibli"*, *"watercolor anime"*, *"vintage 8mm film"*. Train on 30–80 clips that share the visual style.
- **Character** — a specific person, animal, or fictional creature. Train on 20–50 clips of them in different poses / scenes.
- **Camera technique** — *"slow dolly zoom"*, *"first-person motorcycle riding"*. Train on clips that demonstrate the camera move.

Layout:

```
data/
├── captions.json       # [{"video": "clips/cat_01.mp4", "caption": "..."}, ...]
└── clips/
    ├── cat_01.mp4
    ├── cat_02.mp4
    └── ...             # 20–100 short clips, ~2–5 seconds each
```

Captions matter a lot. Write them in WAN's preferred caption style — descriptive, including subject + action + environment + style. Example: *"an orange tabby cat lounges on a brown leather sofa, golden afternoon light streaming through a window, cinematic shallow depth of field"*. If your dataset is style-focused, include the style name as a *trigger word* the user will type at sample time (e.g., *"in mystyle, an orange tabby cat ..."*).

### Train

```bash
accelerate launch train_lora.py \
    --data-root data/ \
    --output-dir runs/my-style/ \
    --rank 16 \
    --steps 2000
```

Hyperparameters worth understanding before you tune them:

| Knob | Default | What it controls |
|---|---|---|
| `--rank` | 16 | Capacity of the adapter. Higher = more parameters = can capture richer behavior, but slower to train and more prone to overfitting. Standard range: 4 (subtle), 16 (default), 64 (heavy). |
| `--alpha` | 16 | Scaling factor; effective contribution = `α/r`. Setting `alpha == rank` keeps the scale at 1.0. Some authors use `alpha = 2*rank` for more aggressive adaptation. |
| `--lr` | 1e-4 | Learning rate for the LoRA params. ~10× higher than what you'd use for full fine-tuning (LoRA params are randomly initialized and small). |
| `--steps` | 2000 | Total optimization steps. Style LoRAs converge in 1k–3k; character LoRAs need 2k–5k. Watch the loss curve — usually plateaus before overfitting kicks in. |
| `--n-frames`, `--height`, `--width` | 17, 256, 256 | Per-clip dimensions. Sized for 4090; raise on bigger GPUs (see the Hardware table). Lower spatial resolution hurts quality more than fewer frames; if you have headroom, prefer raising `--height` / `--width` over `--n-frames`. |
| `--gradient-checkpointing` | on | Trades ~30% wall clock for activation memory; required to fit a 4090. Pass `--no-gradient-checkpointing` to disable on H100 / A100. |
| `--grad-accum` | 8 | Effective batch size = `batch_size * grad_accum * num_gpus`. On 1× 4090 that's 8; on 4× 4090 that's 32. Higher = smoother gradient but more wall-clock. |

You'll see logs like:

```
trainable params: 5.74M  (0.44% of 1.30B base)
step     1  loss 0.4218
step    50  loss 0.3104
step   100  loss 0.2812
...
step  2000  loss 0.1954
saved runs/my-style/lora_step02000.safetensors  (288 tensors, 138.6 MB at bf16)
```

The loss number is meaningless in absolute terms (depends on your data's velocity magnitudes); what matters is that it's monotonically decreasing.

### Sample

After training, generate side-by-side with the base model to actually see what your LoRA learned:

```bash
# With your trained LoRA:
python sample_lora.py \
    --prompt "in mystyle, an orange tabby cat on a sofa" \
    --lora runs/my-style/lora_step02000.safetensors \
    --out my_lora.mp4

# Base WAN, same prompt and seed:
python sample_lora.py \
    --prompt "in mystyle, an orange tabby cat on a sofa" \
    --out base.mp4
```

`--lora-scale` is your most useful knob at sample time — it multiplies the LoRA's contribution. Try `0.0` (== base), `0.5` (subtle), `1.0` (trained), `1.5` (exaggerated). Different LoRAs have different sweet spots.

### What to expect

- **Style LoRAs converge fastest.** A coherent visual style across 30 clips, 1500 steps, rank 8–16 — usually visible after the first checkpoint.
- **Character LoRAs are harder.** Identity-preservation across poses requires more clips and higher rank. Common failure mode: the model picks up the *outfit* the character was wearing in your clips and renders that on everything.
- **Loss going down ≠ outputs improving.** Eyeball the samples every few hundred steps. If they stop changing or start drifting, you've converged or overfit.
- **The base was trained on hundreds of millions of clips; you have 50.** Your LoRA can *bend* the model's behavior, not retrain it. Don't expect it to learn truly new concepts (a Pokemon WAN has never seen) from a tiny dataset.

## Survey: full SFT and continued pretraining

LoRA is post-training-on-a-budget. Here's what the rest of the field looks like, and when you'd reach for it.

### Full supervised fine-tuning (SFT)

Same loss as LoRA training (flow-matching MSE on velocity), but **all** base parameters are unfrozen and updated.

```
                  LoRA fine-tune        Full SFT
                  ─────────────         ─────────
trainable params  ~5–20 M               ~1.3 B (Wan-1.3B)
                                        ~14 B (Wan-2.2 A14B)
optimizer state   small                 huge (~10 GB fp32 for 1.3 B; 8-bit Adam helps)
hardware          1× 4090 24 GB         1× 4090 with 8-bit Adam + GC (1.3 B);
                                        8× H100 with FSDP for A14B
artifact size     50–200 MB             5+ GB per checkpoint
typical use       style / character     production model variants
                  / fan ports           ("WAN-anime", "WAN-realistic")
```

Why production models do full SFT instead of LoRA: the rank-`r` constraint ultimately caps how much the base can change. For a major capability shift (re-aiming the model at a different domain, or substantially upgrading quality), you need the full parameter space. Stability-AI's various SD3 variants, Black-Forest-Labs's FLUX-dev → FLUX-pro, Wan-Lightning all involve full SFT on top of pretrained checkpoints.

For hands-on full SFT of WAN-2.1 T2V-1.3B on a 4090 (using 8-bit AdamW + gradient checkpointing to fit), see **lab 4.4**. The diffusers reference is `diffusers/examples/text_to_video/train_text_to_video_lora.py` (LoRA) and `train_text_to_video.py` (full SFT) — same code shape, just `unet.requires_grad_(False)` becomes `unet.requires_grad_(True)`.

### Continued pretraining

Re-run *pretraining-style* training (huge data, low learning rate, full parameters) on a domain-specific corpus to shift the model's distribution before any task-specific fine-tune. Most people don't have the data or compute for this.

## Files

| File | What it is |
| --- | --- |
| `data.py` | `VideoCaptionDataset` — reads `captions.json` + `.mp4` clips, decodes with `decord`, normalizes to `[-1, 1]`. |
| `train_lora.py` | LoRA training loop. Loads Wan-2.1 T2V-1.3B, applies a `peft` LoRA config, trains only the adapter weights with flow-matching MSE. |
| `sample_lora.py` | Inference CLI. Loads WAN base, optionally applies a trained LoRA, generates an `.mp4`. |
| `requirements.txt` | `diffusers`, `peft`, `accelerate`, `transformers`, `decord`, `safetensors`. |

## Discussion

### When to use which post-training

```
Goal                            Reach for
────────────────────            ──────────────────────
add a style / character         LoRA  (this lab)
fix a model behavior            LoRA  (often) or full SFT
ship a production variant       full SFT  (lab 4.4)
domain-shift the model          continued pretraining
```

LoRA covers more of the practical "I want to ship something" cases than its parameter budget would suggest. The rest are reached for when LoRA's capacity ceiling is the bottleneck.

### What changes vs lab 3.2

Architecturally, almost nothing — WAN is a DiT, the loss is flow matching, the conditioning is text via cross-attention plus AdaLN-Zero modulation. The pieces you built up to lab 3.2 all map directly onto blocks inside WAN.

What's new is *practical*:

- **Production codebases are big.** `diffusers`' Wan implementation is thousands of lines. You don't read all of it; you read the bits that matter (the transformer block, the pipeline forward, the loss).
- **Hyperparameter culture.** rank, alpha, learning rate, target modules — there's a community norm for each, and you mostly want to follow it before innovating.
- **Distribution is the second half.** Training a LoRA is half the work; making it usable for someone else (uploading the adapter, providing trigger words, packaging it into a ComfyUI custom node) is the other half. That's lab 4.5.

### Where to go deeper

- [LoRA paper (Hu et al. 2021)](https://arxiv.org/abs/2106.09685) — the original. Sections 4.1 and 4.2 establish the low-intrinsic-rank claim empirically; the rest is application to GPT-3.
- [`peft` library docs](https://huggingface.co/docs/peft) — beyond LoRA: prefix tuning, IA³, AdaLoRA. Same wrapping pattern, different parametric form.
- [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) — the official training and inference repo. Their `train.py` is a good reference for production-grade training infrastructure (FSDP, DeepSpeed configs, multi-resolution training).
- [`musubi-tuner`](https://github.com/kohya-ss/musubi-tuner) — the de facto community LoRA trainer for WAN / Hunyuan video models. More features than this lab's `train_lora.py`, more knobs, more complex.

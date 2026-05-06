# Module 4.4 — Post-training — Full SFT

**Goal**: take WAN-2.1 T2V-1.3B and **fully** fine-tune it — every parameter trainable, no LoRA adapter — on a tiny custom video-caption set. Trade-offs vs lab 4.3's LoRA:

|                            | LoRA (lab 4.3) | Full SFT (this lab)         |
| -------------------------- | -------------- | --------------------------- |
| trainable params           | ~5 M (rank 16) | ~1.3 B                      |
| optimizer state            | ~40 MB         | ~2.6 GB (8-bit Adam)        |
| artifact size              | ~140 MB        | ~2.6 GB                     |
| typical wall clock         | 6–8 h on 4090  | 10–14 h on 4090             |
| overfits at                | ~5k steps      | ~500 steps on a tiny set    |
| quality ceiling            | rank-r capped  | unbounded (in principle)    |
| how production ships       | community LoRAs (Civitai) | model variants (FLUX-pro, Wan-Lightning, "WAN-anime") |

**When to reach for full SFT vs LoRA**: LoRA covers ~90% of practical post-training (style, character, small behavior shifts). You reach for full SFT when LoRA's rank-r constraint is the bottleneck — major capability shifts, re-aiming the model at a different domain, or shipping a "production variant" of the base. Stability-AI's SD3 → SD3.5, Black-Forest-Labs's FLUX-dev → FLUX-pro, and Wan-Lightning all involve full SFT on top of pretrained checkpoints.

**Why this lab teaches the technique even though LoRA is more common**: the *memory machinery* you need for 1.3B-param SFT (8-bit AdamW, gradient checkpointing, mixed precision, FSDP at scale) generalizes directly to any other "fit a model bigger than your VRAM" problem. The flow-matching forward pass is identical to lab 4.3 — what's new is everything around it.

## What's actually different vs lab 4.3

The forward pass and loss are **identical**:

```python
# lab 4.3 and lab 4.4 — same lines, same math
z_0 = vae.encode(videos).latent_dist.sample()
text_embeds = text_encoder(captions).last_hidden_state
noise = torch.randn_like(z_0)
t = torch.rand(...)
z_t = (1 - t) * z_0 + t * noise
pred = transformer(z_t, t, text_embeds, ...)
loss = MSE(pred, noise - z_0)
```

What changes is the wrapping:

| | lab 4.3 (LoRA) | lab 4.4 (full SFT) |
|---|---|---|
| `transformer.requires_grad_(...)` | `False`, then add LoRA adapter | `True` |
| `peft.LoraConfig` + `add_adapter()` | yes | **no** |
| optimizer | `torch.optim.AdamW` (over ~5M params) | `bitsandbytes.optim.AdamW8bit` (over 1.3B params) |
| learning rate | 1e-4 | **1e-5** (10× lower — more on this below) |
| save | LoRA delta only (~140 MB) | full transformer state dict (~2.6 GB) |

**Why the LR drops 10×.** LoRA adapters are randomly initialized; their weights start at zero contribution and need a hot LR to warm up. Full SFT updates *pretrained* weights — they already encode hundreds of millions of training steps' worth of structure. A high LR overwrites that structure (catastrophic forgetting). Production SFT runs use 1e-5 to 5e-5.

## Memory math

This is the meat of why the lab exists. The fp32 AdamW state for 1.3B params is the bottleneck.

### Where the bytes go (1.3B params, bf16 forward, single 4090)

|                                        | fp32 Adam     | **8-bit Adam** (this lab) |
| -------------------------------------- | ------------- | ------------------------- |
| params (bf16)                          | 2.6 GB        | 2.6 GB                    |
| gradients (bf16)                       | 2.6 GB        | 2.6 GB                    |
| optimizer state (m + v)                | **10.4 GB**   | **2.6 GB**                |
| static total                           | 15.6 GB       | 7.8 GB                    |
| activations @ 256² × 17 frames, GC on  | ~5–8 GB       | ~5–8 GB                   |
| **VRAM total**                         | **~22 GB**    | **~14 GB**                |

The fp32 column technically fits a 4090 with no headroom. In practice you also want headroom for fragmentation, kernel workspaces, and the larger latent shapes you'll want to try later — so 8-bit Adam is what makes the lab actually work.

### The three memory tricks

1. **8-bit AdamW** (`bitsandbytes`). Adam stores per-parameter `m` (1st moment) and `v` (2nd moment). Default: both fp32, total 8 bytes/param. 8-bit Adam quantizes both to int8 with block-wise scales, getting it down to ~2 bytes/param. <1% quality cost in published comparisons. **The single most important line in the training script.**

2. **Gradient checkpointing** (`transformer.enable_gradient_checkpointing()`). Activation memory at 256² × 17 frames otherwise blows past 20 GB. Checkpointing trades ~30% wall-clock for activation recompute on the backward pass, capping activation memory at the per-block cost.

3. **Pure bf16, no fp32 master copy.** Some production setups keep fp32 master weights for numerical stability (LossScaler tradition); we skip them to save the 5.2 GB. Empirically bf16-only is fine for SFT in our scale; full pretraining sometimes needs the fp32 copy.

## Hardware

Same shape as lab 4.3's table — full SFT just bumps the per-step cost. Part 4's default is **4× 4090**.

| GPU | What fits | Wall clock @ 2000 steps |
|---|---|---|
| **4× 4090 24 GB** (Part 4 default, DDP) | 256² × 17 frames per GPU, grad-accum 8, GC on, 8-bit Adam; effective batch = 32 | **~3–4 hours** |
| **1× 4090 24 GB** (single-card fallback) | same per-GPU config; effective batch = 8 | ~10–14 hours |
| A100 40 GB (1×) | 384² × 17 frames; can use fp32 Adam | ~5–7 hours |
| A100 80 GB (1×) | 480p × 25 frames; fp32 Adam, no GC | ~4–5 hours |
| H100 80 GB (1×) | 480p × 33 frames; fp32 Adam, no GC | ~3–4 hours |

For this scale of SFT (1.3B model), DDP is the right scaling shape — every GPU keeps a full model copy. **FSDP** (sharding params/grads/optim across GPUs) is what you reach for when even one model copy doesn't fit per GPU; for 1.3B at bf16 + 8-bit Adam that's not the case yet, but it kicks in for 14B (Wan-2.2 A14B) full SFT.

## Setup

```bash
cd lab4.4/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`bitsandbytes` requires CUDA. M3 / CPU won't work for this lab — see lab 4.1's compute notes.

## Prepare your data

Same layout as lab 4.3:

```
data/
├── captions.json    # [{"video": "clips/cat_01.mp4", "caption": "..."}, ...]
└── clips/
    ├── cat_01.mp4
    └── ...          # 20–100 short clips, ~2–5 seconds each
```

**Caveat for full SFT specifically**: small datasets *destroy* full SFT runs much faster than they destroy LoRA. With 50 clips and 1.3B trainable parameters, the model has more than enough capacity to memorize your dataset perfectly in ~500 steps. The reasonable working ranges:

- **Tiny dataset (20–100 clips)**: stop at 200–500 steps, sample frequently, watch for over-fitting (samples lose diversity, snap to specific clip artifacts).
- **Medium dataset (500–5k clips)**: 1k–3k steps is the typical sweet spot.
- **Large dataset (50k+ clips)**: full SFT starts to look like continued pretraining; the "production model variant" recipe lives here.

If you only have 50 clips, use lab 4.3's LoRA — that's the right tool. This lab exists to teach the *technique*; production results need production data.

## Train

```bash
accelerate launch train_sft.py \
    --data-root data/ \
    --output-dir runs/my-sft/ \
    --steps 2000
```

Logs:

```
trainable params: 1.30B  (full transformer)
step     1  loss 0.4218
step    50  loss 0.3104
...
step   500  loss 0.2244
saved runs/my-sft/transformer_step00500.safetensors  (288 tensors, 2.62 GB)
```

Hyperparameters worth understanding:

| Knob                          | Default       | What it controls                                                                                                                                                  |
| ----------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--lr`                        | 1e-5          | 10× lower than LoRA. Higher than this and you'll catastrophically forget the base.                                                                                |
| `--steps`                     | 2000          | For tiny datasets, sample much earlier (every 100–200 steps) and stop when samples start to over-fit.                                                              |
| `--n-frames`, `--height`, `--width` | 17, 256, 256 | Sized for 4090. Keep low until you confirm the run is stable; raise on bigger GPUs.                                                                              |
| `--gradient-checkpointing`    | on            | Required for the 4090 path. Disable on H100/A100 if you have headroom — gains you ~30% wall clock.                                                              |
| `--grad-accum`                | 8             | Effective batch = `batch * grad_accum * num_gpus`. 1× 4090 → 8; 4× 4090 → 32.                                                                                    |

## Sample

```bash
# SFT'd model:
python sample_sft.py \
    --prompt "an orange tabby cat lounging on a brown leather sofa, golden hour" \
    --ckpt runs/my-sft/transformer_step02000.safetensors \
    --out sft.mp4

# Same prompt + seed, base WAN for comparison:
python sample_sft.py \
    --prompt "an orange tabby cat lounging on a brown leather sofa, golden hour" \
    --out base.mp4
```

If your dataset and steps were chosen well, the SFT'd output should reflect the dataset's bias (style / subject / camera). If they weren't, both videos look similar — or worse, the SFT'd one is over-fit to a single training clip.

## Discussion

### When you'd actually do this in production

Real production SFT runs look like:

- **Dataset size**: 100k–10M clips with rich captions.
- **Model size**: 1B–14B parameters (Wan-1.3B / Wan-A14B).
- **Compute**: 8–256× H100 with FSDP for the bigger variants.
- **Duration**: days to weeks.
- **LR schedule**: cosine decay, warmup, sometimes a slow LR ramp.
- **Output**: a checkpoint that *replaces* the base for downstream LoRA fine-tuning.

This lab is the *minimum viable* version of that recipe — same machinery, much smaller scale.

### Where to go after SFT

The minimum-viable post-training chain looks like:

```
pretrained base (lab 3.2 vibes, 100M+ clips)
       │
       ▼
   full SFT (this lab) — domain or quality shift
       │
       ▼
   ship: HF model card + ComfyUI workflow + hosted endpoint (lab 4.5)
```

Big production runs add more steps on top, but those are out of scope for this curriculum — they need different infrastructure and are reached for after a model is already shipped with a specific quality or cost complaint.

### Where to read

- **8-bit AdamW**: [Dettmers et al. 2022](https://arxiv.org/abs/2110.02861). The original 8-bit optimizers paper.
- **Gradient checkpointing**: [Chen et al. 2016](https://arxiv.org/abs/1604.06174). Older trick, still ubiquitous.
- **FSDP (when you scale past one GPU's worth of model)**: [Zhao et al. 2023](https://arxiv.org/abs/2304.11277), and HuggingFace's `accelerate` FSDP integration docs.

# Module 4.2 — Inference acceleration

**Goal**: take WAN inference from "it works" (lab 4.1) to "it's fast." Read what production *inference* engines actually do — kernel fusion, attention backend selection, feature caching, parallelism — by mapping each technique to (1) the upstream library that implements it, (2) which lab in this curriculum introduced the underlying concept, and (3) which `sglang` CLI flag turns it on. Run a benchmark that compares stock `diffusers` vs SGLang-Diffusion on the same prompt.

This lab is **inference-only**. Every technique below targets generation latency on a fixed model. Acceleration techniques that overlap with *training* (FlashAttention, mixed precision, sequence parallelism, FSDP) live in lab 4.3 / 4.5.

## Why a separate lab from 4.1

Lab 4.1 used `WanPipeline.from_pretrained(...).__call__(...)` directly. That's the readable, hackable, ~30-second-load reference path. It's also slow:

- Every `Linear → activation → Linear` runs as separate CUDA kernels — kernel-launch overhead dominates at small token counts.
- Attention runs on whatever PyTorch picks (`scaled_dot_product_attention` → cuDNN or FlashAttention 2 if compiled-in). No SageAttention, no quantized attention, no diffusion-specific kernels.
- No caching of redundant DiT computation across diffusion steps.
- Single-GPU only — no sequence or tensor parallelism.

A production inference engine fixes all of those. SGLang-Diffusion is the canonical open-source one (sister project to SGLang for LLMs). The blog post claims up to 5× speedup over baseline; the techniques below are why.

## The technique map

Every optimization in SGLang-Diffusion comes from somewhere. Most are upstream libraries; SGLang-Diffusion is the orchestration layer that picks and composes them.

| Technique | Where it comes from | What it does | Maps back to |
|---|---|---|---|
| **FlashAttention 2/3** | [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) — external | Tiled softmax over Q@Kᵀ that never materializes the full attention matrix. Core LLM-serving primitive. | lab 1.3 (attention math) |
| **SageAttention 2/3** | [`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention) — external | Quantized attention (INT8 / FP8 Q/K/V). Cuts attention FLOPs and memory at the cost of <1% quality. | lab 1.3 (attention math) |
| **FlashInfer RoPE** | [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) — external | Inplace, fused rotary embedding. Replaces ~5 PyTorch ops with one kernel. | lab 3.1 (RoPE-2D) |
| **Fused QKV** | model-adapter pattern, not a library | One `Linear(hidden, 3·hidden)` + split, instead of three `Linear`s. | lab 1.3 (Q/K/V projection) |
| **Fused gate+up+SiLU** (SwiGLU) | usually `flashinfer.silu_and_mul` | One kernel for `silu(gate(x)) * up(x)`. | lab 1.4 (FFN block) |
| **JIT QK-norm kernel** | in-house Triton / `torch.compile` | Fuses the per-head Q/K RMSNorm that some DiT variants use. | lab 1.3 (norm + attention) |
| **Custom timestep CUDA kernel** | in-house | Sinusoidal `t` embedding written as one CUDA kernel instead of many tiny PyTorch ops. | lab 1.1 (sinusoidal embed) |
| **Cache-DiT** | [`vipshop/cache-dit`](https://github.com/vipshop/cache-dit) — external | DiT-specific feature caching: skip recomputing block outputs that haven't changed across diffusion steps. ~1.7× alone in the SGLang-Diffusion blog. | new — diffusion-specific |
| **Layerwise weight offload** | in-house orchestration | Prefetch layer N+1 onto GPU during layer N's compute. Hides PCIe transfer cost. | new — production-only |
| **Sequence parallelism** (Ulysses + Ring USP) | [`xfuser` / xDiT](https://github.com/xdit-project/xDiT) — external (DeepSpeed-Ulysses lineage) | Shard the long sequence (long video → many tokens) across GPUs along the sequence dim. | new — covered conceptually here |
| **Tensor parallelism** | standard | Shard linear-layer weights across GPUs. | new — covered conceptually here |

The pattern: SGLang-Diffusion's *original code* is the runtime that schedules these kernels and applies offload / parallelism plans. Each named library can be A/B-tested by toggling its flag.

## Toggle map (the SGLang-Diffusion flags)

What makes SGLang-Diffusion teachable: every optimization above is independently turn-on-able from the CLI. From the [official CLI docs](https://sgl-project.github.io/diffusion/api/cli.html):

| Optimization | Flag |
|---|---|
| Cache-DiT | `--cache-dit-config <path>` (or `SGLANG_CACHE_DIT_ENABLED=true`) |
| Attention backend | `--attention-backend {fa,sage,xformers,native,_flash_3_hub}` |
| Layerwise weight offload | `--dit-layerwise-offload true` |
| Sequence parallelism | `--sp-degree N`, `--ulysses-degree N`, `--ring-degree N` |
| Tensor parallelism | `--tp-size N` |
| VAE memory | `--vae-tiling`, `--vae-slicing` |
| Text-encoder offload | `--text-encoder-cpu-offload`, `--pin-cpu-memory` |

Useful learning exercise: pick one flag, run the same prompt with and without it, compare wall clock. The `benchmark.py` in this lab is a starting point.

## Compute reality

CUDA-required for the meaningful optimizations. Apple MPS works for SGLang-Diffusion via a separate install path (`uv pip install -e "python[all_mps]"`), but most kernel-level speedups are CUDA-only. WAN 2.1 T2V-1.3B fits a 4090.

| Hardware | What works |
|---|---|
| **1× 4090 / 4080 / 3090** (≥12 GB VRAM) | All single-GPU paths (FlashAttention, SageAttention, Cache-DiT, layerwise offload). The realistic target. |
| **Multi-4090 / multi-A100** | Adds sequence + tensor parallelism. Useful when you push to higher resolutions / longer clips. |
| **B200 / H200** | Required for WAN 2.2-A14B (the MoE flagship). T2V-1.3B doesn't need it. |
| **Apple M3 / MPS** | Stock diffusers works (slowly). SGLang-Diffusion's MPS backend exists but is the courtesy backend — most kernels won't apply. |

## Files

| File | What it is |
|---|---|
| `benchmark.py` | Same prompt / seed / steps through stock `diffusers.WanPipeline` (Python) and `sglang generate` (subprocess). Reports wall clock + peak VRAM. |
| `requirements.txt` | `diffusers>=0.36`, plus a comment with the `uv pip install "sglang[diffusion]" --prerelease=allow` invocation. |

## Setup

```bash
cd lab4.2/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# SGLang-Diffusion is installed separately because it requires uv + prerelease pins:
pip install --upgrade uv
uv pip install "sglang[diffusion]" --prerelease=allow
```

Verify SGLang is on PATH:

```bash
sglang generate --help    # should print the CLI flags listed in the toggle map
```

## Run the benchmark

Generate the same 3-second clip via both backends:

```bash
python benchmark.py --prompt "a fluffy red panda eating bamboo on a tree branch"
```

Output:

```
prompt: 'a fluffy red panda eating bamboo on a tree branch'
shape:  49 frames @ 832×480,  steps=30,  cfg=5.0,  seed=42

=== diffusers ===
  load:        45.2s
  generate:   220.4s
  peak VRAM:  10.78 GB
  saved out_diffusers.mp4

=== sglang ===
  $ sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers ...
  wall:       82.1s  (includes load + generate, can't separate)
  saved out_sglang.mp4

=== comparison ===
backend                    wall    peak VRAM
--------------------------------------------
diffusers                 220.4s     10.78 GB
sglang-diffusion           82.1s    (see sglang logs)

speedup: 2.69× (sglang vs diffusers, end-to-end)
```

(Numbers will vary by GPU and SGLang version. The shape — single-digit-multiple speedup with the same prompt + seed + steps — is what you're verifying.)

## Suggested A/B exercises

After the basic benchmark works, the lab's deeper exercise is *which optimization buys you what*. Run `sglang generate` with each flag toggled in isolation:

```bash
# Baseline (FlashAttention default):
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "..." \
    --attention-backend fa --save-output run_fa.mp4

# Switch to SageAttention (quantized):
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "..." \
    --attention-backend sage --save-output run_sage.mp4

# Add Cache-DiT:
SGLANG_CACHE_DIT_ENABLED=true sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "..." --attention-backend fa --save-output run_fa_cachedit.mp4

# Layerwise offload (helpful on 12 GB cards):
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "..." \
    --dit-layerwise-offload true --save-output run_offload.mp4
```

Each run gives you a wall-clock number. Build a small table; the per-technique deltas are roughly what the SGLang-Diffusion blog reports.

## Discussion

### Training vs inference

Every technique in this lab is for inference. The training-relevant subset:

- **FlashAttention** — works in training too (and `WanPipeline`'s training flavor in `diffusers` already uses it under the hood when available).
- **Mixed precision (bf16)** — training, yes (lab 4.3 / 4.5 already use it).
- **Sequence parallelism** — works in training too via `xfuser` / DeepSpeed-Ulysses; it's used to train long-video models that don't fit on one GPU.
- **Cache-DiT** — *inference-only*. Caching only works when steps converge to similar outputs, which doesn't apply during training where every step changes the loss landscape.
- **Quantized attention (SageAttention)** — inference-only at production quality. Quantized training is its own research area (FP8 training is reaching production but mostly for LLMs).

### What SGLang-Diffusion isn't

- **Not a model**. It's an inference engine. The model weights are still WAN's.
- **Not a training framework**. For training, lab 4.3 (LoRA) and 4.5 (full SFT) use `accelerate` + `bitsandbytes` + `peft` directly. Some kernels overlap; the orchestration is different.
- **Not the only option**. xfuser / xDiT focuses on diffusion-specific parallelism; ComfyUI has its own scheduler; private companies run on TensorRT-LLM-Diffusion variants. SGLang-Diffusion is the most production-shaped open-source one as of mid-2026.

### Where to go deeper

- [LMSYS SGLang-Diffusion blog](https://www.lmsys.org/blog/2025-11-07-sglang-diffusion/) — original release post; the optimization list and benchmark plots.
- [SGLang-Diffusion docs](https://sgl-project.github.io/diffusion/) — install, CLI, and cookbooks for WAN 2.1, WAN 2.2, FLUX, Qwen-Image variants.
- [FlashAttention paper (Dao 2022)](https://arxiv.org/abs/2205.14135) — the tiled-softmax trick, still the most important attention kernel.
- [Cache-DiT paper](https://arxiv.org/abs/2410.05317) and [`vipshop/cache-dit`](https://github.com/vipshop/cache-dit) — the diffusion-specific caching insight.
- [xDiT / xfuser](https://github.com/xdit-project/xDiT) — diffusion sequence parallelism.
- [SageAttention paper](https://arxiv.org/abs/2410.02367) — quantized attention.

# Module 4.2 — Inference acceleration

**Goal**: take WAN inference from "it works" (lab 4.1) to "it's fast." Read what production *inference* engines actually do — kernel fusion, attention backend selection, feature caching, parallelism — by mapping each technique to (1) the upstream library that implements it, (2) which lab in this curriculum introduced the underlying concept, and (3) which `sglang` CLI flag turns it on. Run a benchmark that compares stock `diffusers` vs SGLang-Diffusion on the same prompt.

This lab is **inference-only**. Every technique below targets generation latency on a fixed model. Acceleration techniques that overlap with *training* (FlashAttention, mixed precision, sequence parallelism, FSDP) live in lab 4.3 / 4.5.

**4× 4090 (96 GB total VRAM) is the default** for this lab; single-GPU still covers the basic benchmark and the `torch.compile` deep dive, but the sequence-parallelism content (USP, Ring, Ulysses) needs multi-GPU.

## Acceleration over lab 4.1

Lab 4.1 used `WanPipeline.from_pretrained(...).__call__(...)` directly. That's the readable, hackable, ~30-second-load reference path — and on a recent CUDA + PyTorch build it already gets you a fair amount for free:

- **FlashAttention 2** via PyTorch's `F.scaled_dot_product_attention` dispatch — no extra setup needed.
- **bf16** weights and activations.

What it *doesn't* get you, and what lab 4.2's deeper engines add:

- Every `Linear → activation → Linear` runs as separate CUDA kernel launches — fusion (`fused_qkv`, `gate+up+SiLU`, JIT'd QK-norm, custom timestep kernel) collapses dozens of launches into one.
- Only the SDPA/FlashAttention-2 backend is reachable; no SageAttention, no FlashAttention 3, no FlashInfer RoPE — all of which SGLang-Diffusion swaps in by flag.
- No caching of redundant DiT computation across diffusion steps (Cache-DiT contributes ~1.7× by itself).
- Single-GPU only — no USP / Ring / Ulysses sequence parallelism, no layerwise weight offload.

A production inference engine fixes all of those.

## The technique map

Every optimization in SGLang-Diffusion comes from somewhere. Most are upstream libraries; SGLang-Diffusion is the orchestration layer that picks and composes them.

| Technique | Where it comes from | What it does |
|---|---|---|
| **FlashAttention 2/3** | [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) — external | Tiled softmax over Q@Kᵀ that never materializes the full attention matrix. Core LLM-serving primitive. |
| **SageAttention 2/3** | [`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention) — external | Quantized attention (INT8 / FP8 Q/K/V). Cuts attention FLOPs and memory at the cost of <1% quality. |
| **FlashInfer RoPE** | [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) — external | Inplace, fused rotary embedding. Replaces ~5 PyTorch ops with one kernel. |
| **Fused QKV** | model-adapter pattern, not a library | One `Linear(hidden, 3·hidden)` + split, instead of three `Linear`s. |
| **Fused gate+up+SiLU** (SwiGLU) | usually `flashinfer.silu_and_mul` | One kernel for `silu(gate(x)) * up(x)`. |
| **JIT QK-norm kernel** | in-house Triton / `torch.compile` | Fuses the per-head Q/K RMSNorm that some DiT variants use. |
| **Custom timestep CUDA kernel** | in-house | Sinusoidal `t` embedding written as one CUDA kernel instead of many tiny PyTorch ops. |
| **Cache-DiT** | [`vipshop/cache-dit`](https://github.com/vipshop/cache-dit) — external | DiT-specific feature caching: skip recomputing block outputs that haven't changed across diffusion steps. ~1.7× alone in the SGLang-Diffusion blog. |
| **Layerwise weight offload** | in-house orchestration | Prefetch layer N+1 onto GPU during layer N's compute. Hides PCIe transfer cost. |
| **Sequence parallelism** (Ulysses + Ring USP) | [`xfuser` / xDiT](https://github.com/xdit-project/xDiT) — external (DeepSpeed-Ulysses lineage) | Shard the long sequence (long video → many tokens) across GPUs along the sequence dim. |
| **Tensor parallelism** | standard | Shard linear-layer weights across GPUs. |

The pattern: SGLang-Diffusion's *original code* is the runtime that schedules these kernels and applies offload / parallelism plans. Each named library can be A/B-tested by toggling its flag.

## Setup

From the repo root with the shared venv activated:

```bash
pip install -r lab4.2/requirements.txt

# SGLang-Diffusion is installed separately because it requires uv + prerelease pins:
pip install --upgrade uv
uv pip install "sglang[diffusion]" --prerelease=allow
```

Verify SGLang is on PATH:

```bash
sglang generate --help    # should print the CLI flags listed in the toggle map
```

## Run the benchmark

Generate the same 3-second clip via three backends — stock diffusers, diffusers + `torch.compile`, and SGLang-Diffusion:

```bash
cd lab4.2
python benchmark.py --prompt "a fluffy red panda eating bamboo on a tree branch"
```

Output:

```
prompt: 'a fluffy red panda eating bamboo on a tree branch'
shape:  49 frames @ 832×480,  steps=30,  cfg=5.0,  seed=42

=== diffusers ===
  load:        45.2s
  generate:   220.4s (steady-state, post-warmup)
  peak VRAM:  10.78 GB
  saved out_diffusers.mp4

=== diffusers + torch.compile(default) ===
  load:        45.0s
  warming up torch.compile (this is the slow part — ~30-60s)...
  warmup:      48.7s
  generate:   132.8s (steady-state, post-warmup)
  peak VRAM:  11.42 GB
  saved out_diffusers_compile_default.mp4

=== diffusers + torch.compile + CFG parallel (2 GPUs) ===
  load:        47.5s
  warming up torch.compile (~30-60s × 2 GPUs in parallel)...
  warmup:      52.1s
  generate:    78.4s (steady-state, post-warmup)
  peak VRAM:   9.65 GB  (max across both GPUs)
  saved out_cfg_parallel.mp4

=== sglang ===
  $ sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers ...
  wall:       82.1s  (includes load + generate, can't separate)
  saved out_sglang.mp4

=== comparison ===
backend                                  wall     peak VRAM
----------------------------------------------------------
diffusers                               220.4s     10.78 GB
diffusers + torch.compile               132.8s     11.42 GB
diffusers + compile + CFG parallel       78.4s      9.65 GB
sglang-diffusion                         82.1s    (see sglang logs)

torch.compile speedup:        1.66×  (vs diffusers baseline)
compile + CFG parallel:       2.81×  (vs diffusers baseline)
sglang:                       2.69×  (vs diffusers baseline)
```

Numbers vary by GPU, drivers, and SGLang version. The shape:
- `torch.compile` alone gets ~1.5–1.7× (Inductor kernel fusion).
- `torch.compile + CFG parallel` gets ~2.5–2.9× (kernel fusion *and* the two CFG forwards run concurrently on separate GPUs — exactly the speedup `mode="reduce-overhead"` would have given on a single GPU if it didn't break with WAN's CFG aliasing). Auto-skipped on single-GPU machines.
- `sglang-diffusion` gets ~2.5–3× even on a single GPU by composing kernels (FlashInfer RoPE, fused QKV/QK-norm/SwiGLU) and Cache-DiT that `torch.compile` can't reach.

Skip flags: `--skip-compile`, `--skip-cfg-parallel`, `--skip-sglang`.

## Toggle map (the SGLang-Diffusion flags)

What makes SGLang-Diffusion teachable: every optimization in the technique map up top is independently turn-on-able from the CLI. From the [official CLI docs](https://sgl-project.github.io/diffusion/api/cli.html):

| Optimization | Flag |
|---|---|
| Cache-DiT | `--cache-dit-config <path>` (or `SGLANG_CACHE_DIT_ENABLED=true`) |
| Attention backend | `--attention-backend {fa,sage,xformers,native,_flash_3_hub}` |
| Layerwise weight offload | `--dit-layerwise-offload true` |
| Sequence parallelism | `--sp-degree N`, `--ulysses-degree N`, `--ring-degree N` |
| Tensor parallelism | `--tp-size N` |
| VAE memory | `--vae-tiling`, `--vae-slicing` |
| Text-encoder offload | `--text-encoder-cpu-offload`, `--pin-cpu-memory` |

## Suggested A/B exercises

The lab's deeper exercise, once the basic benchmark works: pick one flag from the toggle map, run the same prompt with and without it, compare wall clock. *Which optimization buys you what.*

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

## Deep dive: torch.compile + AOT for inference

The other production lever, complementary to SGLang-Diffusion's kernel composition: PyTorch's compiler. **One line of code, ~1.3–1.8× speedup on a single GPU**, no SGLang install needed. Works on top of stock `WanPipeline`.

### `torch.compile` (JIT)

```python
pipe = WanPipeline.from_pretrained(...).to("cuda")
pipe.transformer = torch.compile(pipe.transformer, mode="default")
```

That's it. The first inference call triggers compilation (~30–60 s warmup); subsequent calls are fast.

> **Why `mode="default"` and not `mode="reduce-overhead"` on a single 4090.** `reduce-overhead` is theoretically the faster mode for diffusion sampling (it adds CUDA graph capture on top of Inductor fusion). It does **not** work on a single GPU running `WanPipeline` because the pipeline calls the transformer **twice per step** (conditional + unconditional, for CFG), and the second call's CUDA Graph replay aliases over the first call's output buffer while it's still being read for the `v_uncond + s · (v_cond − v_uncond)` extrapolation. You see `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`. Workarounds — wrap the transformer to clone its output, run the two CFG branches on separate GPUs (CFG parallel, ~1.8× over single-GPU default), or wait for diffusers to ship `cudagraph_mark_step_begin()` between calls — are detailed in Pitfalls below. **For a single-4090 baseline, `mode="default"` is the realistic ceiling.**

What `mode` does:

- **`default`** — TorchDynamo + AOTAutograd + Inductor. Constant folding, dead-code elimination, kernel fusion via Triton. **What this lab uses.**
- **`reduce-overhead`** — adds CUDA graph capture on top. Eliminates per-step Python and kernel-launch overhead. Theoretically the best mode for diffusion sampling loops, but **broken with `WanPipeline`'s two-pass CFG**: the second transformer call's CUDA Graph replay overwrites the first call's output buffer while it's still being read for the `v_uncond + s · (v_cond − v_uncond)` extrapolation, raising `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`. Workarounds: clone the transformer output between calls (requires wrapping the transformer module), or call `torch.compiler.cudagraph_mark_step_begin()` between the conditional and unconditional forwards (requires patching diffusers' pipeline).
- **`max-autotune`** — autotunes Triton kernels at compile time, no CUDA Graphs. ~5–10 minutes of compile, ~5–10% extra steady-state speedup over `default`. Worth it for production builds, not dev iteration.

Pitfalls:

- **Dynamic shapes recompile.** Vary `num_frames` between calls and you'll trigger a recompile. Either fix the shape, or accept the recompile cost.
- **First call is slow.** Always exclude warmup from your benchmarks.
- **VAE is rarely worth compiling.** Most of the compute is in the transformer; compiling the VAE adds compile-time cost without proportional speedup.
- **CFG-batched pipelines and CUDA Graphs.** Any pipeline that calls the model twice per step (CFG) is incompatible with `mode="reduce-overhead"` unless you clone outputs between calls. SGLang-Diffusion handles this in its own runtime; `torch.compile` doesn't.

`benchmark.py` in this lab includes a `torch.compile` path so you can A/B against the stock diffusers baseline on the same prompt/seed.

### AOT compilation (`torch.export` + AOTInductor)

`torch.compile` is JIT — compile happens in your process, on first call. **AOTInductor** lets you compile *ahead of time* and ship a self-contained `.pt2` artifact you can load without Python in the loop.

```python
# Compile once, save artifact:
exported = torch.export.export(pipe.transformer, args=example_inputs)
torch._inductor.aoti_compile_and_package(exported, "wan_transformer.pt2")

# Later, in production:
loaded = torch._inductor.aoti_load_package("wan_transformer.pt2")
output = loaded(z_t, t, text_embeds, mask)
```

Why production cares:

- **No Python overhead at inference.** The packaged binary is self-contained C++.
- **Shippable.** Build once on a beefy box, deploy to any GPU with the right CUDA version.
- **Composable.** Wrap the AOT-compiled transformer in a regular Python sampling loop; only the hot inner forward is "compiled."

Caveats:

- API is still evolving (`torch._inductor.aoti_compile_and_package`, `torch._inductor.aoti_load_package`) — check current PyTorch docs for the latest signatures.
- AOT with dynamic shapes is doable but harder; many production deployments fix the shape first.

### How `torch.compile` and SGLang-Diffusion overlap

Most of SGLang-Diffusion's "JIT" kernels (the QK-norm fusion, fused gate+up+SiLU, fused QKV) are conceptually what `torch.compile` would generate from the same Python. SGLang-Diffusion adds three things `torch.compile` can't reach:

- **Ahead-of-time decisions** — which attention backend, which precision, which caching policy.
- **External kernel libraries** — FlashInfer, Cache-DiT — `torch.compile` uses Inductor + Triton, doesn't dispatch into other libraries.
- **Multi-GPU orchestration** — USP (sequence parallelism), layerwise weight offload — outside `torch.compile`'s purview entirely.

For single-GPU inference, **`torch.compile` alone gets you most of the way.** SGLang-Diffusion adds another 1.5–2× on top by composing all the parts `torch.compile` can't reach.

## Deep dive: sequence parallelism (USP = Ring + Ulysses)

Why video DiT specifically needs this: at 81 frames × 720p, the patchified token count is millions. The attention `Q @ Kᵀ` matrix at that length doesn't fit a single GPU's memory regardless of any other optimization. Image DiT hits ~16k tokens at most and never needs sequence sharding; video forces it.

This is **the technique that earns the curriculum's 4090×4 compute target**. On 1× 4090 you'd just be reading the mental model; on 4× 4090 you can actually run the toggles below and watch wall-clock scale.

### Ring Attention — shard sequence, stream K/V

Each of N GPUs holds 1/N of the sequence's `Q`, `K`, `V`. To compute attention, every GPU needs to see every other GPU's K/V slice. Ring Attention does this by *streaming* K/V chunks around the GPUs:

```
4 GPUs, sequence sharded 4-way
─────────────────────────────────────────────────
GPU 0:  Q[0]    K[0],V[0]
GPU 1:  Q[1]    K[1],V[1]
GPU 2:  Q[2]    K[2],V[2]
GPU 3:  Q[3]    K[3],V[3]

Step 1:  each GPU does partial attention(Q[i], K[i], V[i])
Step 2:  K/V chunks rotate clockwise; GPU i now has K[i-1], V[i-1]
         partial attention(Q[i], K[i-1], V[i-1]); update online softmax
Step 3:  rotate again; partial attention with K[i-2], V[i-2]
Step 4:  rotate again; partial attention with K[i-3], V[i-3]
─────────────────────────────────────────────────
After N steps, every GPU has the full attention output for *its* Q chunk.
```

The softmax is updated using the **same online-softmax trick FlashAttention uses** (track running max + rescale partial sums). After N rounds, results are mathematically identical to single-GPU attention.

**Communication overlaps with compute** — while GPU `i` computes attention with the current K/V chunk, the next chunk is in-flight from GPU `i+1`. Bandwidth-efficient; latency-tolerant.

### Ulysses Attention — shard heads, all-to-all

Same goal, totally different communication pattern:

```
Initial layout: sequence-sharded
  GPU i has [Q[i], K[i], V[i]]   shape (seq/N, all_heads, head_dim)

→ All-to-all #1: redistribute to head-sharded
  GPU i has [Q, K, V]            shape (full_seq, n_heads/N, head_dim)

→ Local attention: each GPU does standard attention on its head slice

→ All-to-all #2: redistribute back to sequence-sharded
  GPU i has output[i]            shape (seq/N, all_heads, head_dim)
```

Communication: **2× all-to-all per attention layer**. Lower latency than Ring (`2` rounds vs `N` rounds), but consumes more total bandwidth at large `N`. Better for shorter sequences / fewer GPUs.

### USP — combine both

xDiT's contribution: shard the sequence in *two* dimensions. Use Ring across many GPUs (good bandwidth utilization) and Ulysses *within* a small Ring group (low latency for the inner step).

```
8-GPU example:  --ulysses-degree 2 --ring-degree 4

  Inner: 2-way Ulysses (head-shard via all-to-all)
  Outer: 4-way Ring   (sequence-shard via streaming K/V)
```

Tuning rule of thumb: **Ulysses degree should match NVLink topology** (NVLink-connected pairs do the all-to-all efficiently); **Ring degree spans across NVLink boundaries** (higher latency tolerance there).

### Diffusion-specific complications

1. **CFG batching.** Conditional and unconditional forwards run as one `(2·B, ...)` batch. Sequence parallelism has to be CFG-aware so the all-to-all doesn't shuffle the conditional samples into the unconditional ones.
2. **3D RoPE positions.** Sharded sequences need consistent `(t, h, w)` indices per token — each shard must know which slice of the global position grid it owns. xDiT handles this; if you write your own SP, it's the bug to watch for.
3. **Cross-attention K/V are tiny** (77 text tokens × hidden). **Don't** sequence-shard them — replicate per GPU. They're cheap.

### Multi-GPU exercise (4× 4090 default)

```bash
# Single GPU baseline:
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "..." \
    --sp-degree 1

# Pure Ulysses (all-to-all only), 4 GPUs:
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "..." \
    --sp-degree 4 --ulysses-degree 4 --ring-degree 1

# Pure Ring (streaming K/V only), 4 GPUs:
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "..." \
    --sp-degree 4 --ulysses-degree 1 --ring-degree 4

# USP hybrid (2-way Ulysses × 2-way Ring), 4 GPUs:
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "..." \
    --sp-degree 4 --ulysses-degree 2 --ring-degree 2
```

All four should produce identical output for the same seed (sequence parallelism is mathematically equivalent to single-GPU). Differences are in wall clock and per-GPU memory. At this scale (1.3B model, 480p video), the four 4090s aren't NVLinked so all-to-all bandwidth is PCIe-limited — Ring tends to win over Ulysses. At production scales (14B + 720p × 81 frames on NVLinked H100s), USP hybrid wins, and it's the difference between "fits" and "doesn't fit."

### Where to read

- **[Ring Attention paper (Liu et al. 2023)](https://arxiv.org/abs/2310.01889)** — the streaming-K/V scheme.
- **[DeepSpeed-Ulysses paper (Jacobs et al. 2023)](https://arxiv.org/abs/2309.14509)** — the all-to-all scheme.
- **[`xfuser` / xDiT source](https://github.com/xdit-project/xDiT)** — production implementation; the diffusion-specific glue (CFG-aware SP, 3D RoPE handling) lives here.
- **[`Wan-Video/Wan2.2/wan/distributed/`](https://github.com/Wan-Video/Wan2.2/tree/main/wan/distributed)** — a real production team's adaptation of xfuser; smaller surface area than xfuser itself.

## Discussion

### Training vs inference

Every technique in this lab is for inference. The training-relevant subset:

- **FlashAttention** — works in training too (and `WanPipeline`'s training flavor in `diffusers` already uses it under the hood when available).
- **Mixed precision (bf16)** — training, yes (lab 4.3 / 4.5 already use it).
- **Sequence parallelism** — works in training too via `xfuser` / DeepSpeed-Ulysses; it's used to train long-video models that don't fit on one GPU.
- **FSDP (params + grads + optim sharded across GPUs)** — *training-only*. The dominant strategy for training a model that doesn't fit one GPU's worth of parameters + Adam state. Lab 4.5's full SFT discusses where it kicks in (1.3B doesn't need it; 14B / Wan-2.2-A14B does). At inference there's no optimizer state and no backward pass to shard, so FSDP doesn't apply — the equivalent shape is **tensor parallelism** (already in this lab's technique map).
- **Cache-DiT** — *inference-only*. Caching only works when steps converge to similar outputs, which doesn't apply during training where every step changes the loss landscape.
- **Quantized attention (SageAttention)** — inference-only at production quality. Quantized training is its own research area (FP8 training is reaching production but mostly for LLMs).

### What SGLang-Diffusion isn't

- **Not a model**. It's an inference engine. The model weights are still WAN's.
- **Not a training framework**. For training, lab 4.3 (LoRA) and 4.5 (full SFT) use `accelerate` + `bitsandbytes` + `peft` directly. Some kernels overlap; the orchestration is different.
- **Not the only option** — see the comparison below.

### Comparison of inference frameworks

| Framework | Shape | Best for |
|---|---|---|
| **SGLang-Diffusion** (this lab) | full engine: kernel composition + multi-backend attn + Cache-DiT + USP | most thorough open-source production engine; deepest single-GPU and multi-GPU |
| **diffusers** + `torch.compile` | reference + JIT compiler | hackable baseline; ~1.5–1.8× over plain diffusers |
| **[OneDiff](https://github.com/siliconflow/onediff)** | graph compiler, drop-in for diffusers | one-line speedup without rewriting kernels |
| **[xDiT / xfuser](https://github.com/xdit-project/xDiT)** | parallelism library | SP/TP wrappers you layer onto stock diffusers |
| **[TensorRT-LLM-Diffusion](https://github.com/NVIDIA/TensorRT-LLM)** | AOT graph compiler, NVIDIA-blessed | single-tenant H100/B200 deployments |
| **[vLLM-Diffusion](https://github.com/vllm-project/vllm)** | continuous-batching engine (LLM heritage) | batch-serving; less mature for video DiT as of mid-2026 |

SGLang-Diffusion sits at the upper end of the *full-engine* end of this spectrum. OneDiff is the easy on-ramp; xDiT is what you reach for when you only need parallelism; TensorRT is for "I'll pay setup cost for max single-GPU throughput."

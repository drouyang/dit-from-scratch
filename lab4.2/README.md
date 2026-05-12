# Module 4.2 — Diffusers inference optimization

> Part 4 — DiT in Production · [DiT from Scratch](../README.md)

**Goal**: take WAN inference from lab 4.1's "it works" to "it's fast" — *without leaving `diffusers`*. The toolset is everything PyTorch + `diffusers` give you out of the box: `torch.compile` (JIT and AOT), `F.scaled_dot_product_attention` backend selection, and `diffusers`' model-offload helpers. Lab 4.3 adds the next layer (the SGLang production runtime); this lab establishes what the *hackable* layer reaches.

**Why this lab is diffusers-only**: every optimization here is something you can read, modify, and reason about with vanilla PyTorch. Single-file scripts, no extra runtime to install, no compiled-kernel ABI to manage. That makes this the right place to learn *what each lever does* before moving to lab 4.3, where the same techniques are flags on a production engine.

**Compute**: this lab is **single-GPU-friendly**. A 4090 / A100 / H100 is enough for everything below. (Multi-GPU sequence parallelism + tensor parallelism live in lab 4.3, which is where 4× 4090 starts to matter.)

## Acceleration over lab 4.1

Lab 4.1 ran `WanPipeline.from_pretrained(...).__call__(...)` directly. On a recent CUDA + PyTorch build that already gives you:

- **FlashAttention 2** via PyTorch's `F.scaled_dot_product_attention` dispatch — no extra setup.
- **bf16** weights and activations.

What it leaves on the table — and what this lab fixes:

- Every `Linear → activation → Linear` runs as separate CUDA kernel launches. **`torch.compile`** fuses these via Inductor + Triton.
- **PyTorch's compiler also offers ahead-of-time (AOT)** export — `torch.export` + AOTInductor produce a portable `.pt2` binary that loads without re-paying compile cost.
- PyTorch picks an SDPA backend automatically, but you can **force a specific one** (`FLASH_ATTENTION` / `EFFICIENT_ATTENTION` / `MATH`) and benchmark them against each other.
- VRAM-tight rigs OOM on load. **`enable_model_cpu_offload()`** / **`enable_sequential_cpu_offload()`** push idle pieces to CPU; the right one buys you a fit on a 12 GB card with a small latency cost.

What this lab *doesn't* reach — and lab 4.3 (SGLang) does:

- Hand-written attention kernels (SageAttention, FlashInfer RoPE, fused QKV).
- Multi-GPU sequence parallelism (USP / Ring / Ulysses).
- CFG-parallel (cond + uncond on separate GPUs concurrently).
- DiT-specific feature caching (Cache-DiT).

Those need a different runtime; that's lab 4.3's job.

## Setup

This lab uses the shared root `.venv` every other lab in the curriculum uses — no per-lab venv needed.

```bash
# From the repo root:
source .venv/bin/activate
pip install -r lab4.2/requirements.txt
```

The benchmark below loads Wan-2.1 T2V-1.3B from HuggingFace (~5 GB total: VAE + text encoder + transformer; cached on first run). The repo is public — no auth required.

## Run the benchmarks

`benchmark.py` runs **one config per invocation** (orthogonal `--compile`, `--aot`, `--sdpa-backend`, `--offload` flags) and times **two inference calls back-to-back** — `first call` (cold; includes JIT compile or AOT load) and `second call` (warm steady-state). The four experiments below each compose multiple invocations into a comparison.

Each invocation reloads the model (~45 s mostly for the VAE / text encoder); accepted as the cost of treating every config as a fresh run. Activate the venv from the repo root first:

```bash
source .venv/bin/activate
cd lab4.2
```

### Baseline

```bash
python benchmark.py
```

One row, no optimizations. The reference point every experiment compares against. Measured on a single 4090: `model load ≈ 5.8 s, first ≈ 77.1 s, second ≈ 77.2 s, peak VRAM ≈ 20.5 GB`. Both call columns match because there's no JIT compile or AOT load to amortize.

### Experiment 1 — `torch.compile` modes

```bash
# Prefix with TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) so each invocation gets a
# fresh Inductor cache — otherwise re-runs hit the disk cache and the "first
# call" number drops from ~119 s (cold compile) to ~70 s (warm cache).
TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) python benchmark.py --compile default
TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) python benchmark.py --compile max-autotune-no-cudagraphs
```

Key API (`benchmark.py:160`) — the CLI flag flows straight into `torch.compile`'s `mode=` kwarg:

```python
pipe.transformer = torch.compile(pipe.transformer, mode=args.compile)
```

**PyTorch ships four named modes.** They compose three orthogonal levers (Inductor fusion, CUDA Graph capture, Triton autotune):

| Mode | Inductor fusion | CUDA Graphs | Autotune | Works on WAN? |
|---|---|---|---|---|
| `default` | ✓ | — | — | ✓ |
| `reduce-overhead` | ✓ | ✓ | — | ✗ (CFG aliasing) |
| `max-autotune` | ✓ | ✓ | ✓ | ✗ (CFG aliasing) |
| `max-autotune-no-cudagraphs` | ✓ | — | ✓ | ✓ |

**Why `reduce-overhead` and `max-autotune` both fail on WAN.** Both modes enable **CUDA Graph capture**, which records a stream of kernel launches once and replays it with near-zero CPU overhead. That replay reuses the same output buffer across calls. WAN's pipeline calls the transformer **twice per step** (conditional + unconditional, for CFG), then computes `v_uncond + s · (v_cond − v_uncond)` — but by the time the extrapolation reads the *first* call's output, the *second* call's replay has already overwritten it. You get:

```
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
```

So this experiment uses `max-autotune-no-cudagraphs` instead of `max-autotune`. `default` works for the same reason — no CUDA Graphs in the first place. Three other workarounds (none yet upstream in diffusers): wrap the transformer output in `.clone()`, call `torch.compiler.cudagraph_mark_step_begin()` between cond and uncond forwards, or run cond + uncond on separate GPUs (CFG-parallel, lab 4.3).

What you're measuring: how much Inductor + Triton fusion + autotune (optional) accelerates the *transformer* forward, and how much warmup costs you for that win.

Expected, single 4090:

| config | model load | first call | second call | speedup (second) | peak VRAM |
|---|---|---|---|---|---|
| baseline | 5.8 s | 77 s | 77 s | 1× | 20.5 GB |
| `--compile default` | 6.3 s | 119 s | 62 s | 1.24× | 20.5 GB |
| `--compile max-autotune-no-cudagraphs` | 6.3 s | 229 s | 62 s | 1.25× | 20.5 GB |

Second-call latency is the production-relevant number (steady-state). First-call latency is what you'd pay every cold start without AOT.

**Why `--compile default`'s first call (119 s) is *slower* than the baseline first call (77 s).** Compile work happens *during* the first forward, not at `torch.compile(...)` time. Dynamo traces the graph, AOTAutograd captures it, Inductor lowers to Triton, and Triton emits CUDA kernels — roughly ~56 s of that for WAN's transformer. Only *after* all that does the compiled forward actually run (in ~62 s, matching second-call). So:

```
baseline first call:           uncompiled_gen                  ≈ 77 s
--compile default first call:  compile_overhead + compiled_gen ≈ 56 + 62 = 119 s
--compile default second call: compiled_gen                    ≈ 62 s   ← the win
```

You pay the compile tax once per process. After that, every call is faster.

**Why `max-autotune-no-cudagraphs` doesn't beat `default` here.** Steady-state is essentially identical (62 s vs 62 s), but the first call jumps from 119 s to 229 s — Triton's autotune searches the kernel-parameter space (tile sizes, num warps, etc.) for the matmul + epilogue ops, adding ~170 s of compile time. On bigger DiTs (SD3 2B, FLUX 12B) autotune typically buys ~5–10% steady-state. For WAN 1.3B the hot kernels are *already* picked well by Inductor's defaults — there's nothing for autotune to find. So `max-autotune-no-cudagraphs` is a 3× compile cost for ~0% return on this model. Useful to verify; not useful to ship.

**Why 1.24× is on the low end** (typical for DiTs is 1.5–1.7×). Two plausible reasons:

1. WAN's transformer is small (1.3B params). Kernel-launch overhead is already a smaller fraction of total time, so Inductor's fusion has less to save. Bigger transformers (SD3 2B, FLUX 12B) typically see bigger compile wins.
2. bf16 + FlashAttention via SDPA is already very efficient. PyTorch's SDPA dispatcher routes attention to FlashAttention 2 automatically on Ampere+; the matmuls run in bf16 on Tensor Cores. The dominant ops are already well-tuned kernels, so Inductor's fusion is mostly cleaning up the small ops *around* them (norms, residual adds, gates) — not the hot path itself.

### Experiment 2 — AOT cold-start vs JIT

```bash
# One-time export (slow — warms compile, then packages):
python benchmark.py --aot save                # writes wan_transformer.pt2

# Then measure the AOT cold-start path:
python benchmark.py --aot load                # first call skips compile entirely
# (The --compile default row in the table below is the same measurement
# from Experiment 1 — no need to re-run.)
```

Key APIs — export (write `.pt2`), then load (skip JIT on next process start):

```python
# benchmark.py:223 — --aot save: trace + package
exported = torch.export.export(pipe.transformer, args=example_inputs)
torch._inductor.aoti_compile_and_package(exported, package_path=args.aot_path)

# benchmark.py:268 — --aot load: deserialize into the transformer slot
loaded = torch._inductor.aoti_load_package(args.aot_path)
pipe.transformer = _AOTWrapper(loaded, pipe.transformer)  # mimics transformer.__call__
```

What you're measuring: AOT's value is in the **first-call** column, not the second. After warmup both paths converge to the same steady state.

Expected, single 4090:

| config | model load | first call | second call | peak VRAM |
|---|---|---|---|---|
| baseline | 5.8 s | 77 s | 77 s | 20.5 GB |
| `--compile default` (JIT) | 6.3 s | 119 s | 62 s | 20.5 GB |
| `--aot load` | 5.8 s | 62 s | 62 s | 20.5 GB |

The headline: **AOT's first call (62 s) matches its second call (62 s)**, while JIT's first call (119 s) pays a ~57 s compile tax. AOT moves that compile work to `--aot save` (one-time, ahead-of-time), and `--aot load` deserializes the already-compiled `.pt2` — the next process's first call runs at full steady-state speed immediately. For serverless / autoscaling deploys where every container restart re-pays cold-start cost, this is the big win.

> **Caveat**: AOTInductor's API has moved between PyTorch versions. If `--aot save` fails with an `AttributeError` on `torch._inductor.aoti_compile_and_package`, your PyTorch is newer or older than the script targets — check [PyTorch's AOTI tutorial](https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html) for the current call signature.

### Deep dive: AOT compilation (`torch.export` + AOTInductor)

`torch.compile` is JIT — compile happens *in your process, on first call*. The compiled artifact dies with the process. **AOTInductor** lets you compile *ahead of time* and ship a self-contained `.pt2` binary you can load without paying the JIT cost again.

```python
import torch
from diffusers import WanPipeline

pipe = WanPipeline.from_pretrained(...).to("cuda")

# 1) Build example inputs that capture the full input signature of transformer.forward
example_inputs = (
    torch.randn(1, 16, 13, 60, 104, dtype=torch.bfloat16, device="cuda"),  # hidden_states
    torch.tensor([500.0], device="cuda"),                                   # timestep
    torch.randn(1, 512, 4096, dtype=torch.bfloat16, device="cuda"),         # encoder_hidden_states
)

# 2) Export → AOT compile → package
exported = torch.export.export(pipe.transformer, args=example_inputs)
torch._inductor.aoti_compile_and_package(
    exported,
    package_path="wan_transformer.pt2",
)

# 3) Later (different process, possibly different machine):
loaded = torch._inductor.aoti_load_package("wan_transformer.pt2")
output = loaded(*example_inputs)
```

#### Why production cares

- **No Python overhead at inference.** The packaged binary is self-contained C++. Load it from any wrapper (Python, C++, Triton Inference Server) without paying for `torch.compile`'s JIT warmup or its Python control flow.
- **Shippable.** Build once on a beefy machine, deploy to any GPU with a compatible CUDA version. Frees the inference host from needing the full PyTorch + compiler toolchain.
- **Composable.** AOT-compile only the hot transformer; wrap it in a regular Python sampling loop. The loop stays hackable, the inner forward is "compiled C++."

#### Caveats

- **API is still evolving.** `torch.export` and AOTInductor are stable enough for production use (PyTorch 2.4+) but the exact function names (`torch._inductor.aoti_compile_and_package`, `torch._inductor.aoti_load_package`) may move; check the current PyTorch docs.
- **Dynamic shapes are harder.** AOT export with truly dynamic shapes is doable via `dynamic_shapes` in `torch.export.export(...)`, but most production deployments fix the shape (e.g., one `.pt2` per resolution preset) and accept the storage cost.
- **No backward.** AOTInductor is forward-only. Training keeps using `torch.compile` JIT.

#### When AOT beats JIT

AOT shines when warmup time matters or repeated cold starts hurt:

- Serverless / autoscaling deploys where each container restart would re-pay JIT compile.
- Embedded / mobile-adjacent deployments where Python isn't desired.
- Multi-model orchestrators where you load and unload models dynamically.

For a long-running single-process server, plain `torch.compile` JIT is fine — you pay the warmup once.

### Experiment 3 — SDPA backend

```bash
python benchmark.py --sdpa-backend flash      # FlashAttention 2 forced
python benchmark.py --sdpa-backend efficient  # xFormers-style memory-efficient
python benchmark.py --sdpa-backend cudnn      # cuDNN's fused attention
```

Expected, single 4090:

| config | model load | second call | speedup | peak VRAM |
|---|---|---|---|---|
| baseline (auto) | 5.8 s | 77 s | 1× | 20.5 GB |
| `--sdpa-backend flash` | 5.7 s | 77 s | 1.00× | 20.5 GB |
| `--sdpa-backend efficient` | 5.7 s | 93 s | 0.83× | 20.5 GB |

On this model, `flash` ties the auto-dispatch baseline (~1.00×) and `efficient` runs ~17% slower — long-sequence attention (832×480 × 49 frames produces a long token sequence) genuinely favors FlashAttention's tiled algorithm over xFormers-style memory-efficient attention.

### Deep dive: SDPA backend selection

`F.scaled_dot_product_attention` (used by `WanTransformer3DModel`'s attention internally) picks a backend automatically. You can force a specific one:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = pipe(prompt=...)
```

| Backend | What it is | When to use |
|---|---|---|
| `FLASH_ATTENTION` | FlashAttention 2 (or FA3 on H100 if available). | Default for long sequences. What most production deployments get. |
| `EFFICIENT_ATTENTION` | xFormers' memory-efficient attention. Different kernel, similar idea. | When FlashAttention is unavailable (older hardware, exotic dtypes) or you suspect a FA-specific bug. |
| `MATH` | Naive PyTorch `softmax(QKᵀ/√d) · V`. | Debugging only. Slow. |
| `CUDNN_ATTENTION` | cuDNN's fused attention. | Hopper (H100, B200) — often the fastest path there. **Ada Lovelace (4090) rejects it at WAN's seq_len** and falls back to MATH (which OOMs); skip on consumer GPUs. |

The benchmark accepts `--sdpa-backend` for a quick A/B. For video DiT at production resolutions, `FLASH_ATTENTION` typically wins — long sequences favor FA's tiled algorithm. The spread vs. the others depends on the model and sequence length: on WAN 1.3B @ 832×480 × 49 frames we measure `efficient` at ~0.83× and `cudnn` rejects with an OOM (because cuDNN's attention is Hopper-tuned, not effective on Ada Lovelace at this seq_len). The other interesting thing is verifying that **all backends that *do* run produce identical output for the same seed** (sequence parallelism / quantized attention break this; SDPA backends don't).

### Experiment 4 — Offload trade-off

```bash
python benchmark.py --offload model           # whole-component CPU offload
python benchmark.py --offload sequential      # per-submodule CPU offload
```

Expected, single 4090:

| config | model load | second call | speedup | peak VRAM |
|---|---|---|---|---|
| baseline | 5.8 s | 77 s | 1× | 20.5 GB |
| `--offload model` | 10.8 s | 78 s | 0.99× | 17.0 GB |
| `--offload sequential` | 11.0 s | 185 s | 0.42× | 15.0 GB |

**Why the two offload modes differ so much** (`--offload model`: ~free / −17% VRAM; `--offload sequential`: 2.4× slower / −27% VRAM).

A diffusion pipeline keeps three neural networks loaded on the GPU:

- **Text encoder** — turns the prompt into embeddings *once* at the start of generation, then sits idle. WAN uses **umT5-XXL** (a 13B-parameter T5 variant), which weighs in at ~11 GB.
- **Transformer (the DiT)** — the main model, runs once per sampling step (30 steps × 2 for CFG = 60 forward passes).
- **VAE** — decodes the final latent into pixel frames *once* at the end.

`enable_model_cpu_offload()` cycles **whole components** (text encoder, transformer, VAE) on and off the GPU. Its normal headline win is moving the text encoder to CPU as soon as it's done — that frees ~11 GB before sampling starts. **But this benchmark already does that manually**: `pipe.text_encoder.to("cpu")` runs right after `encode_prompt`, because without it `torch.compile` OOMs on a 24 GB 4090 (the compiled transformer + Inductor caches need every spare GB during sampling). So by the time `--offload model` activates, the ~11 GB text-encoder win is already baked into the baseline measurement. What's left for the helper to offload is just the transformer↔VAE swap at decode time: ~3 GB savings, sub-second cost. That's why we measure +1% slowdown / −17% VRAM — the *incremental* effect on top of the manually-offloaded baseline, not the offload helper's raw effect on a stock pipeline.

`enable_sequential_cpu_offload()` cycles at a much **finer granularity** — individual submodules (per-block, per-Linear) move on and off GPU on demand. Over a full generation (60 transformer forwards × ~30 blocks each), that's thousands of small PCIe round-trips. Each transfer is fast, but the cumulative overhead dominates: ~107 extra seconds of wall clock for only ~2 GB additional VRAM savings (15.0 vs 17.0 GB). Steep trade-off — useful only when sequential offload is the *only* way the model fits (12 GB consumer card, or running two models concurrently on the same GPU).

## Deep dive: model offloading

For 12 GB / 16 GB consumer cards, the full Wan-2.1-1.3B pipeline (~12 GB at bf16 with VAE + text encoder + transformer all resident) doesn't comfortably fit alongside activations. `diffusers` ships two offload helpers:

```python
pipe.enable_model_cpu_offload()       # default choice
pipe.enable_sequential_cpu_offload()  # more aggressive
```

| Helper | What it does | Latency cost | When to use |
|---|---|---|---|
| `enable_model_cpu_offload()` | Moves *whole components* (VAE, text encoder, transformer) to CPU when not actively in use; brings the one in use back to GPU. | ~5–10% per inference call. | Default for tight-VRAM rigs. Almost no quality of life cost. |
| `enable_sequential_cpu_offload()` | Moves individual *submodules* (per-layer) on and off GPU on demand. | 1.5–3× slower per call. | When even `enable_model_cpu_offload` OOMs. Fits Wan-2.1-1.3B on a 6 GB card. |

`benchmark.py` exposes `--offload model` / `--offload sequential` to A/B these.

## Files

| File | What it is |
| --- | --- |
| `benchmark.py` | Runs the diffusers baseline + `torch.compile(default)` + `torch.compile(max-autotune)` with peak-VRAM tracking, side-by-side comparison, optional `--offload` / `--sdpa-backend` toggles. |
| `requirements.txt` | torch, diffusers, transformers, accelerate, imageio. |

## Discussion

### What diffusers won't get you

Five things, all in lab 4.3:

1. **Custom kernels beyond what Inductor generates.** SageAttention's INT8 attention, FlashInfer's fused RoPE, the JIT'd QK-norm kernel — these are hand-written CUDA, not Triton fusions. `torch.compile` can't emit them.
2. **Multi-GPU sequence parallelism.** The transformer's long-sequence attention sharded across N GPUs (Ulysses + Ring) needs explicit communication primitives. `torch.compile` doesn't insert NCCL ops.
3. **CFG-parallel inference.** Running cond + uncond on separate GPUs concurrently requires a different sampling loop. SGLang ships it as `--enable-cfg-parallel`.
4. **DiT-specific feature caching** (Cache-DiT). Cross-step feature reuse for diffusion needs an outer scheduler that `torch.compile`'s graph can't see.
5. **Tensor parallelism for the transformer**. Sharded linear-layer weights across GPUs. SGLang ships it as `--tp-size`.

If you need any of those, you switch runtimes. That's lab 4.3.

### Where to go deeper

- [PyTorch `torch.compile` docs](https://pytorch.org/docs/stable/torch.compiler.html) — mode reference, dynamic shapes, debugging tips.
- [AOTInductor tutorial](https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html) — end-to-end export + load for a real model.
- [PyTorch SDPA reference](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — backend selection rules.
- [diffusers' optimization guide](https://huggingface.co/docs/diffusers/optimization/torch2.0) — covers `torch.compile`, SDPA, and offloading from the diffusers angle specifically.

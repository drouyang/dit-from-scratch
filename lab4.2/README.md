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

## Run the benchmark

Generate the same 3-second clip via three diffusers paths — stock, `torch.compile(default)`, `torch.compile(max-autotune)`:

```bash
source .venv/bin/activate
cd lab4.2
python benchmark.py --prompt "a fluffy red panda eating bamboo on a tree branch"
```

Output (single 4090, ballpark — varies by driver / PyTorch version):

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

=== diffusers + torch.compile(max-autotune) ===
  load:        45.0s
  warmup:     381.2s  (autotuning Triton kernels — long)
  generate:   118.3s (steady-state)
  peak VRAM:  11.50 GB
  saved out_diffusers_compile_max_autotune.mp4

=== comparison ===
backend                                  wall     peak VRAM
----------------------------------------------------------
diffusers                               220.4s     10.78 GB
diffusers + torch.compile(default)      132.8s     11.42 GB
diffusers + torch.compile(max-autotune) 118.3s     11.50 GB

torch.compile(default) speedup:      1.66×  (vs diffusers baseline)
torch.compile(max-autotune) speedup: 1.86×  (vs diffusers baseline)
```

The headline: **one extra line of code (`torch.compile`) gets you ~1.6–1.9× speedup** on a single GPU, with no model edits, no runtime install, no kernel-level work. That's the whole pitch of this lab.

Pass `--skip-compile` or `--skip-max-autotune` to drop expensive warmup steps during iteration.

## Deep dive: `torch.compile` (JIT)

```python
pipe = WanPipeline.from_pretrained(...).to("cuda")
pipe.transformer = torch.compile(pipe.transformer, mode="default")
```

That's it. The first inference call triggers compilation (~30–60 s warmup); subsequent calls reuse the compiled graph and run fast.

### What `mode` does

| Mode | What it adds | When to use |
|---|---|---|
| **`default`** | TorchDynamo + AOTAutograd + Inductor. Constant folding, dead-code elimination, kernel fusion via Triton. | **Default choice.** Reliable, ~1.5–1.7× speedup, ~30–60 s warmup. |
| **`reduce-overhead`** | Adds CUDA Graph capture on top of `default`. Eliminates per-step Python and kernel-launch overhead. | *Theoretically* best for diffusion sampling loops, but **broken with WAN-style CFG**. See pitfall below. |
| **`max-autotune`** | Autotunes Triton kernels at compile time. No CUDA Graphs. | Production builds. ~5–10 minutes warmup for ~5–10% extra steady-state speedup over `default`. |

### Pitfall: `reduce-overhead` and CFG-batched pipelines

`reduce-overhead` would be the right mode for diffusion sampling — same model, called N times, perfect for CUDA Graphs. It does **not** work on `WanPipeline` as shipped because the pipeline calls the transformer **twice per step** (conditional + unconditional, for CFG), and the second call's CUDA Graph replay overwrites the first call's output buffer while it's still being read for the `v_uncond + s · (v_cond − v_uncond)` extrapolation. You see:

```
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
```

Three known workarounds, none yet shipped in stock diffusers as of writing:

1. **Wrap the transformer's output in a `.clone()`** — adds one buffer copy per call, eliminates the aliasing. ~1.1× slower than ideal but enables `reduce-overhead` to work; usually still a net win vs `default`.
2. **Call `torch.compiler.cudagraph_mark_step_begin()` between cond and uncond forwards** — tells the CUDA Graph that the next call is a separate step, so it allocates fresh output buffers. Needs patching diffusers' pipeline (~5 lines).
3. **Run cond + uncond on separate GPUs (CFG parallel)** — each transformer copy sees only one forward per step → no aliasing. Faster than either of the above, but requires ≥2 GPUs. Covered in lab 4.3 with SGLang's `--enable-cfg-parallel` flag.

For a single-GPU diffusers run, **`mode="default"` is the realistic ceiling** until diffusers ships one of the workarounds.

### Other pitfalls

- **Dynamic shapes recompile.** Vary `num_frames` or `height` / `width` between calls and you trigger a recompile (visible as a 30+ s stall on the next call). Either fix the shape, or accept the recompile cost.
- **First call is slow.** Always exclude warmup from your benchmarks; `benchmark.py` does this for you.
- **VAE rarely worth compiling.** Most compute is in the transformer; compiling the VAE adds compile-time cost without proportional speedup.

## Deep dive: AOT compilation (`torch.export` + AOTInductor)

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

### Why production cares

- **No Python overhead at inference.** The packaged binary is self-contained C++. Load it from any wrapper (Python, C++, Triton Inference Server) without paying for `torch.compile`'s JIT warmup or its Python control flow.
- **Shippable.** Build once on a beefy machine, deploy to any GPU with a compatible CUDA version. Frees the inference host from needing the full PyTorch + compiler toolchain.
- **Composable.** AOT-compile only the hot transformer; wrap it in a regular Python sampling loop. The loop stays hackable, the inner forward is "compiled C++."

### Caveats

- **API is still evolving.** `torch.export` and AOTInductor are stable enough for production use (PyTorch 2.4+) but the exact function names (`torch._inductor.aoti_compile_and_package`, `torch._inductor.aoti_load_package`) may move; check the current PyTorch docs.
- **Dynamic shapes are harder.** AOT export with truly dynamic shapes is doable via `dynamic_shapes` in `torch.export.export(...)`, but most production deployments fix the shape (e.g., one `.pt2` per resolution preset) and accept the storage cost.
- **No backward.** AOTInductor is forward-only. Training keeps using `torch.compile` JIT.

### When AOT beats JIT

AOT shines when warmup time matters or repeated cold starts hurt:

- Serverless / autoscaling deploys where each container restart would re-pay JIT compile.
- Embedded / mobile-adjacent deployments where Python isn't desired.
- Multi-model orchestrators where you load and unload models dynamically.

For a long-running single-process server, plain `torch.compile` JIT is fine — you pay the warmup once.

## Deep dive: SDPA backend selection

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
| `CUDNN_ATTENTION` | cuDNN's fused attention. | Available in recent PyTorch + cuDNN; sometimes the fastest path on Hopper. |

The benchmark accepts `--sdpa-backend` for a quick A/B; usually `FLASH_ATTENTION` wins for video DiT at production resolutions (long sequences favor FA's tiled algorithm) and the others are within ±5%. The interesting thing isn't the win — it's verifying that **all backends produce identical output for the same seed** (sequence parallelism / quantized attention break this; SDPA backends don't).

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

### Why this is a diffusers thing, not a general thing

Both helpers are pipeline-aware: they know which submodules belong to which component and can move them as a unit. SGLang (lab 4.3) has its own offloading flag (`--dit-layerwise-offload`) that does the per-layer version; the pattern is the same, only the runtime differs.

## Files

| File | What it is |
| --- | --- |
| `benchmark.py` | Runs the diffusers baseline + `torch.compile(default)` + `torch.compile(max-autotune)` with peak-VRAM tracking, side-by-side comparison, optional `--offload` / `--sdpa-backend` toggles. |
| `requirements.txt` | torch, diffusers, transformers, accelerate, imageio. |

## Discussion

### What you've learned

After this lab, you can read a diffusion model's forward pass and identify, without external tools:

- Which parts compile cleanly under Inductor (most of the transformer); which don't (data-dependent control flow, dynamic shapes).
- When a CUDA Graph optimization is going to break (any pipeline with two-pass CFG, batched generation, or per-step shape changes).
- Whether to package as a `.pt2` (production, shippable) or stick with JIT compile (dev iteration).
- Which SDPA backend the current PyTorch build is silently dispatching to.
- Whether your rig has enough VRAM to skip offload (and what it'd cost you if not).

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

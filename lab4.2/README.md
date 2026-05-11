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
python benchmark.py --compile default
python benchmark.py --compile max-autotune
```

Key API (`benchmark.py:160`) — the CLI flag flows straight into `torch.compile`'s `mode=` kwarg:

```python
pipe.transformer = torch.compile(pipe.transformer, mode=args.compile)
# args.compile ∈ {"default", "max-autotune", "reduce-overhead"}
```

What you're measuring: how much Inductor + Triton fusion + autotune (optional) accelerates the *transformer* forward, and how much warmup costs you for that win.

Expected, single 4090:

| config | model load | first call | second call | speedup (second) |
|---|---|---|---|---|
| baseline | **5.8 s** | **77 s** | **77 s** | 1× |
| `--compile default` | **6.3 s** | **114 s** | **62 s** | **1.24×** |
| `--compile max-autotune` | TBD | TBD (incl. Triton autotune) | TBD | TBD |

Second-call latency is the production-relevant number (steady-state). First-call latency is what you'd pay every cold start without AOT.

**Why 1.24× is on the low end** (typical for DiTs is 1.5–1.7×). Two plausible reasons:

1. **WAN's transformer is small (1.3B params).** Kernel-launch overhead is already a smaller fraction of total time, so Inductor's fusion has less to save. Bigger transformers (SD3 2B, FLUX 12B) typically see bigger compile wins.
2. **bf16 + FlashAttention via SDPA is already very efficient.** PyTorch's SDPA dispatcher routes attention to FlashAttention 2 automatically on Ampere+; the matmuls run in bf16 on Tensor Cores. The dominant ops are already well-tuned kernels, so Inductor's fusion is mostly cleaning up the small ops *around* them (norms, residual adds, gates) — not the hot path itself.

`--compile max-autotune` will tell us whether there's more headroom (it autotunes Triton kernels) or whether 1.24× is the real ceiling for this model on this GPU. The SDPA-backend experiment is a sanity check that we're already on the fastest attention path.

### Experiment 2 — AOT cold-start vs JIT

```bash
# One-time export (slow — warms compile, then packages):
python benchmark.py --aot save                # writes wan_transformer.pt2

# Now A/B the cold-start path against JIT compile:
python benchmark.py --compile default         # JIT: first call pays compile
python benchmark.py --aot load                # AOT: first call skips compile
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

| config | model load | first call | second call |
|---|---|---|---|
| baseline | **5.8 s** (measured) | **77 s** (measured) | **77 s** (measured) |
| `--compile default` (JIT) | **6.3 s** (measured) | **114 s** (measured; ~52 s compile + ~62 s gen) | **62 s** (measured) |
| `--aot load` | TBD (just `.pt2` load) | TBD | TBD |

`--aot load` should save the JIT-compile portion of cold start *and* skip part of `from_pretrained` (transformer loads from `.pt2` instead). For serverless / autoscaling deploys where every container restart re-pays cold-start cost, this is the big win.

> **Caveat**: AOTInductor's API has moved between PyTorch versions. If `--aot save` fails with an `AttributeError` on `torch._inductor.aoti_compile_and_package`, your PyTorch is newer or older than the script targets — check [PyTorch's AOTI tutorial](https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html) for the current call signature.

### Experiment 3 — SDPA backend

```bash
python benchmark.py --sdpa-backend flash      # FlashAttention 2 forced
python benchmark.py --sdpa-backend efficient  # xFormers-style memory-efficient
python benchmark.py --sdpa-backend cudnn      # cuDNN's fused attention
```

Key API (`benchmark.py:94–102`) — the CLI flag selects an `SDPBackend` and wraps the `pipe(...)` call in a `sdpa_kernel(...)` context manager that restricts dispatch:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
backend = {
    "flash":     SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn":     SDPBackend.CUDNN_ATTENTION,
    "math":      SDPBackend.MATH,
}[args.sdpa_backend]
with sdpa_kernel(backend):
    output = pipe(...)
```

What you're measuring: which attention kernel PyTorch's SDPA dispatcher is *actually* picking, and whether forcing a specific one wins. Stock PyTorch on Ampere+ already auto-dispatches to `flash` — this experiment is mostly *verifying* the default and seeing the small spread between alternatives.

Expected, single 4090:

| config | model load | second call | speedup |
|---|---|---|---|
| baseline (auto) | **5.8 s** | **77 s** | 1× |
| `--sdpa-backend flash` | TBD | TBD | ~1× (expected — already what auto picks) |
| `--sdpa-backend efficient` | TBD | TBD | ~0.98× (expected) |
| `--sdpa-backend cudnn` | TBD | TBD | ~1.02× (expected, sometimes a small Hopper-only win) |

Differences are usually within ±5%. The interesting check is that *all backends produce identical output for the same seed* — they're mathematically equivalent. (SageAttention in lab 4.3 is *not* mathematically equivalent and produces visually-similar-but-not-identical frames; SDPA backends do.)

### Experiment 4 — Offload trade-off

```bash
python benchmark.py --offload model           # whole-component CPU offload
python benchmark.py --offload sequential      # per-submodule CPU offload
```

Key API (`benchmark.py:153–156`) — diffusers ships two offload helpers; the CLI flag picks one:

```python
if args.offload == "model":
    pipe.enable_model_cpu_offload()       # cycle whole modules (text encoder, transformer, VAE)
elif args.offload == "sequential":
    pipe.enable_sequential_cpu_offload()  # cycle individual submodules within each model
```

What you're measuring: how much **peak VRAM** drops when diffusers' offload helpers move idle components to CPU, and how much wall clock you pay for that drop.

Expected, single 4090:

| config | model load | second call | peak VRAM | use case |
|---|---|---|---|---|
| baseline | **5.8 s** | **77 s** | **20.5 GB** | 24 GB+ card |
| `--offload model` | **10.9 s** | **78 s** (+1%) | **17.0 GB** (−17%) | 20 GB card |
| `--offload sequential` | TBD | TBD (+120% expected) | TBD (much lower peak) | 12 GB consumer card |

**Why `--offload model` is essentially free here** (+1% latency vs the ~10–15% you'd expect). The benchmark already does its own manual text-encoder offload (`pipe.text_encoder.to("cpu")` after `encode_prompt`) to avoid an OOM during torch.compile. That's the *expensive* offload — umT5-XXL is ~11 GB. By the time `--offload model` engages, the only remaining cycle work is the transformer ↔ VAE handoff, which is tiny (~3 GB savings, sub-second wall cost). So our measured "+1% slowdown / −17% VRAM" reflects baseline-minus-text-encoder-offload, not raw `enable_model_cpu_offload()`. The trade-off shape still holds — just shifted: each *additional* level of offload costs roughly an order of magnitude more in latency for the next chunk of VRAM. The `--offload sequential` row (per-submodule offload) is where the steep slowdown kicks in.

On a 4090 you don't need offload to fit (baseline is 20.5 GB on a 24 GB card), but the experiment is how you'd plan a deployment to a 12 GB card, or run two models concurrently on the same GPU.

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

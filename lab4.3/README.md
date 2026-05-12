# Module 4.3 — SGLang inference optimization

> Part 4 — DiT in Production · [DiT from Scratch](../README.md)

**Goal**: take WAN inference past what `diffusers` + `torch.compile` reaches (lab 4.2) by adopting a *production inference engine*. The target is **SGLang-Diffusion**, the strongest open-source video-DiT inference runtime as of 2026. Three benchmark categories, each isolating one mechanism:

1. **Baseline** — what `sglang generate` buys you over `diffusers + torch.compile` with default flags (kernel fusion + better attention dispatch).
2. **Kernel sweep** — A/B the attention backend (`fa2` / `fa3` / `sage` / `flashinfer`) on the same prompt. Watch which one wins on your hardware.
3. **Parallelism** — CFG parallel (2 GPUs), Ulysses (head-shard sequence parallelism), Ring (streaming-K/V sequence parallelism), USP (Ulysses+Ring hybrid), tensor parallelism (sharded linear weights).

Each is a flag on `sglang generate`. The lab teaches what each does, what it costs, and when to reach for which.

**Compute**: this lab is **multi-GPU**. The baseline + kernel sweep run on 1 GPU; the parallelism sweep needs ≥2 GPUs to demonstrate anything (Ring / Ulysses comparisons are most informative at ≥4 GPUs). Part 4's compute target — **4× 4090** — is what this lab uses to its full extent.

## Acceleration over lab 4.2

Lab 4.2 demonstrated that vanilla PyTorch + `torch.compile` gets you ~1.6–1.9× over stock `diffusers`. What's left on the table that a production engine adds:

| Category | What `torch.compile` can do | What SGLang adds |
|---|---|---|
| Attention kernel | FlashAttention 2 via SDPA | FlashAttention 3, SageAttention (INT8), FlashInfer RoPE — externally-built hand-written kernels |
| Other fused kernels | Inductor + Triton fusions | Hand-written CUDA kernels for QK-norm, fused gate+up+SiLU, fused QKV — competitive with or faster than Inductor's |
| Sampling-loop caching | No (graph-level only) | DiT-aware feature caching across diffusion steps (Cache-DiT) — ~1.7× alone |
| Multi-GPU parallelism | None | CFG parallel, sequence parallelism (Ring / Ulysses / USP), tensor parallelism — all configurable via flags |

The pattern: SGLang is the *orchestration runtime*. Most of its wins come from composing third-party libraries (FlashAttention, SageAttention, FlashInfer, Cache-DiT, xfuser) into one inference pipeline. Each is independently togglable; the lab teaches you which knob to turn when.

## The technique map

Every optimization in SGLang-Diffusion comes from somewhere. Most are upstream libraries; SGLang is the layer that picks and composes them.

| Technique | Where it comes from | What it does |
|---|---|---|
| **FlashAttention 2/3** | [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) — external | Tiled softmax over Q@Kᵀ that never materializes the full attention matrix. Core LLM-serving primitive. |
| **SageAttention 2/3** | [`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention) — external | Quantized attention (INT8 / FP8 Q/K/V). Cuts attention FLOPs and memory at the cost of <1% quality. |
| **FlashInfer RoPE / kernels** | [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer) — external | Inplace, fused rotary embedding + other custom kernels. Replaces ~5 PyTorch ops with one CUDA kernel. |
| **Fused QKV** | model-adapter pattern, not a library | One `Linear(hidden, 3·hidden)` + split, instead of three `Linear`s. |
| **Fused gate+up+SiLU** (SwiGLU) | usually `flashinfer.silu_and_mul` | One kernel for `silu(gate(x)) * up(x)`. |
| **JIT QK-norm kernel** | in-house Triton / `torch.compile` | Fuses the per-head Q/K RMSNorm that some DiT variants use. |
| **Custom timestep CUDA kernel** | in-house | Sinusoidal `t` embedding written as one CUDA kernel instead of many tiny PyTorch ops. |
| **Cache-DiT** | [`vipshop/cache-dit`](https://github.com/vipshop/cache-dit) — external | DiT-specific feature caching: skip recomputing block outputs that haven't changed across diffusion steps. |
| **Layerwise weight offload** | in-house orchestration | Prefetch layer N+1 onto GPU during layer N's compute. Hides PCIe transfer cost. |
| **Sequence parallelism** (Ulysses + Ring USP) | [`xfuser` / xDiT](https://github.com/xdit-project/xDiT) — external | Shard the long video sequence across GPUs along the sequence dim. |
| **Tensor parallelism** | standard | Shard linear-layer weights across GPUs. |
| **CFG parallel** | in-house | Run cond + uncond forwards on separate GPUs concurrently. |

## Toggle map (the SGLang CLI flags)

What makes this lab teachable: every optimization is independently turn-on-able from the CLI. From the [official CLI docs](https://sgl-project.github.io/diffusion/api/cli.html):

| Optimization | Flag |
|---|---|
| Attention backend | `--attention-backend {fa,sage,xformers,native,_flash_3_hub}` |
| Cache-DiT | `--cache-dit-config <path>` (or `SGLANG_CACHE_DIT_ENABLED=true`) |
| Layerwise weight offload | `--dit-layerwise-offload true` |
| CFG parallelism | `--enable-cfg-parallel` |
| Sequence parallelism | `--sp-degree N`, `--ulysses-degree N`, `--ring-degree N` |
| Tensor parallelism | `--tp-size N` |
| VAE memory | `--vae-tiling`, `--vae-slicing` |
| Text-encoder offload | `--text-encoder-cpu-offload`, `--pin-cpu-memory` |

## Setup

This lab needs **one isolated venv** at `lab4.3/.venv-sglang/`. SGLang ships compiled CUDA kernels (`flashinfer-cubin`, `sgl-kernel`) pinned to a specific torch ABI; installing it into the shared root `.venv` would replace torch/cuDNN and break the diffusers path you spent labs 1.x–4.2 setting up. Keep sglang in its own venv.

```
dit-from-scratch/
├── .venv/                 ← shared root venv (labs 1.x–4.6 diffusers)
└── lab4.3/
    └── .venv-sglang/      ← only sglang lives here
```

```bash
# From the repo root, create the isolated sglang venv. uv is recommended
# (the install needs --prerelease=allow, which pip doesn't support as a flag):
cd lab4.3
python3 -m venv .venv-sglang
source .venv-sglang/bin/activate
pip install --upgrade pip uv
uv pip install "sglang[diffusion]" --prerelease=allow
deactivate
cd ..
```

Verify:

```bash
lab4.3/.venv-sglang/bin/sglang generate --help   # should print the SGLang CLI flags
```

The benchmark scripts below activate the shared root `.venv` for their orchestration code, then invoke `sglang generate` as a subprocess against `.venv-sglang/bin/sglang`. They never import SGLang into the diffusers venv.

## Run the benchmarks

`benchmark_baseline.py` / `benchmark_kernels.py` / `benchmark_parallel.py` each run **one config per invocation** and time **two inference calls back-to-back** — `first call` (includes any sglang warmup: kernel autotune, weight prefetch, optional CUDA Graph capture) and `second call` (warm steady-state). Each invocation reloads the model; accepted as the cost of treating every config as a fresh run. The scripts shell out to `lab4.3/.venv-sglang/bin/sglang generate` for each row — your shell stays in the diffusers venv. Activate it from the repo root first:

```bash
source .venv/bin/activate
cd lab4.3
```

### Baseline

```bash
python benchmark_baseline.py --prompt "a fluffy red panda eating bamboo on a tree branch"
```

Compares stock diffusers (lab 4.2's baseline number) against `sglang generate` with default flags. The win comes from FlashAttention-2 (auto-selected), hand-written QKV / SwiGLU / QK-norm fusions, and SGLang's scheduler overhead being lower than `WanPipeline.__call__`.

Expected, single 4090:

| config | model load | first call | second call | speedup (second) | peak VRAM |
|---|---|---|---|---|---|
| diffusers baseline (lab 4.2) | 5.8 s | 77 s | 77 s | 1× | 20.5 GB |
| sglang default | ~? s | ~? s | ~? s | ~2.5–3× | ~? GB |

### Experiment 1 — attention backend

```bash
python benchmark_kernels.py --prompt "..."     # sweeps all backends
```

Underneath, each row of the sweep is one `sglang generate` invocation with a different `--attention-backend`:

```bash
sglang generate --attention-backend fa            # FlashAttention 2 (default)
sglang generate --attention-backend _flash_3_hub  # FlashAttention 3 (Hopper-only)
sglang generate --attention-backend sage          # SageAttention (INT8 Q/K/V)
sglang generate --attention-backend xformers      # xFormers memory-efficient
sglang generate --attention-backend native        # PyTorch SDPA (no backend hint)
```

The dominant kernel in WAN's transformer (and any video DiT) is the attention `Q @ Kᵀ → softmax → · V` for the long video sequence. Lab 4.2's diffusers baseline reaches FA2 via SDPA dispatch; SGLang exposes the choice as a flag.

Expected, single 4090:

| config | model load | first call | second call | speedup (second) | peak VRAM |
|---|---|---|---|---|---|
| sglang default (= `fa`) | ~? s | ~? s | ~? s | 1× | ~? GB |
| `--attention-backend fa` | ~? s | ~? s | ~? s | ~1.00× | ~? GB |
| `--attention-backend _flash_3_hub` | ~? s | n/a | n/a | n/a (Hopper-only) | n/a |
| `--attention-backend sage` | ~? s | ~? s | ~? s | ~1.1–1.3× | ~? GB |
| `--attention-backend xformers` | ~? s | ~? s | ~? s | ~0.95–1.00× | ~? GB |
| `--attention-backend native` | ~? s | ~? s | ~? s | ~0.95–1.00× | ~? GB |

What each one does:

- **FA2 (`fa`)** — the standard tiled-softmax kernel. CUDA / SM80+. Same kernel SDPA dispatches to by default in PyTorch.
- **FA3 (`_flash_3_hub`)** — Hopper-specific rewrite. Async TMA (Tensor Memory Accelerator) for loads + WGMMA for matmul, overlapping memcpy and compute via warp specialization. Strict speedup over FA2 on H100; no-op-or-worse on older GPUs. The `_hub` suffix means SGLang pulls a prebuilt FA3 binary from a HuggingFace cubin hub — don't try to build FA3 from source.
- **SageAttention (`sage`)** — quantizes Q/K/V to INT8 before the matmul. ~2× FLOPs reduction in the attention step. Wall-clock win is smaller (~1.1–1.3×) because attention isn't the whole forward. **Output is not bit-identical to FA** — quality drop is typically <1% on standard benchmarks, but check visually for your specific prompts before committing.
- **xFormers / native** — alternative implementations, included for compatibility / debugging. Usually within ±5% of FA2 wall clock.

Attention dominates the transformer's wall clock at video sequence lengths (~30–50% of forward time at 49 frames × 832×480). A 1.4× attention speedup is ~1.15× end-to-end. Not enormous, but adds to the other levers.

#### Deep dive: why diffusers + `torch.compile` can't reach this

`torch.compile` uses Inductor + Triton. Its kernels are *Triton-generated*, not hand-tuned C++/CUDA. For most ops (linears, norms, elementwise) the generated kernels are near-optimal. For attention specifically, the hand-tuned FA2/FA3 kernels beat Inductor by a non-trivial margin, and SageAttention's INT8 path isn't reachable from Triton at all.

SGLang's value is that it *picks the right kernel per op* — Inductor for the things Inductor is good at, FlashAttention / SageAttention / FlashInfer for the things they're good at — and orchestrates them in one inference pipeline.

### Experiment 2 — CFG parallel (2 GPUs)

```bash
python benchmark_parallel.py --config cfg-parallel --prompt "..."
# Underneath:
sglang generate --enable-cfg-parallel --prompt "..."
```

The classifier-free-guidance step needs **two transformer forwards per sampling step** — one with the prompt, one with the negative prompt — and the two are independent until the extrapolation. Put one branch on each of two GPUs and run them concurrently:

```
single-GPU CFG (serial):                  CFG parallel (concurrent):

   transformer(prompt)        ──►          GPU 0: transformer(prompt)    ┐  concurrent
   ↓ wait                                                                │
   transformer(neg)           ──►          GPU 1: transformer(neg)       ┘
   ↓                                            ↓
   v_uncond + s·(v_cond − v_uncond)             gather → v_uncond + s·(v_cond − v_uncond)
```

Wall clock per step drops from ~2T to ~T (where T is one forward), at the memory cost of a second transformer copy.

Expected, 2× 4090:

| config | model load | first call | second call | speedup (second) | peak VRAM (per GPU) |
|---|---|---|---|---|---|
| sglang default (1 GPU) | ~? s | ~? s | ~? s | 1× | ~? GB |
| `--enable-cfg-parallel` (2 GPUs) | ~? s | ~? s | ~? s | ~1.8× | ~? GB (~2× per-GPU footprint) |

**~1.8× speedup, ~2× memory.**

#### Deep dive: what `--enable-cfg-parallel` does internally

The pattern in plain PyTorch — what SGLang's runtime composes around your prompt and the model:

```python
# Two transformer copies (deepcopy, one per GPU)
transformer_0 = pipe.transformer.to("cuda:0")
transformer_1 = copy.deepcopy(pipe.transformer).to("cuda:1")

# Inside the sampling loop, per step:
z_t_1 = z_t.to("cuda:1", non_blocking=True)
cond   = transformer_0(z_t,   t, encoder_hidden_states=prompt_embeds_0)[0]
uncond_remote = transformer_1(z_t_1, t, encoder_hidden_states=neg_embeds_1)[0]
uncond = uncond_remote.to("cuda:0")                          # implicit sync
noise  = uncond + guidance * (cond - uncond)
```

The two `transformer_*(...)` calls dispatch to different CUDA streams on different devices and return immediately (CUDA is async). The actual GPU compute runs **in parallel**. The `.to("cuda:0")` on the unconditional output is what synchronizes — it blocks until cuda:1's transformer finishes, then memcpies the result.

No `torchrun`, no `torch.distributed`, no NCCL — just two CUDA contexts coordinated through PyTorch's stream model. SGLang's `--enable-cfg-parallel` adds production touches (text encoder offload after encoding, error handling when the two GPUs go out of sync, composability with USP / TP).

#### Deep dive: why this isn't in diffusers

`WanPipeline.__call__` is structured as one sampling loop calling `pipe.transformer` twice per step. CFG-parallel needs a different sampling loop that dispatches the two calls to separate devices — not a flag, a rewrite. Lab 4.2's `torch.compile(reduce-overhead)` pitfall is the related issue: CUDA Graphs alias the two consecutive calls' output buffers. SGLang's runtime owns the sampling loop, so it can structure it correctly.

### Experiment 3 — sequence parallelism (≥2 GPUs)

```bash
python benchmark_parallel.py --config ulysses-4 --prompt "..."
python benchmark_parallel.py --config ring-4    --prompt "..."
python benchmark_parallel.py --config usp-2x2   --prompt "..."
```

Underneath:

```bash
sglang generate --sp-degree 4 --ulysses-degree 4 --ring-degree 1   # pure Ulysses
sglang generate --sp-degree 4 --ulysses-degree 1 --ring-degree 4   # pure Ring
sglang generate --sp-degree 4 --ulysses-degree 2 --ring-degree 2   # USP hybrid
```

Why video DiT specifically needs sequence parallelism: at 81 frames × 720p, the patchified token count is millions. The attention `Q @ Kᵀ` matrix at that length doesn't fit a single GPU's memory regardless of any other optimization. Image DiT hits ~16k tokens at most and never needs sequence sharding; video forces it.

Lab 4.1's WAN 2.1 T2V-1.3B at 832×480 × 49 frames runs fine on one GPU. The benchmarks below demonstrate the *technique* at this smaller scale; production WAN 2.2 14B at 1280×720 × 81 frames is where SP becomes mandatory rather than nice-to-have.

Expected, 4× 4090:

| config | model load | first call | second call | speedup (second) | peak VRAM (per GPU) |
|---|---|---|---|---|---|
| sglang default (1 GPU) | ~? s | ~? s | ~? s | 1× | ~? GB |
| Ulysses-4 (`--ulysses-degree 4 --ring-degree 1`) | ~? s | ~? s | ~? s | ~? × | ~? GB |
| Ring-4 (`--ulysses-degree 1 --ring-degree 4`) | ~? s | ~? s | ~? s | ~? × | ~? GB |
| USP 2×2 (`--ulysses-degree 2 --ring-degree 2`) | ~? s | ~? s | ~? s | ~? × | ~? GB |

At this scale (1.3B model, 480p video, 4× 4090 without NVLink), all-to-all bandwidth is PCIe-limited so **Ring tends to win over Ulysses**. At production scales (14B + 720p × 81 frames on NVLinked H100s), **USP hybrid wins**, and it's the difference between "fits" and "doesn't fit." All configurations should produce **identical output** for the same seed (parallelism is mathematically equivalent to single-GPU).

#### Deep dive: Ring Attention — shard sequence, stream K/V

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

**Communication overlaps with compute** — while GPU `i` computes attention with the current K/V chunk, the next chunk is in-flight from GPU `i+1`. Bandwidth-efficient; latency-tolerant; favors inter-node IB-bottlenecked setups.

#### Deep dive: Ulysses Attention — shard heads, all-to-all

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

Communication: **2× all-to-all per attention layer**. Lower latency than Ring (`2` rounds vs `N` rounds), but consumes more total bandwidth at large `N`. Favors NVLink-rich intra-node setups (~900 GB/s makes all-to-all cheap).

#### Deep dive: USP — combine both

xDiT's contribution: shard the sequence in *two* dimensions. Use Ring across many GPUs (good bandwidth utilization) and Ulysses *within* a small Ring group (low latency for the inner step).

```
8-GPU example:  --ulysses-degree 2 --ring-degree 4

  Inner: 2-way Ulysses (head-shard via all-to-all)
  Outer: 4-way Ring   (sequence-shard via streaming K/V)
```

Tuning rule of thumb: **Ulysses degree should match NVLink topology** (NVLink-connected pairs do the all-to-all efficiently); **Ring degree spans across NVLink boundaries** (higher latency tolerance there).

#### Deep dive: diffusion-specific complications

1. **CFG batching.** Conditional and unconditional forwards run as one `(2·B, ...)` batch. Sequence parallelism has to be CFG-aware so the all-to-all doesn't shuffle the conditional samples into the unconditional ones.
2. **3D RoPE positions.** Sharded sequences need consistent `(t, h, w)` indices per token — each shard must know which slice of the global position grid it owns. xDiT handles this; if you write your own SP, it's the bug to watch for.
3. **Cross-attention K/V are tiny** (77 text tokens × hidden). **Don't** sequence-shard them — replicate per GPU. They're cheap.

### Experiment 4 — tensor parallelism (2 GPUs)

```bash
python benchmark_parallel.py --config tp-2 --prompt "..."
# Underneath:
sglang generate --tp-size 2 --prompt "..."
```

`--tp-size N` shards the transformer's linear-layer weights across N GPUs. Useful when the model itself doesn't fit on one card — Wan-2.2 14B at bf16 is ~28 GB and won't fit on a single 24 GB 4090, but `--tp-size 2` cleanly halves the weight memory.

Expected, 2× 4090:

| config | model load | first call | second call | speedup (second) | peak VRAM (per GPU) |
|---|---|---|---|---|---|
| sglang default (1 GPU) | ~? s | ~? s | ~? s | 1× | ~? GB |
| `--tp-size 2` | ~? s | ~? s | ~? s | ~0.7–0.9× | ~? GB (~½ of baseline) |

TP introduces per-layer all-reduces (after each split linear) that don't overlap with compute as cleanly as sequence parallelism's communication. For models that *do* fit on one GPU, TP costs throughput; sequence parallelism is the right tool there. For models that don't fit, TP is necessary.

For Wan-2.1 T2V-1.3B (this lab's model) TP isn't needed — it fits comfortably on a 24 GB card. The benchmark runs `--tp-size 2` here for completeness on the curriculum's 4090 quartet; you'd reach for it seriously when running Wan-2.2 14B on smaller cards.

## Files

| File | What it is |
| --- | --- |
| `benchmark_baseline.py` | Compares stock diffusers vs `sglang generate` with default flags. ~3× win on a single GPU expected. |
| `benchmark_kernels.py` | A/B's the attention backend on the same prompt+seed: `fa` / `_flash_3_hub` / `sage` / `xformers` / `native`. |
| `benchmark_parallel.py` | A/B's parallelism configurations: CFG-parallel, Ulysses, Ring, USP hybrid, tensor-parallel. Auto-skips configs needing more GPUs than available. |
| `requirements.txt` | Tiny — just what the orchestrating scripts need; sglang itself lives in `.venv-sglang/`. |

## Discussion

### Comparison of inference frameworks

| Framework | Shape | Best for |
|---|---|---|
| **SGLang-Diffusion** (this lab) | full engine: kernel composition + multi-backend attn + Cache-DiT + USP | Most thorough open-source production engine for video DiT; deepest single-GPU and multi-GPU wins. |
| **diffusers** + `torch.compile` (lab 4.2) | reference + JIT compiler | Hackable baseline; ~1.5–1.9× over plain diffusers. Stops there. |
| **[xDiT / xfuser](https://github.com/xdit-project/xDiT)** | parallelism library | SP/TP wrappers you layer onto stock diffusers. SGLang itself depends on xfuser for its sequence-parallel layer. |
| **[OneDiff](https://github.com/siliconflow/onediff)** | graph compiler, drop-in for diffusers | One-line speedup without rewriting kernels. Less aggressive than SGLang. |
| **[TensorRT-LLM-Diffusion](https://github.com/NVIDIA/TensorRT-LLM)** | AOT graph compiler, NVIDIA-blessed | Single-tenant H100 / B200 deployments. Closed-vendor tooling. |
| **[vLLM-Diffusion](https://github.com/vllm-project/vllm)** | continuous-batching engine (LLM heritage) | Batch-serving; less mature for video DiT as of 2026. |

### What SGLang isn't

- **Not a model**. It's an inference engine. The model weights are still WAN's.
- **Not a training framework**. For training, lab 4.4 (LoRA) and 4.5 (full SFT) use `accelerate` + `bitsandbytes` + `peft` directly. Some kernels overlap; the orchestration is different.

### Training vs inference

Almost everything in this lab is inference-only. Two crossover techniques:

- **FlashAttention** — works in training too (and `WanPipeline`'s training flavor in `diffusers` uses FA2 under the hood when available). Lab 4.5's full-SFT training script benefits from the same kernel.
- **Sequence parallelism** — also works in training, via `xfuser` / DeepSpeed-Ulysses. Used to train long-video models that don't fit on one GPU. Lab 4.5 mentions this in passing; the actual training-side parallelism story is in [`EFFICIENCY.md`](../EFFICIENCY.md).

Techniques that don't transfer to training:

- **Cache-DiT** — caching only works when steps converge to similar outputs, which doesn't apply during training where every step changes the loss landscape.
- **CFG parallel** — there's no CFG at training time; the model learns from `(image, caption)` pairs without the cond/uncond split.
- **Quantized attention (SageAttention)** — inference-only at production quality. FP8 *training* is reaching production but mostly for LLMs, not diffusion (yet).

### Where to go deeper

- **[SGLang-Diffusion docs](https://sgl-project.github.io/diffusion/)** — full CLI reference, technique map, benchmark blog posts.
- **[`xfuser` / xDiT source](https://github.com/xdit-project/xDiT)** — the sequence-parallel implementation SGLang depends on.
- **[`Wan-Video/Wan2.2/wan/distributed/`](https://github.com/Wan-Video/Wan2.2/tree/main/wan/distributed)** — a production team's adaptation of xfuser; smaller surface area than xfuser itself.
- **[Ring Attention paper (Liu et al. 2023)](https://arxiv.org/abs/2310.01889)** — the streaming-K/V scheme.
- **[DeepSpeed-Ulysses paper (Jacobs et al. 2023)](https://arxiv.org/abs/2309.14509)** — the all-to-all scheme.
- **[FlashAttention-3 (Shah et al. 2024)](https://arxiv.org/abs/2407.08608)** — Hopper-async pipelining.
- **[`EFFICIENCY.md`](../EFFICIENCY.md)** — the curriculum's curated reading list for GPU-eng efficiency work, covering both training and inference.

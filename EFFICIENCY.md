# Training & inference efficiency for video DiT — further reading

> Read this after Parts 1–4 of [DiT from Scratch](./README.md).

## Foundational mental models

Read these first if you don't already have a working model of multi-GPU parallelism and activation memory. They underpin everything below.

- **[Megatron-LM (Shoeybi et al. 2019)](https://arxiv.org/abs/1909.08053)** and **[Megatron-LM v3 (Korthikanti et al. 2022)](https://arxiv.org/abs/2205.05198)** — the canonical reference for 3D parallelism (data + tensor + pipeline). The v3 paper introduces *selective activation recomputation*, which is what every production team actually runs (full recomputation costs 30–40% throughput, no recomputation OOMs).
- **[ZeRO (Rajbhandari et al. 2019)](https://arxiv.org/abs/1910.02054)** and follow-ups — defines the optimizer/gradient/parameter sharding spectrum (ZeRO-1/2/3). PyTorch's FSDP-2 is essentially ZeRO-3 with a cleaner API.
- **[Reducing Activation Recomputation in Large Transformer Models (Korthikanti et al. 2022)](https://arxiv.org/abs/2205.05198)** — the selective-recomputation paper specifically. Applies directly to DiT blocks.

---

## Training infrastructure

**Where to start, ranked by leverage for a video-DiT HPC engineer:**

1. **FSDP-2 wrapping + sharding strategy** — biggest single throughput lever. Tuning wrapping policy + reduce-scatter overlap is most of the work.
2. **Sequence parallelism (Ring / Ulysses) + how it interacts with 3D RoPE** — the video-specific story. At long clips you *cannot* train without sequence sharding regardless of FSDP.
3. **VAE latent precomputation** — biggest *infrastructure* lever; an easy 25–40% throughput win that doesn't require kernel work.
4. **Selective activation checkpointing** — fits between (1) and (5); needed for any model >7 B.
5. **Distributed checkpointing** — boring but non-skippable at 14 B+ params.

Skip the inference acceleration sections below until you're solid on the parallelism layer above — TeaCache / KV-caching / quantization are optimizations on top of a working training stack, not the stack itself.

### Sharding & parallelism

- **[PyTorch FSDP-2 docs](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)** — the modern (compose-with-everything) fully-sharded data parallel API. Replaces FSDP-1. Read the wrapping-policy section carefully; it's where 80% of throughput tuning happens.
- **[Ring Attention (Liu et al. 2023)](https://arxiv.org/abs/2310.01889)** — sequence-parallel attention that shards the sequence dimension across GPUs. The video-DiT killer feature: at 81 frames × 480p, sequence length × hidden_dim exceeds single-GPU activation memory regardless of FSDP. Ring Attention is how you train it.
- **[DeepSpeed-Ulysses (Jacobs et al. 2023)](https://arxiv.org/abs/2309.14509)** — alternative sequence-parallel scheme; different all-to-all communication pattern than Ring. Both ship in production frameworks.
- **[`xDiT` / `xfuser`](https://github.com/xdit-project/xDiT)** — the open-source reference implementation of sequence + tensor parallelism for diffusion transformers. **Used in production by Wan-Video, FLUX deployments, and others.** Read the source — the parallelism primitives are well-organized and the diffusion-specific tricks (CFG-batched sequence parallel, etc.) are explicit.
- **[Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2/tree/main/wan/distributed)** — their `distributed/` subdirectory shows how a production team wires xfuser into the training and inference paths. Smaller code surface than xfuser itself; good as a "minimal real-world consumer" example.

### Activation memory

- **[`torch.utils.checkpoint`](https://pytorch.org/docs/stable/checkpoint.html)** — selective vs full recomputation in PyTorch. Use `use_reentrant=False`.
- **[Selective Activation Recomputation in PyTorch](https://pytorch.org/docs/stable/distributed.algorithms._checkpoint.checkpoint_wrapper.html)** — `apply_activation_checkpointing` for FSDP-wrapped modules; the production pattern.

### Mixed precision

- **[NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)** — fp8 training/inference primitives for Hopper/Blackwell. Hybrid bf16-fp8 paths (matmuls in fp8, accumulators / norms / softmax in bf16) — what production teams are starting to deploy in 2025–26.
- **[`torch.amp` / autocast docs](https://pytorch.org/docs/stable/amp.html)** — bf16 autocast caveats. The non-obvious ops that need fp32 (reductions, AdaLN's affine, MSE accumulation) are listed here.
- **[Mixed Precision Training (Micikevicius et al. 2017)](https://arxiv.org/abs/1710.03740)** — original mixed-precision paper. fp16 era, but the loss-scaling and master-weights mental models still apply when you debug bf16 instabilities.

### Data pipeline

- **[WebDataset](https://github.com/webdataset/webdataset)** — tar-shard streaming dataset format that production video-DiT teams use. Encodes well to S3/object stores, parallelizes cleanly, plays nicely with FSDP's distributed dataloader. The whole "encode VAE latents once, train DiT for many epochs" pipeline ships these as shards.
- **[Wan-Video/Wan2.2 dataset prep](https://github.com/Wan-Video/Wan2.2)** — search for their latent-caching / data-preparation scripts; production-grade reference for "encode the dataset's videos through 3D causal VAE once, store latents." 25–40% of training step time disappears when you do this well.
- **Multi-aspect-ratio / multi-resolution training** — there's no canonical paper, but [SDXL's report (Podell et al. 2023, §2.3)](https://arxiv.org/abs/2307.01952) describes the bucketing approach (group clips by similar shape, rotate buckets between batches) that every modern image/video DiT training pipeline uses. Read SDXL's appendix even if you only care about video — the mechanism transfers directly.
- **Sequence packing for variable-length clips** — bucketing pads each batch to the max shape in its bucket; *packing* goes further by concatenating multiple short clips into one sequence with attention masks that prevent cross-clip attention, so no padding compute is wasted. Subtle for video DiT because 3D RoPE positions need to *reset* at each clip boundary in the packed sequence. There's no canonical reference; read [`musubi-tuner`](https://github.com/kohya-ss/musubi-tuner)'s data pipeline and [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)'s training data utilities for two real implementations.

### Distributed checkpointing

- **[`torch.distributed.checkpoint`](https://pytorch.org/docs/stable/distributed.checkpoint.html)** — sharded save/load for FSDP-trained models. Without it, a 14B-param checkpoint takes minutes from rank 0; with it, parallel writers save in seconds. Boring but unavoidable at scale.

---

## Inference acceleration

The runtime layer is GPU-engineer territory: compilation, caching infrastructure, quantization runtime, multi-GPU inference. The *algorithmic* layer (deciding which features to cache across steps, recipe-level scheduler choices) is ML-engineering work — you cooperate with it but you don't own the recipe. Subsections below are split accordingly; spend the bulk of your time on the first half.

### Runtime — what GPU eng owns

#### Compilation & graph capture

- **[`torch.compile` modes](https://pytorch.org/docs/stable/torch.compiler.html)** — `default` vs `reduce-overhead` vs `max-autotune`. For diffusion sampling loops, `reduce-overhead` (CUDA graphs under the hood) is usually the right choice; `max-autotune` is for "I'm willing to wait 10 minutes for autotuning to save 5% per step."
- **[CUDA graphs guide (PyTorch)](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)** — diffusion sampling loops are *the* canonical use case for graph capture: same model forward called N times with only `t` changing. Pure runtime work, no model modification.

#### Cross-attention KV caching

- **Cross-attention KV caching** — diffusion has no causal KV cache (non-autoregressive), but the *cross-attention* keys/values from text tokens don't change across sampling steps. Cache them once. Most production pipelines do this; the diffusers source (`WanPipeline.__call__`) is the reference implementation. Pure runtime change — no model modification, no ML-side decision.

#### Quantization runtime

- **[`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes)** — NF4 / int8 / fp4. The HF/diffusers default. CUDA-only.
- **[`torchao`](https://github.com/pytorch/ao)** — Meta's modern alternative. Better `torch.compile` interop, native PyTorch (no separate ext). int8, fp8, and weight-only paths.
- **[SmoothQuant (Xiao et al. 2022)](https://arxiv.org/abs/2211.10438)** — activation-aware W8A8. The technique most production int8 deployments use; key insight is that activation outliers concentrate in a few channels and you can pre-shift them into the weights.
- **[AWQ (Lin et al. 2023)](https://arxiv.org/abs/2306.00978)** — activation-aware W4A16. The 4-bit-weight equivalent of SmoothQuant; what most LLM W4A16 deployments use. Less battle-tested for diffusion/video but the math is general.
- **[LLM.int8() (Dettmers et al. 2022)](https://arxiv.org/abs/2208.07339)** — the original "8-bit inference for transformers" paper. Read for the outlier-feature mental model that informs SmoothQuant / AWQ.

#### Multi-GPU inference

Same `xfuser` / Wan2.2 distributed code referenced in the training section is also what runs production multi-GPU inference for the larger WAN variants (Wan-2.2 14B doesn't fit on a single 80 GB GPU at full resolution). Sequence parallelism splits the very long video sequence across N GPUs at inference time; you implement the comm overlap, ML doesn't touch this.

### ML-team-owned algorithm changes (context only, not your day job)

These design recipe-level decisions about what features are temporally coherent enough to cache (TeaCache / DeepCache). You implement the runtime hooks ML asks for; you don't pick the recipe. Read once for awareness, deprioritize against the runtime work above.

#### Step / feature caching

- **[DeepCache (Ma et al. 2023)](https://arxiv.org/abs/2312.00858)** — cache block features across denoising steps.
- **[TeaCache (Liu et al. 2024)](https://arxiv.org/abs/2411.19108)** — DiT-specific timestep-aware caching; tested on WAN.

---

## Production codebases worth reading

Reading actual production code is often higher-leverage than papers for an HPC engineer. These are the open repos serious teams ship from in 2026:

- **[`Wan-Video/Wan2.2`](https://github.com/Wan-Video/Wan2.2)** — official training + inference. Read the `distributed/` directory, the train script, and the data prep utilities.
- **[`xDiT` / `xfuser`](https://github.com/xdit-project/xDiT)** — production-grade parallel inference for diffusion transformers. **The reference for sequence-parallel diffusion.** Used by Wan-Video and adjacent video-DiT teams.
- **[`musubi-tuner`](https://github.com/kohya-ss/musubi-tuner)** — community LoRA trainer for WAN / Hunyuan-Video / LTX. Heavily optimized — gradient checkpointing modes, custom mixed-precision paths, latent caching pipelines. Well-commented; good for "how does a single-author project squeeze every last bit of throughput out."
- **[`diffusion-pipe`](https://github.com/tdrussell/diffusion-pipe)** — alternative WAN trainer with different parallelism choices. Worth reading alongside musubi-tuner for the comparison.
- **[`huggingface/diffusers` — Wan source](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/wan)** — the canonical Python implementation of the WAN pipeline. Reference for what "correct" looks like; everyone optimizes from this baseline.
- **[`huggingface/accelerate` source](https://github.com/huggingface/accelerate)** — the FSDP/DeepSpeed wrapper that most public training scripts use. Read at least the `prepare()` flow once; it's the layer between your code and the parallelism backends.

---

## What's not on this list (deliberate omissions)

- LLM-specific tricks that don't transfer (causal KV cache, speculative decoding) — diffusion is non-autoregressive.
- Autograd / lower-level kernel writing (Triton, CUTLASS) — important if you write kernels, but the curriculum's audience is one level up.
- Backend / API serving (Triton Inference Server, vLLM, Modal, Replicate). Per request — explicitly out of scope.
- Image-only diffusion optimizations (no T axis) — most carry over to video, but you'd find them via the video sources anyway.

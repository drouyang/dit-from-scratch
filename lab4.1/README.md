# Module 4.1 — WAN 2.2 inference + code tour

**Goal**: load **WAN 2.2 TI2V-5B**, generate a 5-second video from a text prompt, and trace every component of the running system back to a lab from Parts 1–3. By the end you can open `Wan-Video/Wan2.2`'s source and recognize patchify, AdaLN-Zero, RoPE, cross-attention, flow matching, CFG — exactly the building blocks you wrote yourself, just extended to one more dimension.

## Compute reality

This is the first lab where M3 isn't the primary target. WAN 2.2 TI2V-5B is a 5B-param video DiT and the validated paths all need a real GPU.

| Path | Hardware | Notes |
|---|---|---|
| **diffusers `WanPipeline`** | ~24GB VRAM (4090, A100, H100) | Cleanest API; recommended for the code tour |
| **Official `Wan-Video/Wan2.2`** | ~24GB VRAM with `--offload_model --t5_cpu` | Production codebase; what the call-path tour points into |
| **ComfyUI + GGUF (community)** | 8GB VRAM, MPS-compatible | Q3_K / Q4_K quantizations runnable on M3, but third-party and significantly slower. No official support. |

For the read-and-trace exercise (the meat of this lab), only the source code matters — you can do that on M3 without running anything. For actually generating a video, plan on a rented 4090 / H100 for an hour, or pull the GGUF in ComfyUI on M3 if you accept the speed penalty.

## Files

| File | What it is |
|---|---|
| `inference_diffusers.py` | Thin wrapper around diffusers' `WanPipeline` — text prompt → mp4. ~50 lines. |
| `requirements.txt` | `torch`, `diffusers` (from source), `accelerate`, `transformers`, `imageio[ffmpeg]` |

## Setup

### diffusers path (recommended for reading)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pulls diffusers from main because `WanPipeline` isn't in any stable release yet. First inference run downloads the TI2V-5B checkpoint (~10GB) into the HF cache.

### Official Wan2.2 path (recommended for understanding production code)

```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

## Run inference

### Path A: diffusers

```bash
python inference_diffusers.py --prompt "a fluffy red panda eating bamboo on a tree branch"
```

Output: `out.mp4` — 121 frames at 1280×704, 24 fps (≈5 seconds). Generation takes ~9 minutes on a 4090.

### Path B: official repo

```bash
cd Wan2.2
python generate.py --task ti2v-5B --size 1280*704 \
    --ckpt_dir ./Wan2.2-TI2V-5B \
    --offload_model True --convert_model_dtype --t5_cpu \
    --prompt "a fluffy red panda eating bamboo on a tree branch"
```

The flags: `--offload_model True` spills the model to CPU when not in active use; `--t5_cpu` keeps the umT5-XXL text encoder on CPU (~11B params); `--convert_model_dtype` casts weights to the target dtype. These bring the GPU footprint to ~24GB.

## Concept-organized code tour

The lab's main deliverable. Every production component → call path → file:function in `Wan-Video/Wan2.2`, with which lab introduced it.

### Entry call path

```
generate.py  (--task ti2v-5B)
   └─► wan/textimage2video.py :: WanTI2V.generate()
          ├─► wan/modules/t5.py :: T5EncoderModel(...)              ← text encoding
          │      (tokenizer in wan/modules/tokenizers.py)
          ├─► torch.randn(...)                                       ← initial latent z_T
          ├─► sampling loop (FlowUniPCMultistepScheduler):
          │      └─► wan/modules/model.py :: WanModel(z_t, t, c)    ← DiT forward
          │             ├─► WanAttentionBlock                         ← block (×N)
          │             │      ├─► WanSelfAttention + RoPE-3D         ← image self-attn
          │             │      └─► WanCrossAttention                  ← attends to text
          │             └─► (returns predicted velocity)
          └─► wan/modules/vae2_2.py :: Wan2_2_VAE.decode(z_0)        ← video output
```

This is structurally identical to lab 3.2's `sample.py`. Only the modalities, resolutions, and parameter counts changed.

### Component map

| WAN component | File:function | Lab where you built it | Same as the lab | What's new |
|---|---|---|---|---|
| `WanAttentionBlock` | `wan/modules/model.py` | lab 3.1 (`DiTBlock`), lab 3.2 (added cross-attn) | LN → attn → +res structure, AdaLN-Zero modulation, gate, MLP sublayer | RoPE-3D applied inside attn |
| `WanSelfAttention` (with RoPE) | `wan/modules/model.py` | lab 3.1 (`Attention`) | Multi-head self-attention with RoPE on Q/K | RoPE has 3 axes (t, h, w) instead of 2 |
| `WanCrossAttention` | `wan/modules/model.py` | lab 3.2 (`CrossAttention`) | Image queries attend to text K/V | Larger text encoder (umT5-XXL); same op |
| `rope_params()`, `rope_apply()` | `wan/modules/model.py` | lab 3.1 (`rope_freqs`, `apply_rope`) | Same rotation-by-position trick | 3 frequency bands (t, h, w), concatenated |
| 3D patchify | `wan/modules/model.py` | lab 3.1 (Conv2d patchify) | Conv-based tokenization | `Conv3d` with patch shape `(p_t, p_h, p_w)` |
| `unpatchify()` | `wan/modules/model.py` | lab 3.1 (`unpatchify`) | Reverse of patchify | 3D rearrangement |
| `Wan2_2_VAE.encode/decode` | `wan/modules/vae2_2.py` | lab 3.2 (SD-VAE) | Latent diffusion: encode pixels → latent | 3D causal: compresses time + space; latent shape `(C, T, H, W)` |
| `T5EncoderModel` | `wan/modules/t5.py` | lab 3.2 (CLIP text encoder) | Pretrained, frozen, returns per-token features | umT5-XXL is ~100× larger; richer language understanding |
| `FlowUniPCMultistepScheduler` | imported into `textimage2video.py` | lab 2.2 (Euler ODE) | Flow-matching sampler | Higher-order multi-step solver, fewer steps for the same quality |
| AdaLN-Zero modulation | `wan/modules/model.py` | lab 3.1 (`adaLN_modulation`) | `c → SiLU → Linear` decoded into shift/scale/gate per block | Identical mechanism |
| CFG | `wan/textimage2video.py` | lab 2.2 / 3.2 | `v_uncond + s · (v_cond − v_uncond)` | Identical formula |

If everything in this table maps cleanly back to a lab, you have a complete mental model of WAN. The remaining "new" stuff is purely dimensional extensions, documented in the parent README's *From image to video — what changes architecturally* section.

### What's actually new (the 3D extensions)

Three things are *not* in lab 3.2:

1. **3D causal VAE** (`Wan2_2_VAE`) — encodes a sequence of frames into a `(C, T, H, W)` latent, with causal masking in time so later frames don't peek at earlier ones during reconstruction. Lab 3.2's SD-VAE was 2D image-only.
2. **3D patchify** — tokenize across time as well as space; patch shape `(p_t, p_h, p_w)` produces tokens of dimension `p_t · p_h · p_w · C`. The implementation is just `Conv3d` instead of `Conv2d`.
3. **3D RoPE** — three frequency bands, one per axis (t, h, w). Construction is the same as lab 3.1's 2D RoPE, just with one more direction; rotated halves are concatenated.

DiT block, AdaLN-Zero, cross-attention, flow matching, CFG — all unchanged from lab 3.2.

## Self-check

After reading the tour:

- [ ] Open `wan/modules/model.py` and find `WanAttentionBlock`. Confirm you can map every line of its forward pass to either lab 3.1 (self-attn / AdaLN / MLP) or lab 3.2 (cross-attn).
- [ ] Open `wan/modules/vae2_2.py` and identify the `Conv3d` layers — that's the 3D causal extension over SD-VAE's 2D `Conv2d`.
- [ ] Find `rope_apply()` and confirm three concatenated rotation bands. Compare against lab 3.1's `apply_rope()` (which has two: row + column).

If you can do these three things, you've completed the lab.

## What you've now built end-to-end

```
Part 1: MLP, attention, transformer block          (lab 1.1, 1.3, 1.4)
Part 2: VAE, flow matching, CFG                    (lab 2.1, 2.2)
Part 3: DiT architecture (lab 3.1) → text-to-image (lab 3.2)
Part 4.1: same recipe in production at scale       (WAN 2.2 video — this lab)
```

Lab 4.2 will fine-tune WAN with LoRA on a custom dataset; lab 4.3 will deploy.

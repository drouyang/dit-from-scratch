# Module 4.1 — WAN 2.1 inference + code tour

**Goal**: load **WAN 2.1 T2V-1.3B**, generate a 3-second video from a text prompt, and trace every component of the running system back to a lab from Parts 1–3. By the end you can open `Wan-Video/Wan2.1`'s source and recognize patchify, AdaLN-Zero, RoPE, cross-attention, flow matching, CFG — exactly the building blocks you wrote yourself, just extended to one more dimension.

WAN 2.1 T2V-1.3B is the smallest official WAN checkpoint (~1.3B params); inference fits comfortably on a single 4090 (24 GB), with VRAM to spare.

## Setup

### diffusers path (recommended for reading)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`WanPipeline` for the WAN 2.1 family is in stable diffusers — no from-source install needed. First inference run downloads the T2V-1.3B checkpoint (~5 GB total: VAE + umT5 + transformer) into the HF cache.

### Official Wan2.1 path (recommended for understanding production code)

```bash
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1
pip install -r requirements.txt
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```

## Run inference

### Path A: diffusers

```bash
python inference_diffusers.py --prompt "a fluffy red panda eating bamboo on a tree branch"
```

Output: `out.mp4` — 49 frames at 832×480, 16 fps (≈3 seconds). Generation takes ~3–5 minutes on a 4090 at default settings.

### Path B: official repo

```bash
cd Wan2.1
python generate.py --task t2v-1.3B --size 832*480 \
    --ckpt_dir ./Wan2.1-T2V-1.3B \
    --prompt "a fluffy red panda eating bamboo on a tree branch"
```

T2V-1.3B doesn't need the `--offload_model` / `--t5_cpu` flags that the bigger WAN 2.2 variants require — it just fits.

## Concept-organized code tour

The lab's main deliverable. Every production component → call path → file:function in `Wan-Video/Wan2.1`, mapped to which lab introduced it.

### Entry call path

```
generate.py  (--task t2v-1.3B)
   └─► wan/text2video.py :: WanT2V.generate()
          ├─► wan/modules/t5.py :: T5EncoderModel(...)              ← text encoding
          │      (tokenizer in wan/modules/tokenizers.py)
          ├─► torch.randn(...)                                       ← initial latent z_T
          ├─► sampling loop (FlowUniPCMultistepScheduler):
          │      └─► wan/modules/model.py :: WanModel(z_t, t, c)    ← DiT forward
          │             ├─► WanAttentionBlock                         ← block (×N)
          │             │      ├─► WanSelfAttention + RoPE-3D         ← image self-attn
          │             │      └─► WanCrossAttention                  ← attends to text
          │             └─► (returns predicted velocity)
          └─► wan/modules/vae.py :: WanVAE.decode(z_0)               ← video output
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
| `WanVAE.encode/decode` | `wan/modules/vae.py` | lab 3.2 (SD-VAE) | Latent diffusion: encode pixels → latent | 3D causal: compresses time + space; latent shape `(C, T, H, W)` |
| `T5EncoderModel` | `wan/modules/t5.py` | lab 3.2 (CLIP text encoder) | Pretrained, frozen, returns per-token features | umT5-XXL is ~100× larger; richer language understanding |
| `FlowUniPCMultistepScheduler` | imported into `text2video.py` | lab 2.2 (Euler ODE) | Flow-matching sampler | Higher-order multi-step solver, fewer steps for the same quality |
| AdaLN-Zero modulation | `wan/modules/model.py` | lab 3.1 (`adaLN_modulation`) | `c → SiLU → Linear` decoded into shift/scale/gate per block | Identical mechanism |
| CFG | `wan/text2video.py` | lab 2.2 / 3.2 | `v_uncond + s · (v_cond − v_uncond)` | Identical formula |

## Self-check

After reading the tour:

- [ ] Open `wan/modules/model.py` and find `WanAttentionBlock`. Confirm you can map every line of its forward pass to either lab 3.1 (self-attn / AdaLN / MLP) or lab 3.2 (cross-attn).
- [ ] Open `wan/modules/vae.py` and identify the `Conv3d` layers — that's the 3D causal extension over SD-VAE's 2D `Conv2d`.
- [ ] Find `rope_apply()` and confirm three concatenated rotation bands. Compare against lab 3.1's `apply_rope()` (which has two: row + column).

If you can do these three things, you've completed the lab.

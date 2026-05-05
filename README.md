# DiT from Scratch

A hands-on path to understanding Diffusion Transformers. Build each prerequisite, then assemble a working latent-space DiT for text-to-image. Part 4 takes you from "I built a tiny one" to "I can read, run, fine-tune, and deploy a real production WAN video model".

- **Time**: 3-6 hours per module (Part 4 is heavier and partly cloud-based)
- **Stack**: PyTorch + MacBook Pro M3 (Parts 1–3); Part 4 adds rented GPU compute

Each module has its own lab directory with detailed instructions (e.g. `lab1.1/` for Module 1.1).

---

## Part 1 — Building Blocks

The primitives DiT is made of.

| Module | Topic | Lab |
|---|---|---|
| 1.1 | MLP (warm-up) — FFN blocks, AdaLN, time embeddings | [lab1.1](./lab1.1) |
| 1.2 | CNN basics — just enough for VAE encoder/decoder | [lab1.2](./lab1.2) |
| **1.3** | **Attention** ★ | [lab1.3](./lab1.3) |
| **1.4** | **Transformer** ★ | [lab1.4](./lab1.4) |
| 1.5 | GPT-2 family — load OpenAI weights, compare sizes (optional) | [lab1.5](./lab1.5) |

★ = the architectural backbone of every modern transformer — DiT, GPT, LLaMA, ViT, BERT all build on these two.

## Part 2 — Diffusion Essentials

The training and sampling framework DiT is trained under.

| Module | Topic | Lab |
|---|---|---|
| 2.1 | VAE | [lab2.1](./lab2.1) |
| 2.2 | Flow Matching with Conditioning | [lab2.2](./lab2.2) |

## Part 3 — DiT

Put it together.

| Module | Topic | Lab |
|---|---|---|
| 3.1 | DiT architecture — patchify, AdaLN-Zero, RoPE, class conditioning | [lab3.1](./lab3.1) |
| 3.2 | Latent DiT — VAE + DiT end-to-end (class-conditional) | [lab3.2](./lab3.2) |
| 3.3 | Text conditioning — frozen CLIP/T5 + cross-attention; end-to-end mini text-to-image | [lab3.3](./lab3.3) |

End of the core curriculum: at this point you've built every component of a modern image DiT from scratch.

## From image to video — what changes architecturally

Going from a working image DiT (lab 3.3) to a video DiT (production: WAN, LTX, HunyuanVideo) is a small dimensional extension, not a redesign. Four things change; everything else is unchanged:

1. The VAE becomes a **3D causal VAE** that compresses both spatial and temporal axes, producing latents of shape `(C, T, H, W)`. "Causal" in time means future frames don't leak into past frames during encoding.
2. Patchify becomes **3D patchify** that tokenizes across time as well as space — patch shape `(p_t, p_h, p_w)` instead of `(p_h, p_w)`.
3. RoPE generalizes to **3D RoPE** for `(t, h, w)` positions — same construction with one more frequency band.
4. Compute requirements jump 1–2 orders of magnitude — production-scale training needs a cluster, which is why this curriculum doesn't include "build a video DiT from scratch on a laptop." Read WAN's code in Part 4 to see the extensions in production form.

DiT block structure, AdaLN-Zero, flow matching loss, CFG, text conditioning — all carry over unchanged.

## Part 4 — DiT in Production

A deliberate shape change: the labs above are *build from scratch*; these are *read, run, and modify a real production codebase*. The target is [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) — a state-of-the-art open text-to-video / image-to-video DiT family. After Parts 1–3 you can open WAN's source and recognize every block; this part makes you fluent in working with it, including the 3D-VAE / 3D-patchify / 3D-RoPE extensions described above.

| Module | Topic | Lab |
|---|---|---|
| 4.1 | WAN inference + code tour — load a real checkpoint, generate video, trace every component back to labs 1–3 (including the 3D extensions) | [lab4.1](./lab4.1) |
| 4.2 | LoRA fine-tuning — adapt WAN to a small custom dataset (rented GPU compute) | [lab4.2](./lab4.2) |
| 4.3 | Deployment — `diffusers` integration, ComfyUI workflow, hosted inference endpoint | [lab4.3](./lab4.3) |

Compute notes: 4.1 runs on M3 with quantization (or rented GPU for full precision); 4.2 needs a rented H100 / A100 for a few hours; 4.3 is cloud deployment.

## Reference Library

**Core architecture**

- [DiT paper (Peebles & Xie 2022)](https://arxiv.org/abs/2212.09748) and [facebookresearch/DiT](https://github.com/facebookresearch/DiT) — the architecture this course builds toward.
- [Stable Diffusion 3 / MMDiT (Esser et al. 2024)](https://arxiv.org/abs/2403.03206) — the modern canonical paper. DiT + latent diffusion + flow matching + multimodal text/image attention, all in one.
- [Flow Matching (Lipman et al. 2022)](https://arxiv.org/abs/2210.02747), [Rectified Flow (Liu et al. 2022)](https://arxiv.org/abs/2209.03003) — the production training paradigm.
- Hugging Face [`diffusers`](https://github.com/huggingface/diffusers) — reference implementations of SD3, FLUX, WAN, and friends.

**Video (Part 4)**

- [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) — the production codebase Part 4 targets.
- [LTX-Video](https://github.com/Lightricks/LTX-Video), [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) — adjacent video DiT codebases worth comparing against WAN.

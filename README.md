# DiT from Scratch

A hands-on path to understanding Diffusion Transformers. Build each prerequisite, then assemble a working latent-space DiT for text-to-image. Part 4 takes you from "I built a tiny one" to "I can read, run, fine-tune, and deploy a real production WAN video model".

- **Stack**: PyTorch + MacBook Pro M3 (Parts 1–3); Part 4 adds rented GPU compute
- **Time**: see each Part heading below for total hours

Each module has its own lab directory with detailed instructions (e.g. `lab1.1/` for Module 1.1).

---

## Part 1 — Building Blocks (~15-25 hours)

The primitives DiT is made of.

| Module | Topic | Lab |
|---|---|---|
| 1.1 | MLP (warm-up) — FFN blocks, AdaLN, time embeddings | [lab1.1](./lab1.1) |
| 1.2 | CNN basics — just enough for VAE encoder/decoder | [lab1.2](./lab1.2) |
| **1.3** | **Attention** ★ | [lab1.3](./lab1.3) |
| **1.4** | **Transformer** ★ | [lab1.4](./lab1.4) |
| 1.5 | GPT-2 family — load OpenAI weights, compare sizes (optional) | [lab1.5](./lab1.5) |

★ = the architectural backbone of every modern transformer — DiT, GPT, LLaMA, ViT, BERT all build on these two.

## Part 2 — Diffusion Essentials (~6-10 hours)

The training and sampling framework DiT is trained under.

| Module | Topic | Lab |
|---|---|---|
| 2.1 | VAE | [lab2.1](./lab2.1) |
| 2.2 | Flow Matching with Conditioning | [lab2.2](./lab2.2) |

## Part 3 — DiT (~5-8 hours)

Put it together.

| Module | Topic | Lab |
|---|---|---|
| 3.1 | DiT architecture — patchify, AdaLN-Zero, RoPE-2D | [lab3.1](./lab3.1) |
| 3.2 | Latent text-to-image DiT — VAE , text embedding with cross-attention | [lab3.2](./lab3.2) |

3.1 builds and verifies the DiT architecture on raw pixels with class labels. 3.2 is the synthesis: same DiT, but operating on VAE latents and conditioned on text — the full production stack at small scale. At this point you've built every component of a modern image DiT from scratch.

## Part 4 — DiT in Production (~10-15 hours)

A deliberate shape change: the labs above are *build from scratch*; these are *read, run, and modify a real production codebase*. The target is [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) — a state-of-the-art open **video** DiT family (text-to-video and image-to-video). Going from image to video is a small dimensional extension, not a redesign — the 2D VAE becomes a 3D causal VAE producing `(C, T, H, W)` latents, patchify and RoPE pick up a temporal axis, compute jumps ~10×, and everything else (DiT block, AdaLN-Zero, flow matching, CFG, text conditioning) carries over unchanged.

| Module | Topic | Lab |
|---|---|---|
| 4.1 | WAN inference + code tour | [lab4.1](./lab4.1) |
| 4.2 | Post-training WAN — hands-on LoRA fine-tune, survey of full SFT / Diffusion-DPO / distillation | [lab4.2](./lab4.2) |
| 4.3 | Distribution — publish your LoRA, ComfyUI custom node, quantization | [lab4.3](./lab4.3) |

## Reference Library

**Core architecture**

- [DiT paper (Peebles & Xie 2022)](https://arxiv.org/abs/2212.09748) and [facebookresearch/DiT](https://github.com/facebookresearch/DiT) — the architecture this course builds toward.
- [Stable Diffusion 3 / MMDiT (Esser et al. 2024)](https://arxiv.org/abs/2403.03206) — the modern canonical paper. DiT + latent diffusion + flow matching + multimodal text/image attention, all in one.
- [Flow Matching (Lipman et al. 2022)](https://arxiv.org/abs/2210.02747), [Rectified Flow (Liu et al. 2022)](https://arxiv.org/abs/2209.03003) — the production training paradigm.
- Hugging Face [`diffusers`](https://github.com/huggingface/diffusers) — reference implementations of SD3, FLUX, WAN, and friends.

**Video (Part 4)**

- [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) — the production codebase Part 4 targets.
- [LTX-Video](https://github.com/Lightricks/LTX-Video), [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) — adjacent video DiT codebases worth comparing against WAN.

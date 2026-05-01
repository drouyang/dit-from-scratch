# DiT from Scratch

A hands-on path to understanding Diffusion Transformers. Build each prerequisite, then assemble a working latent-space DiT for text-to-image.

- **Time**: 3-6 hours per module
- **Stack**: PyTorch + MacBook Pro M3

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
| 2.1 | VAE — the latent space DiT operates in | [lab2.1](./lab2.1) |
| 2.2 | Flow Matching with CFG | [lab2.2](./lab2.2) |

## Part 3 — DiT

Put it together.

| Module | Topic | Lab |
|---|---|---|
| 3.1 | DiT architecture — patchify, AdaLN-Zero, RoPE, class conditioning | [lab3.1](./lab3.1) |
| 3.2 | Latent DiT — VAE + DiT end-to-end | [lab3.2](./lab3.2) |
| 3.3 | Text conditioning — T5/CLIP encoder + cross-attention | [lab3.3](./lab3.3) |

## Capstone

**Mini text-to-image**: Latent DiT (3.2) with text conditioning (3.3), trained with flow matching + CFG (2.2) on a small text-image dataset.

**Extending to video (WAN, LTX-Video, HunyuanVideo style).** The capstone stops at images because video training needs a cluster, not a laptop. To take the same stack to video, four pieces change: (1) the VAE becomes a **3D causal VAE** that compresses both spatial and temporal axes, producing latents of shape `(C, T, H, W)`; (2) patchify becomes **3D patchify** that tokenizes across time as well as space; (3) RoPE generalizes to **3D RoPE** for `(t, h, w)` positions; (4) compute requirements jump 1–2 orders of magnitude. Everything else — DiT block structure, AdaLN-Zero, flow matching loss, CFG, text conditioning — carries over unchanged.


## Reference Library

- [DiT paper (Peebles & Xie 2022)](https://arxiv.org/abs/2212.09748) and [facebookresearch/DiT](https://github.com/facebookresearch/DiT) — the architecture this course builds toward.
- [Stable Diffusion 3 / MMDiT (Esser et al. 2024)](https://arxiv.org/abs/2403.03206) — the modern canonical paper. DiT + latent diffusion + flow matching + multimodal text/image attention, all in one.
- [Flow Matching (Lipman et al. 2022)](https://arxiv.org/abs/2210.02747), [Rectified Flow (Liu et al. 2022)](https://arxiv.org/abs/2209.03003) — the production training paradigm.
- Hugging Face [`diffusers`](https://github.com/huggingface/diffusers) — reference implementations of SD3, FLUX, and friends.

# DiT from Scratch

A hands-on path to understanding Diffusion Transformers. Build each prerequisite, then assemble a working latent-space DiT for text-to-image. Part 4 extends the architecture to video; Part 5 takes you from "I built a tiny one" to "I can fine-tune and deploy a real production WAN model".

- **Time**: 3-6 hours per module (Parts 4–5 are heavier; Part 5 is partly cloud-based)
- **Stack**: PyTorch + MacBook Pro M3 (Parts 1–4); Part 5 adds rented GPU compute

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

## Capstone (after Part 3)

**Mini text-to-image.** Take lab 3.2's latent DiT, swap the class-label embedding for **text conditioning** — a frozen CLIP or T5 encoder turns prompts into embeddings; the DiT consumes them via cross-attention (the SD3 / FLUX recipe at small scale). Trained with flow matching + CFG (lab 2.2) on a small text-image dataset. End-to-end "type a prompt, get an image."

The conditioning interface to the DiT is the same as in lab 3.2 — a fixed-shape vector flowing into AdaLN / cross-attention — only the *source* changes (text encoder output instead of an embedding lookup). That's why the swap is a capstone integration step rather than its own lab. End of the core curriculum: at this point you've built every component of a modern image DiT from scratch.

## Part 4 — From Image to Video

The architecture pivot from spatial to spatiotemporal. Tiny models on toy video data — the goal is to *prove* the 2D→3D extension works, not to match production quality.

| Module | Topic | Lab |
|---|---|---|
| 4.1 | 3D VAE — extend lab 2.1's VAE with a temporal axis; produce `(C, T, H, W)` latents on moving-MNIST | [lab4.1](./lab4.1) |
| 4.2 | 3D DiT — extend lab 3.1's DiT with 3D patchify and 3D RoPE; class-conditional video on tiny clips | [lab4.2](./lab4.2) |

Four things change going from image to video, all of them small and visible: (1) the VAE becomes a **3D causal VAE** that compresses both spatial and temporal axes; (2) patchify becomes **3D patchify** that tokenizes across time as well as space; (3) RoPE generalizes to **3D RoPE** for `(t, h, w)` positions; (4) compute requirements jump 1–2 orders of magnitude (this part you only *see* — production-scale training needs a cluster). Everything else — DiT block structure, AdaLN-Zero, flow matching loss, CFG, text conditioning — carries over unchanged.

## Part 5 — DiT in Production

A deliberate shape change: the labs above are *build from scratch*; these are *read, run, and modify a real production codebase*. The target is [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) — a state-of-the-art open text-to-video / image-to-video DiT family. After Parts 1–4 you can open WAN's source and recognize every block; this part makes you fluent in working with it.

| Module | Topic | Lab |
|---|---|---|
| 5.1 | WAN inference + code tour — load a real checkpoint, generate video, trace every component back to labs 1–4 | [lab5.1](./lab5.1) |
| 5.2 | LoRA fine-tuning — adapt WAN to a small custom dataset (rented GPU compute) | [lab5.2](./lab5.2) |
| 5.3 | Deployment — `diffusers` integration, ComfyUI workflow, hosted inference endpoint | [lab5.3](./lab5.3) |

Compute notes: 5.1 runs on M3 with quantization (or rented GPU for full precision); 5.2 needs a rented H100 / A100 for a few hours; 5.3 is cloud deployment.

## Reference Library

**Core architecture**

- [DiT paper (Peebles & Xie 2022)](https://arxiv.org/abs/2212.09748) and [facebookresearch/DiT](https://github.com/facebookresearch/DiT) — the architecture this course builds toward.
- [Stable Diffusion 3 / MMDiT (Esser et al. 2024)](https://arxiv.org/abs/2403.03206) — the modern canonical paper. DiT + latent diffusion + flow matching + multimodal text/image attention, all in one.
- [Flow Matching (Lipman et al. 2022)](https://arxiv.org/abs/2210.02747), [Rectified Flow (Liu et al. 2022)](https://arxiv.org/abs/2209.03003) — the production training paradigm.
- Hugging Face [`diffusers`](https://github.com/huggingface/diffusers) — reference implementations of SD3, FLUX, WAN, and friends.

**Video (Parts 4–5)**

- [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) — the production codebase Part 5 targets.
- [LTX-Video](https://github.com/Lightricks/LTX-Video), [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) — adjacent video DiT codebases worth comparing against WAN.
- [Moving MNIST](https://www.cs.toronto.edu/~nitish/unsupervised_video/) — toy video dataset used in lab 4.x.

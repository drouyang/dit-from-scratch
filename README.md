# DiT from Scratch

A hands-on path to understanding Diffusion Transformers. Build each prerequisite, then assemble a working latent-space DiT for text-to-image.

- **Time**: ~6–8 weeks at 10 hrs/week
- **Stack**: PyTorch + a 4090 GPU
- **Deliverable per module**: a working notebook/script and a short "what clicked" note

Each module has its own lab directory with detailed instructions (e.g. `lab1.1/` for Module 1.1).

---

## Part 1 — Building Blocks

The primitives DiT is made of. Skip none of these — every one shows up inside the final model.

| Module | Topic | Lab |
|---|---|---|
| 1.1 | MLP (warm-up) — FFN blocks, AdaLN, time embeddings | [lab1.1](./lab1.1) |
| 1.2 | CNN basics — just enough for VAE encoder/decoder | [lab1.2](./lab1.2) |
| 1.3 | Attention (standalone) | [lab1.3](./lab1.3) |
| 1.4 | Transformer (nanoGPT) | [lab1.4](./lab1.4) |

## Part 2 — Diffusion Essentials

The training and sampling framework DiT is trained under.

| Module | Topic | Lab |
|---|---|---|
| 2.1 | VAE — the latent space DiT operates in | [lab2.1](./lab2.1) |
| 2.2 | DDPM / DDIM / Classifier-Free Guidance | [lab2.2](./lab2.2) |
| 2.3 | Flow Matching / Rectified Flow | [lab2.3](./lab2.3) |

## Part 3 — DiT

Put it together.

| Module | Topic | Lab |
|---|---|---|
| 3.1 | DiT architecture — patchify, AdaLN-Zero, class conditioning | [lab3.1](./lab3.1) |
| 3.2 | Latent DiT — VAE + DiT end-to-end | [lab3.2](./lab3.2) |
| 3.3 | Text conditioning + CFG | [lab3.3](./lab3.3) |

## Capstone

**Mini text-to-image**: Latent DiT (3.2) with text conditioning (3.3), trained with flow matching (2.3) and CFG (2.2) on a small text-image dataset.

---

## Supporting Practices

- **Profiling**: `torch.profiler`, `nsys` traces, `torch.cuda.memory_summary()` — use them at least once per part.
- **Reading**: one paper per week from the module references; read implementations after your own attempt.
- **Writing**: a one-paragraph "what clicked" note per module.
- **Debugging**: overfit a single batch before scaling; print `param.grad.norm()` per layer; start small.

## Reference Library

- [DiT paper (Peebles & Xie)](https://arxiv.org/abs/2212.09748) and [facebookresearch/DiT](https://github.com/facebookresearch/DiT)
- [Latent Diffusion (Rombach et al.)](https://arxiv.org/abs/2112.10752)
- [Flow Matching (Lipman et al.)](https://arxiv.org/abs/2210.02747), [Rectified Flow (Liu et al.)](https://arxiv.org/abs/2209.03003)
- Karpathy — "Neural Networks: Zero to Hero" and [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Lilian Weng's blog](https://lilianweng.github.io/) — diffusion and flow-matching posts
- Hugging Face `diffusers` source — reference implementations of DiT/SD3
- Annotated Transformer / Annotated Diffusion

---

## Milestones

### Part 1 — Building Blocks
- [ ] 1.1 MLP on MNIST to 98%+
- [ ] 1.2 Small CNN autoencoder reconstructing CIFAR-10
- [ ] 1.3 Multi-head attention matching PyTorch built-in
- [ ] 1.4 nanoGPT trained and generating

### Part 2 — Diffusion Essentials
- [ ] 2.1 VAE with interpolatable latent space
- [ ] 2.2 DDPM + DDIM + CFG on MNIST/CIFAR
- [ ] 2.3 Flow matching trained and compared to DDPM

### Part 3 — DiT
- [ ] 3.1 DiT trained with class conditioning (ImageNet-subset or CIFAR)
- [ ] 3.2 Latent DiT: VAE encode → DiT → VAE decode, end-to-end
- [ ] 3.3 Text-conditioned DiT with CFG

### Capstone
- [ ] Mini text-to-image completed end-to-end

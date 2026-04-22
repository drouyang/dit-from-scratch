# Deep Learning from Scratch

A hands-on path from MLPs through modern generative models. Implement first, compare with reference implementations after.

- **Time**: ~8–10 weeks at 10 hrs/week
- **Stack**: PyTorch + a GPU (4090 for Layer 1; 4090 works for Layer 2)
- **Deliverable per module**: a working notebook/script and a short "what clicked" note

Each module has its own lab directory with detailed instructions (e.g. `lab1.1/` for Module 1.1).

---

## Layer 1 — Architectural Building Blocks

Fluency with the primitives. Read any architecture paper and map it to PyTorch code.

| Module | Topic | Lab |
|---|---|---|
| 1.1 | MLP (warm-up) | [lab1.1](./lab1.1) |
| 1.2 | CNN / ResNet | [lab1.2](./lab1.2) |
| 1.3 | RNN / LSTM / GRU | [lab1.3](./lab1.3) |
| 1.4 | Attention (standalone) | [lab1.4](./lab1.4) |
| 1.5 | Transformer (nanoGPT) | [lab1.5](./lab1.5) |
| 1.6 | Mixture of Experts | [lab1.6](./lab1.6) |

## Layer 2 — Generative Model Paradigms

The *types* of generative models and their inference characteristics.

| Module | Topic | Lab |
|---|---|---|
| 2.1 | Autoregressive LMs | [lab2.1](./lab2.1) |
| 2.2 | VAE | [lab2.2](./lab2.2) |
| 2.3 | GAN (brief) | [lab2.3](./lab2.3) |
| 2.4 | Diffusion (DDPM / DDIM / CFG) | [lab2.4](./lab2.4) |
| 2.5 | Flow Matching / Rectified Flow | [lab2.5](./lab2.5) |
| 2.6 | DiT (Diffusion Transformer) | [lab2.6](./lab2.6) |

## Capstone

**Mini text-to-image**: VAE (2.2) + DiT (2.6) + flow matching (2.5) + CFG (2.4), trained on a small text-image dataset.

---

## Supporting Practices

- **Profiling**: `torch.profiler`, `nsys` traces, `torch.cuda.memory_summary()` — use them at least once per layer.
- **Reading**: one paper per week from the module references; read implementations after your own attempt.
- **Writing**: a one-paragraph "what clicked" note per module.
- **Debugging**: overfit a single batch before scaling; print `param.grad.norm()` per layer; start small.

## Reference Library

- Karpathy — "Neural Networks: Zero to Hero"
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Lilian Weng's blog](https://lilianweng.github.io/)
- Hugging Face `diffusers` source
- Annotated Transformer / Annotated Diffusion
- *Deep Learning* (Goodfellow, Bengio, Courville)

---

## Milestones

### Layer 1
- [ ] 1.1 MLP on MNIST to 98%+
- [ ] 1.2 ResNet on CIFAR-10 to 90%+
- [ ] 1.3 Char-RNN generating coherent Shakespeare
- [ ] 1.4 Multi-head attention matching PyTorch built-in
- [ ] 1.5 nanoGPT trained and generating, with KV cache
- [ ] 1.6 MoE transformer with stable routing

### Layer 2
- [ ] 2.1 Multiple decoding strategies on my LM
- [ ] 2.2 VAE with interpolatable latent space
- [ ] 2.3 DCGAN trained, mode collapse observed
- [ ] 2.4 DDPM + DDIM + CFG on MNIST
- [ ] 2.5 Flow matching trained and compared to DDPM
- [ ] 2.6 DiT trained with class conditioning

### Capstone
- [ ] Mini text-to-image completed end-to-end

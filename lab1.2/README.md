# Module 1.2 — CNN Basics

> Part 1 — Building Blocks · [DiT from Scratch](../README.md)

**Goal**: build an encoder–decoder CNN that compresses 32×32 color images into a compact latent vector and reconstructs them from it.

**Why this matters for DiT**: DiT doesn't operate on raw pixels — it operates on latents produced by a VAE. That VAE is a CNN encoder/decoder. The stride-2 `Conv2d` → `ConvTranspose2d` pattern you build here is the literal architecture of the VAE encoder and decoder in Stable Diffusion and DiT. Understanding how spatial resolution is traded for channel depth (and how to invert that) is the prerequisite for lab2.1.

**Deliverable**: `train.py` training a CNN autoencoder on CIFAR-10, with reconstructions visible in `reconstructions.png`.

## What the model learns

**CIFAR-10** is 60,000 32×32 RGB images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). 50k train / 10k test. The classes are visually diverse and share no obvious low-level patterns — this forces the encoder to learn genuinely useful features rather than memorizing color statistics.

The task is **unsupervised reconstruction**: given a raw image, compress it to a small latent vector, then expand it back. There are no class labels involved. The model is evaluated purely by how close the reconstruction is to the original, measured in per-pixel mean squared error (MSE).

```
 input image                    latent                  reconstructed image
(3, 32, 32)  →  [Encoder]  →  (latent_dim,)  →  [Decoder]  →  (3, 32, 32)
```

The encoder discards information by passing the image through a bottleneck smaller than the input (3×32×32 = 3,072 values → 256 by default). The decoder must reconstruct all 3,072 values from those 256. What survives the bottleneck is what the encoder learned to treat as essential.

**Why this architecture previews the VAE**: a Variational Autoencoder is this exact structure with two additions:
1. The encoder outputs *two* vectors — mean **μ** and log-variance **log σ²** — and the latent is *sampled* as `z = μ + σ · ε`, `ε ~ N(0,1)`.
2. The loss gains a KL term that keeps the latent distribution near `N(0,1)`, making the space smooth and interpolatable.

Everything else — the conv/deconv blocks, the bottleneck, the reconstruction loss — is shared.

## Files

| File | What it is |
| --- | --- |
| `cnn.py` | `Encoder`, `Decoder`, and `Autoencoder` modules with shape annotations and concept comments |
| `train.py` | Downloads CIFAR-10, trains, evaluates each epoch, saves weights |
| `visualize.py` | Loads a checkpoint and plots a side-by-side grid of originals vs reconstructions |

## Instructions

### 1. Set up

Python 3.9+. From `lab1.2/`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training auto-selects device: CUDA → Apple MPS → CPU. CIFAR-10 downloads to `./data/` on first run (~170 MB).

### 2. Train with defaults

```bash
python train.py
```

Default: Adam, lr=1e-3, 30 epochs, latent_dim=256. Per-epoch output:

```
epoch  1 |   8.3s | train_loss 0.02847 | test_loss 0.02134
...
epoch 30 |   8.1s | train_loss 0.00712 | test_loss 0.00748
```

`test_loss` is per-pixel MSE — the average squared difference between each reconstructed pixel and the original. A value around 0.007 means the average pixel error is roughly `sqrt(0.007) ≈ 0.084` on the [0,1] scale, which is visually imperfect but structurally recognizable. Shapes and colors come back; fine textures do not — that's a known limitation of MSE (it rewards blurry averages).

### 3. How convolutions work

Open `cnn.py` and read the comments. The core ideas to lock in:

**Local receptive field.** Each output value in a conv layer depends only on a small spatial patch of the input (the kernel window). This is the key difference from an MLP, where every output depends on *all* inputs. CNNs are efficient because most inputs don't affect most outputs.

**Parameter sharing.** The same kernel weights are applied at every spatial position. A kernel that detects a horizontal edge at position (5, 5) uses the same weights as the one at position (12, 20). This is why CNNs generalize well: they don't have to re-learn features at every location.

**Channels as parallel detectors.** `out_channels=32` means 32 independent kernels run in parallel, each producing its own 2D output map. Deep in the network, each channel represents a different abstract feature.

**Stride-2 conv for downsampling.** With `stride=2`, the kernel jumps 2 pixels at a time → output is half the spatial size. Learnable, unlike MaxPool.

**ConvTranspose2d for upsampling.** The decoder's block doubles spatial size: it inserts zeros between input values, then applies a regular conv. The weights are learned — this is not bilinear upsampling.

**Receptive field grows with depth.** After 3 stride-2 conv blocks, each output value in the `(128, 4, 4)` feature map has a receptive field covering the *entire* 32×32 input. The encoder truly "sees" the whole image before projecting to the latent.

### 4. Experiment with the latent dimension

The `--latent-dim` flag controls how aggressively the image is compressed. Run all three:

```bash
python train.py --latent-dim 32   --save-path ae_32.pt
python train.py --latent-dim 256  --save-path ae_256.pt   # default
python train.py --latent-dim 1024 --save-path ae_1024.pt
```

Larger latent → lower MSE, sharper reconstructions, but less compression. Smaller latent → higher MSE, blurrier reconstructions, but the model is forced to learn only the most essential structure.

The compression ratio for each:
- `latent_dim=32`:   3072 → 32  ≈ 96× compression
- `latent_dim=256`:  3072 → 256 ≈ 12× compression
- `latent_dim=1024`: 3072 → 1024 ≈ 3× compression

The point: there's a tradeoff between compression and fidelity. VAEs tune this via the KL weight (the β in β-VAE). DiT's VAE (from Stable Diffusion) uses a spatial latent (not a flat vector) at 8× spatial compression: a 512×512 image becomes a 64×64×4 tensor — a much milder compression than `latent_dim=32` here.

### 5. Visualize reconstructions

```bash
python visualize.py --ckpt cnn_autoencoder.pt --save reconstructions.png
```

Two rows: originals on top, reconstructions on bottom. What to look for:

- **Shapes and colors are preserved** even at `latent_dim=32`. The encoder learns global structure first.
- **Fine textures and sharp edges are blurred**. MSE minimization produces the "average" of plausible reconstructions — blurry but not wrong. Perceptual losses (LPIPS) and adversarial losses (GAN discriminator) fix this; VAEs typically combine MSE + KL.
- **Higher latent_dim → more texture detail**. Compare `ae_32.pt` vs `ae_1024.pt` side by side:

```bash
python visualize.py --ckpt ae_32.pt   --latent-dim 32   --save recon_32.png
python visualize.py --ckpt ae_1024.pt --latent-dim 1024 --save recon_1024.png
```

The `latent_dim=1024` reconstructions are noticeably sharper — the bottleneck is wide enough to pass through more spatial detail.

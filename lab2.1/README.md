# Module 2.1 — VAE

> Part 2 — Diffusion Essentials · [DiT from Scratch](../README.md)

**Goal**: turn lab 1.2's deterministic autoencoder into a **Variational** Autoencoder by adding the two pieces that make the latent space generative — make the encoder output a normal distribution (parameterized by `mu` and `logvar`) and add a loss term that pulls it toward the standard normal `N(0, I)`. Train on MNIST, then verify the latent space is continuous and meaningful by checking that (a) random points in latent space decode to plausible new digits and (b) walking between two encoded digits produces a smooth morph.

**Why this matters for DiT**: DiT does not operate on raw pixels. It operates on **latents produced by a VAE** — the Stable Diffusion VAE, an 8× spatial downsampler with a 4-channel latent. Diffusion learns to navigate that latent space, and the only reason it can navigate it is the VAE's prior regularization: without it, the latent space would be a bag of disconnected points and intermediate latents (which diffusion produces at every denoising step) would decode to garbage. The trick you build here is the same one Stable Diffusion uses; only the encoder/decoder shapes change.

## Refresher: from autoencoder (lab 1.2) to VAE (lab 2.1)

The VAE in this lab is lab 1.2's autoencoder with three small but consequential changes. Recap of both architectures so the diff is precise.

**The shape of the problem.** An autoencoder pushes an image through a **bottleneck** — a flat latent vector much smaller than the image — and is trained to reconstruct the input from that compressed code. The encoder learns to throw away information that doesn't help reconstruction; the decoder learns to undo the compression. With no bottleneck the network would just learn the identity.

```
   image  ─►  Encoder  ─►  z (latent)  ─►  Decoder  ─►  reconstruction
 (B,C,H,W)   downsample      (B, D)         upsample       (B,C,H,W)
              + flatten                    + reshape
                            ▲
                       bottleneck:
                  D ≪ C·H·W forces compression
```

### Lab 1.2 — deterministic autoencoder on CIFAR-10

```
input (B, 3, 32, 32)

── Encoder ──────────────────────────────────────────────────────────
  └─ Conv2d k=4, s=2 ─► (B, 32, 16, 16) ─► BN ─► ReLU
  └─ Conv2d k=4, s=2 ─► (B, 64,  8,  8) ─► BN ─► ReLU
  └─ Conv2d k=4, s=2 ─► (B, 128, 4,  4) ─► BN ─► ReLU
  └─ Flatten         ─► (B, 2048)
  └─ Linear          ─► (B, 256)              ★ bottleneck z = encoder output

── Decoder ──────────────────────────────────────────────────────────
input z (B, 256)
  └─ Linear          ─► (B, 2048)             (expand back from bottleneck)
  └─ Unflatten       ─► (B, 128, 4, 4) ─► BN ─► ReLU
  └─ ConvTranspose2d ─► (B, 64,  8,  8) ─► BN ─► ReLU
  └─ ConvTranspose2d ─► (B, 32, 16, 16) ─► BN ─► ReLU
  └─ ConvTranspose2d ─► (B,  3, 32, 32)
  └─ Sigmoid          ─► reconstruction in [0, 1]
```

Loss: MSE (or BCE) reconstruction only. No KL term, no sampling.

### Lab 2.1 — VAE on MNIST

```
input (B, 1, 28, 28)

── Encoder ──────────────────────────────────────────────────────────
  └─ Conv2d k=3, s=2 ─► (B, 32, 14, 14) ─► ReLU
  └─ Conv2d k=3, s=2 ─► (B, 64,  7,  7) ─► ReLU
  └─ Flatten          ─► (B, 3136)
  ├─ Linear ─► (B, 16)  ← mu
  └─ Linear ─► (B, 16)  ← logvar       ★ two heads instead of one

── Reparameterize ───────────────────────────────────────────────────
   inputs:  mu (B, 16), logvar (B, 16)             from heads above
   z = mu + σ · ε,    σ = exp(½·logvar),    ε ~ N(0, I)
   output:  z (B, 16)                              ★ sampled latent, gradient flows through mu, σ

── Decoder ──────────────────────────────────────────────────────────
input z (B, 16)
  └─ Linear           ─► (B, 3136)            (expand back from bottleneck)
  └─ reshape          ─► (B, 64, 7, 7) ─► ReLU
  └─ ConvTranspose2d  ─► (B, 32, 14, 14) ─► ReLU
  └─ ConvTranspose2d  ─► (B,  1, 28, 28)
                                              (no sigmoid — outputs logits)
```

Loss: `BCE_with_logits(x̂, x) + beta · KL(N(mu, σ²) || N(0, I))`.

### What actually changed

**The one-line VAE idea: encode each image as a *distribution* over latent space, sample from it, and decode the sample.**

| | Lab 1.2 (AE) | Lab 2.1 (VAE) |
| --- | --- | --- |
| What an image gets encoded as | one vector `z ∈ R^d` | a distribution `N(mu, σ²·I)` over `R^d` ★ |
| Dataset | CIFAR-10, (3, 32, 32) | MNIST, (1, 28, 28) |
| Latent dim | 256 | 16 |
| Encoder downsample stages | 3 (32 → 16 → 8 → 4) | 2 (28 → 14 → 7) |
| Encoder bottleneck head | **1** Linear → `z` | **2** Linears → `mu`, `logvar` ★ |
| Sampling | none — `z` is deterministic | `z = mu + σ·eps`, reparameterized ★ |
| Decoder output | `Sigmoid` → pixels in [0, 1] | logits (sigmoid applied at loss time) |
| Norm in conv blocks | BatchNorm2d everywhere | none |
| Loss | reconstruction only | reconstruction **+ KL term** ★ |

The **★** rows are the changes that define a VAE. Everything else is incidental — different dataset (CIFAR is harder; MNIST is good enough at smaller capacity), different normalization choice (BatchNorm helps with deeper RGB stacks; tiny grayscale MNIST converges fine without it), one fewer downsample stage (28 = 7 × 4 wants 2 stages, 32 = 4 × 8 wants 3).

## What the model learns

**MNIST**: 28×28 grayscale digits, 60k train / 10k test. Small and clean enough that the latent space is interpretable in a single afternoon's training.

The architecture is lab 1.2's encoder/decoder, with two changes:

1. **The encoder outputs two vectors** instead of one — `mu` and `logvar`, the mean and log-variance of a normal distribution over the latent. We want to draw a sample from `N(mu, σ²)`. The naive way — `torch.distributions.Normal(mu, σ).sample()` — gives the right sample but the result has no `grad_fn`, so backprop can't flow back to `mu` and `σ` and we can't train the encoder. The fix is the **reparameterization trick**:
   ```
   z = mu + σ * eps,    eps ~ N(0, I),    σ = exp(0.5 * logvar)
   ```
   This is a sample from `N(mu, σ²)` rewritten as a *deterministic* function of `(mu, σ)` plus a parameterless noise source `eps`. Now `∂z/∂mu = 1` and `∂z/∂σ = eps`, so gradients flow normally through the encoder. Same sample; differentiable computation graph.

2. **The loss adds a KL term** that pulls the per-image posterior `N(mu, σ²)` toward the standard normal prior `N(0, I)`:
   ```
   L = recon(x, x̂) + beta * KL(N(mu, σ²) || N(0, I))
   ```
   `recon(x, x̂)` is the reconstruction loss between the input `x` and the decoder's output `x̂` — `BCE_with_logits` in this lab (MSE in many textbooks). This whole loss `L` is the **negative ELBO** ("Evidence Lower BOund"), the standard VAE objective from variational inference: we minimize `L`, which equivalently maximizes the ELBO. That's why the training log later reports `ELBO`, `recon`, and `KL` separately — they're the three terms of this same equation.

   For two normal distributions the KL has a closed form:
   ```
   KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
   ```
   KL stands for Kullback–Leibler divergence, after Solomon Kullback and Richard Leibler (1951). It measures how different two probability distributions are. Without this term the encoder is free to spread points anywhere in latent space — there's no reason for two adjacent latents to decode to similar images. With it, the latents pile up near the origin, the prior matches the aggregate posterior, and `N(0, I)` becomes a usable sampling distribution.

## Files

| File | What it is |
| --- | --- |
| `vae.py` | `Encoder` (with `mu`/`logvar` heads), `Decoder`, `reparameterize`, `vae_loss`, and the `VAE` module |
| `train.py` | Trains the VAE on MNIST with `Adam` + the ELBO loss; reports per-epoch ELBO / recon / KL |
| `visualize.py` | Produces three figures: reconstructions, prior samples, latent interpolations |

## Instructions

### 1. Set up

Python 3.9+. From `lab2.1/`:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Training auto-selects device: CUDA → Apple MPS → CPU. MNIST downloads to `./data/` on first run (~12 MB).

### 2. Train

```bash
python train.py
```

Default: Adam, lr=1e-3, 20 epochs, latent_dim=16, beta=1.0, batch=128. Per-epoch output:

```
epoch  1 |   5.4s | train ELBO  140.21  recon  130.42  KL  9.78 | test ELBO  108.32  recon  100.11  KL  8.21
...
epoch 20 |   105.2s | train ELBO   95.21  recon   75.12  KL 20.09 | test ELBO   97.43  recon   77.08  KL 20.35
```

What to watch:

- **ELBO drops monotonically.** That is the model getting better at modelling pixels under the constraint that latents stay near `N(0, I)`.
- **`recon` falls fast at first, then slows.** Reconstruction quickly learns to roughly match the digit; the slow tail is sharpening edges and strokes.
- **`KL` rises over training.** Counter-intuitive but correct: at init, both `mu` and `logvar` are near zero, so KL is tiny. As the encoder learns to actually *use* the latent space (place different digits at different `mu`), KL grows. KL = 0 means the posterior collapsed and the latent is unused (a known VAE failure mode).

### 3. Reconstruct, sample, interpolate, scatter

```bash
python visualize.py --mode all
```

This produces four figures in `lab2.1/`:

- `reconstructions.png` — top row originals, bottom row VAE reconstructions. Should look like blurry-but-recognizable digits. Reconstruction quality is similar to lab 1.2's autoencoder; the VAE does not reconstruct *better* than an autoencoder. The point of the VAE is what the latent space *can do*, not how well it reconstructs.
- `samples.png` — 64 digits decoded from `z ~ N(0, I)`. **This is the strictest test of the KL term.** A trained VAE produces plausible digits here. An autoencoder without KL would produce noise: random points in its latent space don't correspond to anything the decoder was trained on.
- `interpolation.png` — two random test digits at the ends, eight intermediate frames in between. Walking linearly between `mu_a` and `mu_b` should morph smoothly through digit shapes. Without the KL prior, the line between two valid latents passes through "off-manifold" regions and the intermediate frames look like noise.
- `latent_scatter.png` — 2000 encoded test digits' `mu` values, scattered in 2-D, colored by digit class. Plotted directly when `latent_dim=2`; PCA-projected to 2-D otherwise. You should see ten partially-separated clusters sitting inside roughly a circle of radius ~3 (the spread of `N(0, I)` projected to 2-D). Cleaner cluster separation = the encoder is using the latent space *meaningfully*; clusters all piled at the origin = the encoder collapsed (KL too strong, or training too short).

## Discussion

**Why the VAE is the right setup for diffusion.** A diffusion model is trained to denoise. At any intermediate step it produces a latent that is *not* one of the training latents — it's somewhere on the noise-to-data trajectory. For diffusion to work, "somewhere in between" has to be a valid place in latent space. The VAE's KL prior is precisely what guarantees that: the latent space is a continuous manifold, not a point cloud. The interpolations you produce in this lab are a small-scale demonstration of the same property diffusion relies on at every denoising step.

**What changes for DiT.** Stable Diffusion's VAE differs from this one in three operational ways but is conceptually identical:

| Property | This lab | Stable Diffusion VAE |
|---|---|---|
| Input | 28×28×1 (grayscale) | typically 256×256×3 (RGB) |
| Latent | flat 16-dim vector | spatial: `(4, 32, 32)` for a 256×256 input — 8× spatial downsampling, 4 channels |
| `beta` | 1.0 (textbook ELBO) | ≈ `1e-6` (tiny — diffusion provides the regularization) |
| Decoder loss | BCE on Bernoulli pixels | reconstruction (MSE/L1) + perceptual (LPIPS) + adversarial (PatchGAN) |
| Training | minutes on a laptop | days on a multi-GPU cluster |

The encoder/decoder, posterior `(mu, logvar)`, reparameterization trick, KL term — all of them carry over unchanged. When DiT eventually loads Stable Diffusion's VAE to encode images for the diffusion model, it is using the same compute graph you wrote here; just bigger and trained on far more data.

# Module 2.1 — VAE

> Part 2 — Diffusion Essentials · [DiT from Scratch](../README.md)

**Goal**: turn lab 1.2's deterministic autoencoder into a **Variational** Autoencoder by adding the two pieces that make the latent space generative — a Gaussian posterior `(mu, logvar)` and a regularizer that pulls it toward `N(0, I)`. Train on MNIST, then verify the latent space is well-formed by (a) sampling `z ~ N(0, I)` → decode and (b) interpolating between two encoded digits.

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

| | Lab 1.2 (AE) | Lab 2.1 (VAE) |
| --- | --- | --- |
| Dataset | CIFAR-10, (3, 32, 32) | MNIST, (1, 28, 28) |
| Latent dim | 256 | 16 |
| Encoder downsample stages | 3 (32 → 16 → 8 → 4) | 2 (28 → 14 → 7) |
| Encoder bottleneck head | **1** Linear → `z` | **2** Linears → `mu`, `logvar` ★ |
| Sampling | none — `z` is deterministic | `z = mu + σ·eps`, reparameterized ★ |
| Decoder output | `Sigmoid` → pixels in [0, 1] | logits (sigmoid applied at loss time) |
| Norm in conv blocks | BatchNorm2d everywhere | none |
| Loss | reconstruction only | reconstruction **+ KL term** ★ |

The **★** rows are the three changes that define a VAE. Everything else is incidental — different dataset (CIFAR is harder; MNIST is good enough at smaller capacity), different normalization choice (BatchNorm helps with deeper RGB stacks; tiny grayscale MNIST converges fine without it), one fewer downsample stage (28 = 7 × 4 wants 2 stages, 32 = 4 × 8 wants 3). **The VAE *idea* is just: make the encoder output a normal distribution (`mu`, `logvar` defining `N(mu, σ²·I)`), sample `z` from it, and add a KL term that anchors that distribution to a known prior.**

(For DiT later, this same encoder/decoder pattern shows up at SD-VAE scale: a few-channel spatial latent → conv stack → multiple transposed-conv stages → 256×256 RGB. Different sizes, same recipe, same KL.)

## What the model learns

**MNIST**: 28×28 grayscale digits, 60k train / 10k test. Small and clean enough that the latent space is interpretable in a single afternoon's training.

The architecture is lab 1.2's encoder/decoder, with two changes:

1. **The encoder outputs two vectors** instead of one — `mu` and `logvar`, the mean and log-variance of a Gaussian over the latent. We sample
   ```
   z = mu + σ * eps,    eps ~ N(0, I),    σ = exp(0.5 * logvar)
   ```
   This is the **reparameterization trick**: the random sample is rewritten as a deterministic function of `(mu, σ)` plus an external noise source `eps`, so backprop can flow through `mu` and `σ`. Sampling directly would sever the gradient.

2. **The loss adds a KL term** that pulls the per-image posterior `N(mu, σ²)` toward the standard normal prior `N(0, I)`:
   ```
   L = recon(x, x̂) + beta * KL(N(mu, σ²) || N(0, I))
   ```
   For two Gaussians the KL has a closed form:
   ```
   KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
   ```
   Without this term the encoder is free to spread points anywhere in latent space — there's no reason for two adjacent latents to decode to similar images. With it, the latents pile up near the origin, the prior matches the aggregate posterior, and `N(0, I)` becomes a usable sampling distribution.

KL stands for Kullback–Leibler divergence, after Solomon Kullback and Richard Leibler (1951). It measures how different two probability distributions are:

```
KL(P || Q) = ∫ p(x) · log( p(x) / q(x) ) dx
```

Three properties to know:

- **Non-negative**: `KL(P || Q) ≥ 0`, with equality iff `P = Q`. Minimizing it pulls `P` toward `Q`.
- **Asymmetric**: `KL(P || Q) ≠ KL(Q || P)` in general. The first argument is "the distribution being measured," the second is "the reference."
- **Information-theoretic reading**: it's the expected number of *extra* bits (or nats) you'd need to encode samples from `P` if you used a code optimized for `Q`. Zero if your code is already optimal; large if `P` puts mass where `Q` doesn't.

The asymmetry is why the VAE uses `KL(q || p)` specifically (encoder posterior measured *against* the prior, not the other way around). `KL(q || p)` is **mode-seeking** — heavily penalizes `q` putting mass where `p` doesn't, but tolerates `q` covering only part of `p`. The reverse, `KL(p || q)`, is **mode-covering** and would force `q` to spread out and cover all of `p`. Different choice, different VAE behavior. The standard VAE uses `KL(q || p)` because it falls out of the ELBO derivation and has the closed form above for Gaussians.

```
 input image                     latent                       reconstructed image
(1, 28, 28)  →  [Encoder]  →  (mu, logvar)  →  z = mu + σ·ε  →  [Decoder]  →  (1, 28, 28)
                                  │
                                  └──── KL pulls (mu, logvar) toward (0, 0)
```

`beta = 1` is standard ELBO. `beta > 1` makes the KL weight heavier — smoother latent space, blurrier reconstructions (the **β-VAE** knob). `beta < 1` is the opposite trade-off.

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
python -m venv venv && source venv/bin/activate
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

### 3. The math, briefly

Three ideas to lock in.

**Why a sample, not just `mu`?** A deterministic autoencoder maps every image to a single point. If you decode any *other* point — say, `mu + 0.1·v` — you get nothing useful, because nothing during training trained the decoder to handle non-encoded points. The reparameterized sample injects noise *during training*, forcing the decoder to be robust around each `mu`. Combined with the KL term, this turns the latent space into a continuous manifold rather than a finite set of points.

**Why `logvar`, not `σ`?** A Linear layer can output any real value. `logvar` can be negative (small variance) or positive (large variance), no constraint needed. `σ` would have to be strictly positive — you'd need an exp/softplus head and worry about numerical precision. Using `logvar` is the standard VAE convention and the math is cleaner: `σ = exp(0.5 · logvar)`.

**Why `beta`?** Different tasks want different priors. For pure generative quality you want `beta = 1` (the proper ELBO). For learning *disentangled* latent dimensions you want `beta > 1` (β-VAE: each dim of `z` ends up encoding a single semantic factor, at the cost of recon quality). For sharp reconstructions only, `beta < 1` (relaxes the KL pressure). The Stable Diffusion VAE uses `beta ≈ 1e-6` — a tiny KL weight, because the diffusion model on top of it provides most of the regularization. We default to `beta = 1` because it's the textbook setup.

### 4. Reconstruct, sample, interpolate

```bash
python visualize.py --mode all
```

This produces three figures in `lab2.1/`:

- `reconstructions.png` — top row originals, bottom row VAE reconstructions. Should look like blurry-but-recognizable digits. Reconstruction quality is similar to lab 1.2's autoencoder; the VAE does not reconstruct *better* than an autoencoder. The point of the VAE is what the latent space *can do*, not how well it reconstructs.
- `samples.png` — 64 digits decoded from `z ~ N(0, I)`. **This is the strictest test of the KL term.** A trained VAE produces plausible digits here. An autoencoder without KL would produce noise: random points in its latent space don't correspond to anything the decoder was trained on.
- `interpolation.png` — two random test digits at the ends, eight intermediate frames in between. Walking linearly between `mu_a` and `mu_b` should morph smoothly through digit shapes. Without the KL prior, the line between two valid latents passes through "off-manifold" regions and the intermediate frames look like noise.

To run a single visualization:

```bash
python visualize.py --mode reconstruct --n 12
python visualize.py --mode sample
python visualize.py --mode interpolate --n 16
```

### 5. Experiment with `beta`

Train three models at different KL weights and compare:

```bash
python train.py --beta 0.1 --save-path vae_beta0.1.pt
python train.py --beta 1.0 --save-path vae_beta1.pt   # default
python train.py --beta 4.0 --save-path vae_beta4.pt
```

Then visualize each:

```bash
python visualize.py --ckpt vae_beta0.1.pt --save-prefix beta0.1_
python visualize.py --ckpt vae_beta1.pt   --save-prefix beta1_
python visualize.py --ckpt vae_beta4.pt   --save-prefix beta4_
```

What to look for:

- **`beta=0.1` reconstructions are sharpest** but **prior samples look worst** — the latent space is barely regularized, so `N(0, I)` is mostly off-manifold.
- **`beta=4.0` prior samples look smoothest** but **reconstructions are blurrier** — the encoder is so squeezed toward `N(0, I)` that fine detail can't survive the bottleneck.
- **`beta=1.0` is the balanced default**.

The trade-off — recon quality vs latent-space quality — is the whole reason the β-VAE knob exists.

### 6. Experiment with `latent_dim`

```bash
python train.py --latent-dim 2   --save-path vae_z2.pt    # extreme bottleneck
python train.py --latent-dim 16  --save-path vae_z16.pt   # default
python train.py --latent-dim 64  --save-path vae_z64.pt
```

`latent_dim=2` is so tight that you can plot the entire latent space on a 2-D scatter, color-coded by digit class — a classic VAE figure. Reconstructions at `latent_dim=2` are markedly blurrier (most digit information cannot fit through 2 numbers), but interpolations are particularly clean because the manifold is dense in 2-D.

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

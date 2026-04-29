# Variational Autoencoder, from scratch.
#
# This file defines WHAT the VAE computes.
# train.py defines HOW it is trained on MNIST.
# visualize.py inspects the learned latent space (reconstruction, sampling, interpolation).
#
# The architecture is lab 1.2's autoencoder with two additions:
#   1. The encoder outputs TWO vectors instead of one — `mu` and `logvar`,
#      the mean and log-variance of a Gaussian over the latent. We sample
#          z = mu + sigma * eps,  eps ~ N(0, I),  sigma = exp(0.5 * logvar)
#      This is the "reparameterization trick": it keeps the random sample
#      differentiable w.r.t. mu and sigma, so backprop can flow through.
#   2. The loss adds a KL term that pulls the per-image posterior
#      N(mu, sigma^2) toward the standard normal prior N(0, I). This
#      regularizes the latent space so that random draws from N(0, I)
#      decode to plausible images, and so that interpolating between two
#      latents produces smooth transitions through the data manifold.
#
# Why this matters for DiT:
#   DiT operates on latents produced by a *spatial* VAE (8x downsample,
#   4 channels). The principles — Gaussian posterior, KL regularization,
#   smooth latent space — are identical here; only the encoder/decoder
#   shapes differ. Without KL, a latent space is a bag of unrelated points
#   that diffusion cannot navigate. With it, the space is a continuous
#   manifold that diffusion can traverse step by step.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """28x28x1 -> two latent_dim vectors (mu, logvar).

    We output `logvar = log(sigma^2)` rather than `sigma` directly because:
    - `logvar` can take any real value; a Linear layer's raw output works.
    - `sigma` must be strictly positive; would need an exp() / softplus.
    Using logvar is numerically stabler and the standard VAE convention.
    """

    def __init__(self, latent_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """latent_dim -> 28x28x1 logits.

    Outputs *logits* (no sigmoid). The training loss is
    `binary_cross_entropy_with_logits`, which is numerically stabler than
    sigmoid + BCE. Apply sigmoid yourself when visualizing.
    """

    def __init__(self, latent_dim=16):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 14 -> 28
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(h)


def reparameterize(mu, logvar):
    """The reparameterization trick.

    A naive sample `z ~ N(mu, sigma^2)` breaks backprop: sampling severs the
    gradient path from z back to mu and sigma. The trick rewrites the sample
    as a deterministic function of (mu, sigma) plus an external noise source:

        z = mu + sigma * eps,    eps ~ N(0, I)

    Now z's gradient flows through mu and sigma; eps is just data, not a
    learned thing. Mathematically identical distribution to the naive sample.
    """
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn_like(sigma)
    return mu + sigma * eps


def vae_loss(x, x_recon_logits, mu, logvar, beta=1.0):
    """Negative ELBO, summed over a batch (we minimize this).

    Two terms:

    1. **Reconstruction (BCE)** — how well the decoder reconstructs each
       pixel as a Bernoulli. Summed over pixels (not averaged) so the
       magnitude is comparable to the KL term.

    2. **KL divergence** — closed form for two Gaussians:
            KL(N(mu, sigma^2) || N(0, I))
                = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
       Pulls the per-image posterior toward the standard normal prior.
       Without this term, the encoder is free to spread points anywhere
       and N(0, I) sampling gives garbage.

    `beta` scales the KL weight (the "beta-VAE" knob):
        beta = 1   : standard ELBO.
        beta > 1   : more regularized — smoother latent space, blurrier recons.
        beta < 1   : sharper recons but a less-structured latent space.
    """
    recon = F.binary_cross_entropy_with_logits(x_recon_logits, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


class VAE(nn.Module):
    """Encoder + reparameterize + Decoder.

    The model has exactly the same compute graph as a deterministic
    autoencoder *except* for the sample step in the middle. That single
    stochastic step, plus the KL term in the loss, is what makes the
    latent space generative.
    """

    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_recon_logits = self.decoder(z)
        return x_recon_logits, mu, logvar

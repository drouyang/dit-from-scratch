"""Frozen Stable Diffusion VAE for latent compression / reconstruction.

Production text-to-image models (SD1.x/2.x/XL, SD3, FLUX) all use a 2D spatial
VAE that compresses 256×256×3 images to ~32×32×4 latents — an 8× spatial
downsample with 4 channels. The DiT then operates on those compressed latents.

We use `stabilityai/sd-vae-ft-mse`, the canonical SD 1.x VAE. It works well at
any resolution that's a multiple of 8; for our 64×64 images it produces 8×8×4
latents.

Frozen: the VAE was pretrained at scale on natural images. We don't update it.
The DiT learns to navigate the latent space the VAE defines.

Two things to know about SD-VAE's latent scale:
    - The encoder returns a Gaussian distribution over the latent. We use its
      mean (`mu`) — sampling z ~ N(mu, sigma) is also valid but introduces
      extra noise that diffusion training doesn't need.
    - The latent has a known scale factor (`0.18215`). Multiply by this when
      encoding, divide when decoding. The number comes from SD's training
      pipeline — it normalizes the latent's variance to roughly unit-scale,
      which makes the diffusion model's `N(0, I)` noise prior compatible.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

SCALE_FACTOR = 0.18215  # SD-VAE convention; do not change


class SDVae(nn.Module):
    """Wrapper around diffusers' AutoencoderKL with the standard scale factor."""

    def __init__(self, model_name="stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, x):
        """Encode images in [-1, 1] to latents.

        Args:
            x: (B, 3, H, W) image tensor in [-1, 1]
        Returns:
            z: (B, 4, H/8, W/8) latent tensor, scale-normalized
        """
        # AutoencoderKL.encode returns an AutoencoderKLOutput whose `latent_dist`
        # is a DiagonalGaussian. Take the mean (no sampling needed for training
        # diffusion — the noise prior handles randomness).
        posterior = self.vae.encode(x).latent_dist
        z = posterior.mean * SCALE_FACTOR
        return z

    @torch.no_grad()
    def decode(self, z):
        """Decode latents back to images in [-1, 1].

        Args:
            z: (B, 4, H/8, W/8) latent tensor (scale-normalized)
        Returns:
            x: (B, 3, H, W) image tensor in [-1, 1]
        """
        return self.vae.decode(z / SCALE_FACTOR).sample

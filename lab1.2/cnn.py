# CNN autoencoder for CIFAR-10 reconstruction.
#
# This file defines WHAT the network computes.
# train.py defines HOW it is trained (loss, optimizer, loop).
#
# Architecture overview:
#   Encoder: 3 stride-2 conv blocks compress (3, 32, 32) → a latent_dim-vector.
#   Decoder: linear projection + 3 transposed conv blocks expand back to (3, 32, 32).
#
# Why this matters for DiT:
#   The VAE that DiT operates in has exactly this encoder/decoder structure.
#   The encoder compresses a full-resolution image into a compact latent space;
#   DiT denoises in that space; the decoder reconstructs the image from the
#   denoised latent. The Conv2d → ConvTranspose2d pattern and the bottleneck
#   linear layer here are the literal building blocks of VAE encoders/decoders.

import torch
import torch.nn as nn  # all layer types live here


class Encoder(nn.Module):
    """Three conv blocks that compress a 32×32 RGB image to a latent vector.

    Each block halves the spatial resolution (stride=2) and doubles the channels.
    The final spatial feature map is flattened and projected to latent_dim.
    """

    def __init__(self, latent_dim=256):
        # latent_dim: width of the bottleneck vector. Smaller = more compression,
        #             lower reconstruction quality. Directly analogous to the latent
        #             dimension in a VAE.
        super().__init__()

        # ── How Conv2d works ──────────────────────────────────────────────────
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #
        # At every spatial position (i, j) in the output, the layer applies the
        # *same* learnable kernel (a small weight matrix) to a local patch of the
        # input — this is the "sliding window." The same kernel weights are reused
        # across all positions: this is *parameter sharing*, and it gives CNNs
        # translation invariance (a cat in the top-left vs bottom-right → same filters).
        #
        # Multiple out_channels = multiple independent kernels running in parallel.
        # Each kernel learns to detect a different local feature (edge, color patch, ...).
        # The output tensor has shape (B, out_channels, out_H, out_W).
        #
        # Spatial output size formula:
        #   out_H = floor((in_H + 2*padding - kernel_size) / stride) + 1
        #
        # With kernel_size=4, stride=2, padding=1:
        #   out_H = floor((in_H + 2 - 4) / 2) + 1 = in_H // 2
        #   → each block HALVES spatial dims.
        #
        # ── Why stride-2 conv instead of Conv + MaxPool? ──────────────────────
        # MaxPool2d takes the maximum of a 2×2 window — a fixed, non-learnable op
        # that discards the 3 non-max values. Stride-2 conv is a LEARNABLE
        # downsampler: the network decides how to combine the 4-pixel window into
        # one output value. For autoencoders (and VAEs) we need to reconstruct the
        # image from the bottleneck, so losing information to a fixed operation is
        # worse than letting the network learn a good compression.
        self.encoder = nn.Sequential(
            # Block 1: (B, 3, 32, 32) → (B, 32, 16, 16)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            # BatchNorm2d normalizes the activations of each channel across the
            # batch and spatial dimensions, keeping them near zero-mean/unit-variance.
            # This stabilizes training (less sensitivity to init / lr) and acts as
            # a mild regularizer. Required before ReLU so values aren't all negative.
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 2: (B, 32, 16, 16) → (B, 64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3: (B, 64, 8, 8) → (B, 128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Collapse spatial dims into a flat vector.
            # (B, 128, 4, 4) → (B, 128*4*4) = (B, 2048)
            nn.Flatten(),

            # Project to the bottleneck. All image content must pass through here.
            # In a VAE, this layer is replaced by TWO linear layers: one outputs
            # the mean μ and one outputs log-variance log σ² of the latent distribution.
            # z is then sampled as z = μ + σ * ε, ε ~ N(0,1).
            nn.Linear(128 * 4 * 4, latent_dim),
        )

    def forward(self, x):
        # x: (B, 3, 32, 32)  →  returns z: (B, latent_dim)
        return self.encoder(x)


class Decoder(nn.Module):
    """Three transposed conv blocks that expand a latent vector back to 32×32 RGB.

    Mirror image of the Encoder: project → reshape → 3× spatial upsampling.
    """

    def __init__(self, latent_dim=256):
        super().__init__()

        # ── How ConvTranspose2d works ─────────────────────────────────────────
        # ConvTranspose2d is the learnable inverse of a strided Conv2d.
        # Conceptually: insert (stride-1) zeros between input values (upsampling
        # by stride), then apply a regular convolution with the same kernel.
        # This is sometimes called "fractionally-strided conv" or (loosely)
        # "deconvolution." Output size:
        #   out_H = (in_H - 1) * stride - 2*padding + kernel_size
        #
        # With kernel_size=4, stride=2, padding=1:
        #   out_H = (in_H - 1) * 2 - 2 + 4 = 2 * in_H
        #   → each block DOUBLES spatial dims — the exact inverse of our encoder.
        self.decoder = nn.Sequential(
            # Project latent back to a spatial feature map.
            # (B, latent_dim) → (B, 2048)
            nn.Linear(latent_dim, 128 * 4 * 4),

            # Unflatten restores the spatial structure: (B, 2048) → (B, 128, 4, 4).
            # This gives ConvTranspose2d the 2D layout it needs to upsample spatially.
            nn.Unflatten(1, (128, 4, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 1: (B, 128, 4, 4) → (B, 64, 8, 8)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 2: (B, 64, 8, 8) → (B, 32, 16, 16)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 3: (B, 32, 16, 16) → (B, 3, 32, 32)
            # Sigmoid squashes output to [0, 1] to match the ToTensor input range.
            # No BatchNorm before the output — we want the final pixel values in
            # the correct range, not re-centered.
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # z: (B, latent_dim)  →  returns x_hat: (B, 3, 32, 32)
        return self.decoder(z)


class Classifier(nn.Module):
    """CNN image classifier: same Encoder backbone, linear head instead of Decoder.

    Compresses (3, 32, 32) → latent_dim via the Encoder, then maps latent_dim → 10
    class logits. The encoder weights are identical to the one in Autoencoder — this
    shows that the same conv features can serve reconstruction *or* classification.

    Training uses CrossEntropyLoss (multi-class, not MSE). There is no decoder.
    """

    def __init__(self, latent_dim=256, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        # Linear classifier head: maps the bottleneck vector to class logits.
        # CrossEntropyLoss applies softmax internally, so we output raw logits here.
        self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        # x: (B, 3, 32, 32)  →  logits: (B, num_classes)
        z = self.encoder(x)
        return self.head(z)


class Autoencoder(nn.Module):
    """Encoder + Decoder stacked into a single model.

    forward() returns BOTH the reconstruction AND the latent code, so
    train.py can compute the reconstruction loss and inspect the latent space.
    """

    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        # x:     (B, 3, 32, 32)  — input images
        # z:     (B, latent_dim) — compressed representation (the latent code)
        # x_hat: (B, 3, 32, 32)  — reconstructed images
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

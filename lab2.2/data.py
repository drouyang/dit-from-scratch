"""8 Gaussians toy dataset.

Eight unit-variance Gaussians arranged in a circle. Class label = which mode.
This is the standard toy for demonstrating diffusion / flow matching:
  - Visualizable as a 2-D scatter
  - Multi-modal so generation is non-trivial
  - Class labels make CFG meaningful
  - Cheap enough to train in seconds
"""

import math

import torch


NUM_CLASSES = 8


def mode_centers(radius=5.0):
    """The 8 cluster centers, on a circle of given radius."""
    angles = torch.arange(NUM_CLASSES) * (2 * math.pi / NUM_CLASSES)
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)


def sample_8gaussians(n, std=0.3, radius=5.0, device="cpu"):
    """Sample n points from 8 Gaussians arranged in a circle.

    Returns:
        x: (n, 2) sampled points
        y: (n,) class labels in {0, ..., 7}
    """
    centers = mode_centers(radius=radius).to(device)
    y = torch.randint(0, NUM_CLASSES, (n,), device=device)
    x = centers[y] + std * torch.randn(n, 2, device=device)
    return x, y

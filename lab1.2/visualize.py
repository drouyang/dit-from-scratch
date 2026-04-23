"""Load a trained autoencoder and plot original vs reconstructed images.

Usage:
    python visualize.py                          # defaults
    python visualize.py --latent-dim 32          # if trained with non-default latent dim
    python visualize.py --n 16 --save out.png    # more images, custom output path
"""

import argparse

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from cnn import Autoencoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="cnn_autoencoder.pt")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--n", type=int, default=8, help="number of images per row")
    p.add_argument("--save", default="reconstructions.png")
    args = p.parse_args()

    # Load model
    model = Autoencoder(latent_dim=args.latent_dim)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Load the first n test images. shuffle=False → same images every run.
    test_set = datasets.CIFAR10(args.data_dir, train=False, download=True,
                                transform=transforms.ToTensor())
    images = torch.stack([test_set[i][0] for i in range(args.n)])  # (n, 3, 32, 32)

    # Run through the autoencoder (no gradients needed — we're just inspecting).
    with torch.no_grad():
        recons, z = model(images)  # recons: (n, 3, 32, 32), z: (n, latent_dim)

    # Print a quick latent-space summary to show the bottleneck at work.
    print(f"latent codes shape: {z.shape}")
    print(f"latent mean: {z.mean():.3f}, std: {z.std():.3f}, "
          f"min: {z.min():.3f}, max: {z.max():.3f}")

    # Plot: top row = originals, bottom row = reconstructions.
    # .permute(1, 2, 0) converts (C, H, W) → (H, W, C) for matplotlib.
    # .clamp(0, 1) guards against any tiny out-of-range values from floating point.
    fig, axes = plt.subplots(2, args.n, figsize=(args.n * 1.5, 3.5))
    for i in range(args.n):
        axes[0, i].imshow(images[i].permute(1, 2, 0).clamp(0, 1))
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].permute(1, 2, 0).clamp(0, 1))
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("original", fontsize=9)
    axes[1, 0].set_ylabel("reconstructed", fontsize=9)
    fig.suptitle(
        f"CNN Autoencoder — CIFAR-10 reconstructions  (latent_dim={args.latent_dim})",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(args.save, dpi=120, bbox_inches="tight")
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()

"""Visualize a trained VAE three ways:

  1. reconstruct  — encode a test image, sample z, decode → compare
  2. sample       — draw z ~ N(0, I), decode → grid of "imagined" digits
  3. interpolate  — pick two test images, walk linearly between their means

Run:
    python visualize.py --mode all      # produces all three figures
    python visualize.py --mode sample   # only sampled-from-prior grid
    python visualize.py --mode interpolate --n 12

The three figures together tell you whether the latent space is well-formed:
  - reconstructions should look like digits
  - samples should look like digits (this is the strict test of KL working)
  - interpolations should morph smoothly through digit shapes
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from vae import VAE


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = VAE(latent_dim=ckpt["config"]["latent_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def reconstruct(model, n=8, save="reconstructions.png", device="cpu"):
    """Top row: originals. Bottom row: VAE reconstructions."""
    test_ds = datasets.MNIST("./data", train=False, download=True,
                             transform=transforms.ToTensor())
    indices = torch.randperm(len(test_ds))[:n]
    x = torch.stack([test_ds[i][0] for i in indices]).to(device)

    x_recon_logits, _, _ = model(x)
    x_recon = torch.sigmoid(x_recon_logits).cpu().numpy()
    x_np = x.cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.6))
    for i in range(n):
        axes[0, i].imshow(x_np[i, 0], cmap="gray")
        axes[1, i].imshow(x_recon[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original",      rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("reconstruction", rotation=0, ha="right", va="center")
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


@torch.no_grad()
def sample(model, n=64, save="samples.png", device="cpu"):
    """Sample z ~ N(0, I), decode each.

    If the KL term did its job, these look like plausible digits drawn from
    the data distribution. If the latent space is unregularized, they look
    like noise — so this is the strictest test that the prior matches the
    aggregate posterior.
    """
    z = torch.randn(n, model.latent_dim, device=device)
    x_logits = model.decoder(z)
    x = torch.sigmoid(x_logits).cpu().numpy()

    grid = int(np.sqrt(n))
    assert grid * grid == n, "pass a perfect square for n"
    fig, axes = plt.subplots(grid, grid, figsize=(grid, grid))
    for i in range(grid):
        for j in range(grid):
            axes[i, j].imshow(x[i * grid + j, 0], cmap="gray")
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


@torch.no_grad()
def interpolate(model, n_steps=10, save="interpolation.png", device="cpu"):
    """Encode two random test digits, walk linearly between their means.

    A trained VAE produces a smooth path through digit space. A
    deterministic autoencoder (without KL) produces abrupt jumps or
    garbage in the middle: the points along the line between two valid
    latents are not themselves valid latents.
    """
    test_ds = datasets.MNIST("./data", train=False, download=True,
                             transform=transforms.ToTensor())
    i_a, i_b = torch.randperm(len(test_ds))[:2].tolist()
    x = torch.stack([test_ds[i_a][0], test_ds[i_b][0]]).to(device)
    mu, _ = model.encoder(x)

    ts = torch.linspace(0, 1, n_steps, device=device).unsqueeze(1)  # (n_steps, 1)
    z = (1 - ts) * mu[0] + ts * mu[1]                                # (n_steps, latent_dim)
    x_logits = model.decoder(z)
    x_interp = torch.sigmoid(x_logits).cpu().numpy()

    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.2, 1.4))
    for i in range(n_steps):
        axes[i].imshow(x_interp[i, 0], cmap="gray")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        default="vae.pt")
    p.add_argument("--mode",        choices=["reconstruct", "sample", "interpolate", "all"],
                                    default="all")
    p.add_argument("--save-prefix", default="")
    p.add_argument("--n",           type=int, default=8,
                   help="reconstruct: how many; sample: n=64 (square); interpolate: steps")
    p.add_argument("--seed",        type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    model = load_model(args.ckpt, device)

    pfx = args.save_prefix
    if args.mode in ("reconstruct", "all"):
        reconstruct(model, n=args.n, save=f"{pfx}reconstructions.png", device=device)
    if args.mode in ("sample", "all"):
        sample(model, n=64, save=f"{pfx}samples.png", device=device)
    if args.mode in ("interpolate", "all"):
        steps = args.n if args.mode == "interpolate" else 10
        interpolate(model, n_steps=steps, save=f"{pfx}interpolation.png", device=device)


if __name__ == "__main__":
    main()

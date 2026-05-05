"""DiT visualizations on MNIST.

Three figures, all the image-space analogue of lab 2.2's plots:

  - samples : N per-class samples, one row per class                    (samples.png)
  - cfg     : same noise across CFG scale {0, 1, 2, 4, 7}               (cfg.png)
  - steps   : same noise across step count {1, 2, 4, 8, 16, 50}         (steps.png)

Run:
    python visualize.py --mode all
    python visualize.py --mode steps
"""

import argparse

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from data import IMAGE_SIZE, IN_CHANNELS, NUM_CLASSES, denormalize
from dit import DiT
from flow import fm_euler_sample


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DiT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _to_grid(x, nrow):
    """(N, C, H, W) in [-1, 1]  →  (H', W', C) numpy in [0, 1] for imshow."""
    x = denormalize(x).clamp(0, 1).cpu()
    grid = make_grid(x, nrow=nrow, padding=2, pad_value=1.0)
    return grid.permute(1, 2, 0).numpy()


def _imshow(ax, img):
    if img.shape[-1] == 1:
        ax.imshow(img.squeeze(-1), cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])


@torch.no_grad()
def fig_samples(model, save, device, n_per_class=8, n_steps=50,
                cfg_scale=4.0, seed=0):
    """N samples per class, one row per class."""
    torch.manual_seed(seed)
    classes = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_per_class)
    x = fm_euler_sample(
        model, classes.size(0), n_steps=n_steps,
        shape=(IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
        classes=classes, cfg_scale=cfg_scale, device=device,
    )
    img = _to_grid(x, nrow=n_per_class)

    fig, ax = plt.subplots(figsize=(n_per_class * 0.85, NUM_CLASSES * 0.85))
    _imshow(ax, img)
    # Label each row with its class index.
    cell = IMAGE_SIZE + 2  # patch + padding
    ax.set_yticks([cell * (i + 0.5) for i in range(NUM_CLASSES)],
                  [str(i) for i in range(NUM_CLASSES)])
    ax.set_ylabel("class")
    ax.set_title(f"samples  (steps={n_steps}, cfg={cfg_scale})")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


@torch.no_grad()
def fig_cfg(model, save, device, n_per_class=4, n_steps=50,
            cfg_grid=(0.0, 1.0, 2.0, 4.0, 7.0), seed=0):
    """Same starting noise across a CFG sweep — see conditioning sharpen."""
    fig, axes = plt.subplots(1, len(cfg_grid),
                             figsize=(2.6 * len(cfg_grid), 0.6 * NUM_CLASSES + 1))
    for ax, s in zip(axes, cfg_grid):
        torch.manual_seed(seed)
        classes = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_per_class)
        x = fm_euler_sample(
            model, classes.size(0), n_steps=n_steps,
            shape=(IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
            classes=classes, cfg_scale=s, device=device,
        )
        img = _to_grid(x, nrow=n_per_class)
        _imshow(ax, img)
        ax.set_title(f"cfg = {s}")
    fig.suptitle(f"CFG sweep  (steps={n_steps}; rows = class 0–9)")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


@torch.no_grad()
def fig_steps(model, save, device, n_per_class=4, cfg_scale=4.0,
              step_grid=(1, 2, 4, 8, 16, 50), seed=0):
    """Same noise across N — show how few Euler steps suffice."""
    fig, axes = plt.subplots(1, len(step_grid),
                             figsize=(2.6 * len(step_grid), 0.6 * NUM_CLASSES + 1))
    for ax, n_steps in zip(axes, step_grid):
        torch.manual_seed(seed)
        classes = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_per_class)
        x = fm_euler_sample(
            model, classes.size(0), n_steps=n_steps,
            shape=(IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
            classes=classes, cfg_scale=cfg_scale, device=device,
        )
        img = _to_grid(x, nrow=n_per_class)
        _imshow(ax, img)
        ax.set_title(f"steps = {n_steps}")
    fig.suptitle(f"step-count sweep  (cfg={cfg_scale}; rows = class 0–9)")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      default="dit.pt")
    p.add_argument("--mode",      choices=["samples", "cfg", "steps", "all"],
                                  default="all")
    p.add_argument("--steps",     type=int,   default=50)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--seed",      type=int,   default=0)
    args = p.parse_args()

    device = get_device()
    model = load_model(args.ckpt, device)

    if args.mode in ("samples", "all"):
        fig_samples(model, "samples.png", device,
                    n_steps=args.steps, cfg_scale=args.cfg_scale, seed=args.seed)
    if args.mode in ("cfg", "all"):
        fig_cfg(model, "cfg.png", device, n_steps=args.steps, seed=args.seed)
    if args.mode in ("steps", "all"):
        fig_steps(model, "steps.png", device,
                  cfg_scale=args.cfg_scale, seed=args.seed)


if __name__ == "__main__":
    main()

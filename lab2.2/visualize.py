"""Visualize a trained denoiser on 8 Gaussians.

Modes:
  1. samples    — scatter of generated samples colored by class, with mode
                  centers overlaid for reference
  2. trajectory — show the Euler/ancestral trajectories of a few samples
                  flowing from noise (t=1) to data (t=0)
  3. steps      — same model, same noise, swept across step counts
                  {1, 2, 4, 8, 16, 50}; visualizes how few steps you can get
                  away with (the FM headline result)
  4. cfg        — same model, same noise, swept across CFG scales
                  {0.0, 1.0, 3.0, 7.0}; visualizes how CFG concentrates samples
                  toward their conditional mode

Run:
    python visualize.py --mode all
    python visualize.py --mode steps    --ckpt model_fm.pt
    python visualize.py --mode cfg      --ckpt model_fm.pt
"""

import argparse

import matplotlib.pyplot as plt
import torch

from data import NUM_CLASSES, mode_centers
from flow import fm_euler_sample, ddpm_sample, DDPMSchedule
from mlp import TimeMLP


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TimeMLP(num_classes=ckpt["config"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def sample_from_ckpt(model, ckpt, n_per_class, n_steps, cfg_scale, device,
                     return_trajectory=False, classes=None):
    if classes is None:
        classes = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_per_class)
    if ckpt["paradigm"] == "fm":
        return fm_euler_sample(
            model, classes.size(0), n_steps=n_steps, dim=2,
            classes=classes, cfg_scale=cfg_scale, device=device,
            return_trajectory=return_trajectory,
        ), classes
    else:
        T = ckpt.get("ddpm_T", 100)
        sched = DDPMSchedule(T=T).to(device)
        x = ddpm_sample(
            model, sched, classes.size(0), dim=2,
            classes=classes, cfg_scale=cfg_scale, device=device,
        )
        return x, classes


def _add_centers(ax, radius=5.0):
    centers = mode_centers(radius=radius)
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=80, c="black",
               linewidths=2, zorder=10, label="true centers")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")


@torch.no_grad()
def fig_samples(model, ckpt, save, device, n_per_class=200, n_steps=50, cfg_scale=1.0):
    out, classes = sample_from_ckpt(model, ckpt, n_per_class, n_steps, cfg_scale, device)
    pts = out.cpu().numpy()
    cls = classes.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=cls, cmap="tab10", s=8, alpha=0.6)
    _add_centers(ax)
    ax.set_title(f"samples  ({ckpt['paradigm']}, steps={n_steps}, cfg={cfg_scale})")
    plt.colorbar(sc, ticks=range(NUM_CLASSES), label="class")
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


@torch.no_grad()
def fig_trajectory(model, ckpt, save, device, n_traj=24, n_steps=50, cfg_scale=1.0):
    if ckpt["paradigm"] != "fm":
        print(f"skipping trajectory (only supported for FM, got {ckpt['paradigm']})")
        return
    classes = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_traj // NUM_CLASSES)
    _, traj = fm_euler_sample(
        model, classes.size(0), n_steps=n_steps, dim=2,
        classes=classes, cfg_scale=cfg_scale, device=device,
        return_trajectory=True,
    )
    traj = traj.cpu().numpy()  # (n_steps+1, n_samples, 2)
    cls = classes.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab10")
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 0], traj[:, i, 1], c=cmap(cls[i] / NUM_CLASSES),
                alpha=0.4, linewidth=0.8)
        # Endpoint dot.
        ax.scatter(traj[-1, i, 0], traj[-1, i, 1], c=[cmap(cls[i] / NUM_CLASSES)],
                   s=20, edgecolor="black", linewidth=0.5, zorder=5)
    _add_centers(ax)
    ax.set_title(f"trajectories  ({ckpt['paradigm']}, steps={n_steps})")
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


@torch.no_grad()
def fig_steps(model, ckpt, save, device, n_per_class=100, cfg_scale=1.0,
              step_grid=(1, 2, 4, 8, 16, 50)):
    fig, axes = plt.subplots(1, len(step_grid), figsize=(3 * len(step_grid), 3.5))
    for ax, n_steps in zip(axes, step_grid):
        torch.manual_seed(0)
        out, classes = sample_from_ckpt(model, ckpt, n_per_class, n_steps, cfg_scale, device)
        pts = out.cpu().numpy()
        cls = classes.cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], c=cls, cmap="tab10", s=4, alpha=0.5)
        _add_centers(ax)
        ax.set_title(f"steps = {n_steps}")
    fig.suptitle(f"sample quality vs step count  ({ckpt['paradigm']})")
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


@torch.no_grad()
def fig_cfg(model, ckpt, save, device, n_per_class=200, n_steps=50,
            cfg_grid=(0.0, 1.0, 3.0, 7.0)):
    fig, axes = plt.subplots(1, len(cfg_grid), figsize=(3 * len(cfg_grid), 3.5))
    for ax, cfg_scale in zip(axes, cfg_grid):
        torch.manual_seed(0)
        out, classes = sample_from_ckpt(model, ckpt, n_per_class, n_steps, cfg_scale, device)
        pts = out.cpu().numpy()
        cls = classes.cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], c=cls, cmap="tab10", s=4, alpha=0.5)
        _add_centers(ax)
        ax.set_title(f"cfg = {cfg_scale}")
    fig.suptitle(f"CFG strength sweep  ({ckpt['paradigm']}, steps={n_steps})")
    plt.tight_layout()
    plt.savefig(save, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        default="model_fm.pt")
    p.add_argument("--mode",        choices=["samples", "trajectory", "steps", "cfg", "all"],
                                    default="all")
    p.add_argument("--save-prefix", default="")
    p.add_argument("--steps",       type=int,   default=50)
    p.add_argument("--cfg-scale",   type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    model, ckpt = load_model(args.ckpt, device)
    pfx = args.save_prefix

    if args.mode in ("samples", "all"):
        fig_samples(model, ckpt, save=f"{pfx}samples.png", device=device,
                    n_steps=args.steps, cfg_scale=args.cfg_scale)
    if args.mode in ("trajectory", "all"):
        fig_trajectory(model, ckpt, save=f"{pfx}trajectory.png", device=device,
                       n_steps=args.steps, cfg_scale=args.cfg_scale)
    if args.mode in ("steps", "all"):
        fig_steps(model, ckpt, save=f"{pfx}steps.png", device=device,
                  cfg_scale=args.cfg_scale)
    if args.mode in ("cfg", "all"):
        fig_cfg(model, ckpt, save=f"{pfx}cfg.png", device=device,
                n_steps=args.steps)


if __name__ == "__main__":
    main()

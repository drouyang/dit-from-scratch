"""Visualize a trained denoiser on 8 Gaussians, plus a couple of plots that
just show the training-data structure (no model required).

Model-based modes (need a checkpoint):
  - samples    : scatter of generated samples colored by class
  - trajectory : Euler trajectories from noise (t=1) to data (t=0)
  - steps      : same noise, swept across step counts {1, 2, 4, 8, 16, 50}
  - cfg        : same noise, swept across CFG scales {0, 1, 3, 7}

Data-only modes (no model needed):
  - data       : the 8 Gaussians distribution itself, colored by class
  - crossings  : many trajectories sharing the same conditioning class,
                 to show that they fan from the cluster to noise and
                 cross at intermediate (x, t) points

Run:
    python visualize.py --mode all
    python visualize.py --mode steps    --ckpt model_fm.pt
    python visualize.py --mode crossings   # no checkpoint needed
"""

import argparse
from pathlib import Path

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


def fig_crossings(save, target_class=3, n_trajectories=80):
    """Plot many trajectories all conditioned on the SAME class.

    Each trajectory has a different `(x_0, x_1)` pair but shares the class
    label `c`. Visualizes the fan structure — many lines starting at the
    tight cluster, fanning out to noise — and the fact that many of them
    cross at intermediate `(x_t, t, c)` points. The model averages over
    all such crossings to predict a single velocity per point.

    No model needed — this shows the *training-data* structure.
    """
    from data import sample_8gaussians  # noqa: F401  (kept for symmetry)
    torch.manual_seed(42)

    centers = mode_centers()
    center_c = centers[target_class]
    # n_trajectories random data points within cluster `target_class` (std=0.3).
    x_0 = center_c + 0.3 * torch.randn(n_trajectories, 2)
    # n_trajectories random N(0, I) noise samples.
    x_1 = torch.randn(n_trajectories, 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    # Left: full plane with all 8 cluster centers and many class-c trajectories.
    ax = axes[0]
    for i in range(n_trajectories):
        ax.plot([x_0[i, 0], x_1[i, 0]], [x_0[i, 1], x_1[i, 1]],
                color="steelblue", alpha=0.25, linewidth=0.7, rasterized=True)
    ax.scatter(x_0[:, 0], x_0[:, 1], s=15, c="steelblue", rasterized=True,
               label=f"x_0 (data, class {target_class})", zorder=5)
    ax.scatter(x_1[:, 0], x_1[:, 1], s=15, c="gray", alpha=0.6, rasterized=True,
               label="x_1 (noise)", zorder=5)
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=100, c="black",
               linewidths=2, zorder=10)
    for i in range(NUM_CLASSES):
        ax.annotate(str(i), (centers[i, 0].item(), centers[i, 1].item()),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=10, fontweight="bold")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    ax.set_title(f"{n_trajectories} trajectories with same conditioning c={target_class}\n"
                 f"(different x_0, different x_1)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Right: zoom into the crossing region with a query point highlighted.
    ax = axes[1]
    for i in range(n_trajectories):
        ax.plot([x_0[i, 0], x_1[i, 0]], [x_0[i, 1], x_1[i, 1]],
                color="steelblue", alpha=0.3, linewidth=0.7, rasterized=True)
    ax.scatter(x_0[:, 0], x_0[:, 1], s=20, c="steelblue", rasterized=True, zorder=5)
    ax.scatter(x_1[:, 0], x_1[:, 1], s=20, c="gray", alpha=0.6, rasterized=True, zorder=5)
    ax.scatter(center_c[0], center_c[1], marker="x", s=100, c="black",
               linewidths=2, zorder=10)
    query = (center_c[0].item() / 2, center_c[1].item() / 2)  # midway
    ax.scatter(*query, s=200, c="red", marker="*", zorder=15,
               label=f"query (x≈{query[0]:.2f},{query[1]:.2f}, t≈0.5)")
    ax.add_patch(plt.Circle(query, 0.4, color="red", fill=False, linewidth=2, zorder=14))
    ax.set_xlim(-3, 1)
    ax.set_ylim(-1, 4)
    ax.set_aspect("equal")
    ax.set_title("Zoom: many class-{} trajectories cross\nthrough the same neighborhood".format(target_class))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


def fig_data(save, n_per_class=200):
    """Plot the raw 8-Gaussians dataset. No model needed."""
    from data import sample_8gaussians
    torch.manual_seed(0)
    x, y = sample_8gaussians(n_per_class * NUM_CLASSES)

    fig, ax = plt.subplots(figsize=(6, 6))
    # Rasterize the scatter so the SVG stays small (axes/text remain vector).
    sc = ax.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", s=8, alpha=0.6,
                    rasterized=True)
    centers = mode_centers()
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=120, c="black",
               linewidths=2, zorder=10)
    for i in range(NUM_CLASSES):
        ax.annotate(str(i), (centers[i, 0].item(), centers[i, 1].item()),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=12, fontweight="bold")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    ax.set_title("8 Gaussians  (radius = 5, std = 0.3, 8 classes)")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ticks=range(NUM_CLASSES), label="class")
    plt.tight_layout()
    plt.savefig(save, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save}")


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
    p.add_argument("--mode",        choices=["data", "crossings", "samples", "trajectory", "steps", "cfg", "all"],
                                    default="all")
    p.add_argument("--save-prefix", default="")
    p.add_argument("--steps",       type=int,   default=50)
    p.add_argument("--cfg-scale",   type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    pfx = args.save_prefix

    # Model-independent plots go in img/ — they're embedded in the README.
    # Skip when a save-prefix is set (those are for per-model PNG outputs).
    img_dir = Path("img")
    if not pfx:
        img_dir.mkdir(exist_ok=True)
    if args.mode == "data":
        fig_data(save=str(img_dir / "data_distribution.svg"))
        return
    if args.mode == "crossings":
        fig_crossings(save=str(img_dir / "trajectory_crossings.svg"))
        return

    device = get_device()
    model, ckpt = load_model(args.ckpt, device)

    if args.mode == "all" and not pfx:
        fig_data(save=str(img_dir / "data_distribution.svg"))
        fig_crossings(save=str(img_dir / "trajectory_crossings.svg"))
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

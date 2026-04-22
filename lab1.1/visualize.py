import argparse

import matplotlib.pyplot as plt
import torch

from mlp import MLP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="mlp.pt")
    p.add_argument("--hidden", type=int, nargs=2, default=[512, 256])
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--save", default="first_layer_weights.png")
    args = p.parse_args()

    model = MLP(hidden=tuple(args.hidden))
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    # first Linear layer is net[1] (net[0] is Flatten).
    W = model.net[1].weight.detach().cpu().numpy()  # (h1, 784)

    n = args.rows * args.cols
    fig, axes = plt.subplots(args.rows, args.cols, figsize=(args.cols, args.rows))
    for i, ax in enumerate(axes.flatten()):
        if i < n and i < W.shape[0]:
            w = W[i].reshape(28, 28)
            vmax = abs(w).max()
            ax.imshow(w, cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.axis("off")
    fig.suptitle(f"MLP first-layer weights (showing {min(n, W.shape[0])}/{W.shape[0]} units)")
    plt.tight_layout()
    plt.savefig(args.save, dpi=120, bbox_inches="tight")
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()

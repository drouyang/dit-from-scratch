"""Visualize attention weights from a trained reverse-task model.

For each input sequence, the model produces an (L, L) attention matrix where
row i is the weights output position i put on each input position. A
correctly-trained reverse model puts near-all its weight on position (L-1-i),
so the matrix is an anti-diagonal.

With multiple heads, we either plot them side-by-side (default) or average
them. Averaging obscures head specialization but is easier to read.

Run:
    python visualize.py --ckpt attention.pt --save attention_weights.png
    python visualize.py --ckpt attention.pt --average --save attention_avg.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from attention import AttentionModel


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = AttentionModel(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    default="attention.pt")
    p.add_argument("--save",    default="attention_weights.png")
    p.add_argument("--num-examples", type=int, default=3)
    p.add_argument("--average", action="store_true",
                   help="Average attention weights across heads into a single heatmap")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    model, cfg = load_model(args.ckpt, device)
    L = cfg["seq_len"]
    H = cfg["num_heads"]

    # Sample a few random inputs.
    x = torch.randint(0, cfg["vocab_size"], (args.num_examples, L), device=device)
    with torch.no_grad():
        logits, attn = model(x, return_attn=True)   # attn: (N, H, L, L)
    preds = logits.argmax(dim=-1)

    attn = attn.cpu().numpy()
    x_np = x.cpu().numpy()
    preds_np = preds.cpu().numpy()

    if args.average:
        attn = attn.mean(axis=1, keepdims=True)     # (N, 1, L, L)
        head_labels = ["mean over heads"]
        n_cols = 1
    else:
        head_labels = [f"head {h}" for h in range(H)]
        n_cols = H

    # Grid: one row per example, one column per head (or one).
    fig, axes = plt.subplots(
        args.num_examples, n_cols,
        figsize=(2.4 * n_cols, 2.8 * args.num_examples),
        squeeze=False,
    )

    for i in range(args.num_examples):
        for h in range(n_cols):
            ax = axes[i, h]
            im = ax.imshow(attn[i, h], cmap="viridis", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(head_labels[h], fontsize=10)
            if h == 0:
                # Label the row with the input -> prediction for context.
                input_str = " ".join(str(t) for t in x_np[i])
                pred_str  = " ".join(str(t) for t in preds_np[i])
                ax.set_ylabel(f"in:   {input_str}\npred: {pred_str}",
                              fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=40)
            ax.set_xticks(np.arange(L))
            ax.set_yticks(np.arange(L))
            ax.set_xticklabels(np.arange(L), fontsize=7)
            ax.set_yticklabels(np.arange(L), fontsize=7)
            ax.set_xlabel("key pos (input)", fontsize=8)
            if h == n_cols - 1 and i == 0:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Attention weights — each row shows where output position i looks.\n"
        "Trained on reverse: the anti-diagonal (i -> L-1-i) is the correct pattern.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.save, dpi=140, bbox_inches="tight")
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()

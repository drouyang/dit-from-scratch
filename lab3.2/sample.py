"""Generate an image from a text prompt.

Run:
    python sample.py --prompt "a cat sitting on a chair"
    python sample.py --prompt "a red sports car" --cfg-scale 4.0 --steps 50
    python sample.py --prompts "a dog" "a fish" "a tree" --cfg-scale 3.0
"""

import argparse

import matplotlib.pyplot as plt
import torch

from dit import DiT
from flow import fm_euler_sample
from text_encoder import CLIPTextEncoder
from vae import SDVae


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="model.pt")
    p.add_argument("--prompt",     default=None)
    p.add_argument("--prompts",    nargs="+", default=None,
                   help="multiple prompts; produces a grid")
    p.add_argument("--n-per-prompt", type=int, default=4)
    p.add_argument("--steps",      type=int,   default=50)
    p.add_argument("--cfg-scale",  type=float, default=4.0)
    p.add_argument("--save",       default="generated.png")
    p.add_argument("--seed",       type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    print("loading CLIP / VAE...")
    text_enc = CLIPTextEncoder().to(device)
    vae = SDVae().to(device)

    model = DiT(**cfg, p_uncond=0.1).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Build prompt list.
    if args.prompts is None:
        if args.prompt is None:
            raise ValueError("Provide --prompt or --prompts")
        args.prompts = [args.prompt]

    # Repeat each prompt n_per_prompt times.
    expanded = [p for p in args.prompts for _ in range(args.n_per_prompt)]
    n = len(expanded)

    text_tokens, text_pooled, text_mask = text_enc.encode(expanded, device)
    shape = (cfg["latent_channels"], cfg["latent_size"], cfg["latent_size"])

    z = fm_euler_sample(
        model, n, n_steps=args.steps, shape=shape,
        text_tokens=text_tokens, text_pooled=text_pooled, text_mask=text_mask,
        cfg_scale=args.cfg_scale, device=device,
    )
    images = vae.decode(z).clamp(-1, 1)
    images = (images + 1) / 2  # [-1, 1] -> [0, 1]

    # Layout: rows = unique prompts, cols = n_per_prompt
    n_prompts = len(args.prompts)
    n_cols = args.n_per_prompt
    fig, axes = plt.subplots(n_prompts, n_cols,
                             figsize=(n_cols * 1.5, n_prompts * 1.6),
                             squeeze=False)
    img_np = images.permute(0, 2, 3, 1).cpu().numpy()
    for i, prompt in enumerate(args.prompts):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.imshow(img_np[i * n_cols + j])
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(prompt[:30], rotation=0, ha="right", va="center",
                              fontsize=8)
    fig.suptitle(f"steps={args.steps}  cfg={args.cfg_scale}")
    plt.tight_layout()
    plt.savefig(args.save, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()

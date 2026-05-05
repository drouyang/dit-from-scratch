"""Train the latent text-to-image DiT on a tiny COCO subset.

End-to-end pipeline:
    image (B, 3, 64, 64) ─► SD-VAE.encode ─► z_0 (B, 4, 8, 8)
    caption (str)        ─► CLIP encoder  ─► (text_tokens, text_pooled, mask)
    sample t in [0, 1]
    z_t = (1 - t) * z_0 + t * noise               (flow matching forward)
    pred = DiT(z_t, t, text_tokens, text_pooled, mask)
    loss = MSE(pred, noise - z_0)                 (velocity target)

Defaults are sized for a few-hour M3 run on ~5K COCO image-caption pairs.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from data import TinyCOCO, collate
from dit import DiT
from flow import fm_q_sample
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
    p.add_argument("--steps",         type=int,   default=20_000)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--n-samples",     type=int,   default=5000,
                   help="how many COCO images to use")
    p.add_argument("--image-size",    type=int,   default=64)
    p.add_argument("--label-dropout", type=float, default=0.1,
                   help="probability of replacing text with the null embedding (CFG)")
    p.add_argument("--save-path",     default="model.pt")
    p.add_argument("--seed",          type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    # Frozen pretrained components.
    print("loading CLIP...")
    text_enc = CLIPTextEncoder().to(device)
    print("loading SD-VAE...")
    vae = SDVae().to(device)

    # Data.
    print(f"loading {args.n_samples} COCO image-caption pairs...")
    ds = TinyCOCO(n_samples=args.n_samples, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=2)

    # Trainable model.
    latent_size = args.image_size // 8  # SD-VAE is 8× spatial downsample
    model = DiT(
        latent_size=latent_size,
        latent_channels=4,
        patch_size=2,
        hidden=384,
        num_heads=6,
        num_blocks=8,
        text_dim=text_enc.hidden_size,
        p_uncond=args.label_dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"DiT params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    t0 = time.time()
    step = 0
    while step < args.steps:
        for images, captions in loader:
            if step >= args.steps:
                break

            images = images.to(device)
            with torch.no_grad():
                z_0 = vae.encode(images)
                text_tokens, text_pooled, text_mask = text_enc.encode(captions, device)

            t = torch.rand(images.size(0), device=device)
            z_t, _, target_v = fm_q_sample(z_0, t)
            pred = model(z_t, t, text_tokens, text_pooled, text_mask)
            loss = (pred - target_v).pow(2).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 200 == 0 or step == 1:
                elapsed = time.time() - t0
                print(f"step {step:6d}  |  {elapsed:6.1f}s  |  loss {loss.item():.4f}")
            step += 1

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "latent_size": latent_size,
            "latent_channels": 4,
            "patch_size": 2,
            "hidden": 384,
            "num_heads": 6,
            "num_blocks": 8,
            "text_dim": text_enc.hidden_size,
        },
    }, args.save_path)
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()

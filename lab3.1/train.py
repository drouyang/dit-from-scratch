"""Train DiT on MNIST with flow matching + CFG label-dropout.

The training loop is verbatim from lab 2.2 — sample t uniformly, compute
x_t with the closed-form forward process, predict velocity, MSE on the
target velocity, AdamW. The only differences:

    - data       : MNIST batches (B, 1, 28, 28) instead of (B, 2) toy points
    - model      : DiT (~1.9M params at the default config) instead of TimeMLP
    - loss       : MSE on a (B, 1, 28, 28) tensor instead of a (B, 2) one

The supervision target, the optimizer, the CFG label-dropout — all unchanged.

Run:
    python train.py                          # ~15 min on M3 MPS, default config
    python train.py --epochs 40 --hidden 192 # bigger / longer
    python train.py --label-dropout 0.0      # disable CFG (samples ignore class)
"""

import argparse
import time

import torch

from data import IMAGE_SIZE, IN_CHANNELS, NUM_CLASSES, get_loader
from dit import DiT
from flow import fm_q_sample


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch-size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=0.0)
    p.add_argument("--label-dropout", type=float, default=0.1,
                   help="probability of replacing the class label with the "
                        "null class during training (CFG). 0 disables CFG.")
    p.add_argument("--patch-size",    type=int,   default=4)
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--depth",         type=int,   default=6)
    p.add_argument("--num-heads",     type=int,   default=4)
    p.add_argument("--mlp-ratio",     type=float, default=4.0)
    p.add_argument("--save-path",     default="dit.pt")
    p.add_argument("--log-every",     type=int,   default=200)
    p.add_argument("--seed",          type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    config = {
        "image_size": IMAGE_SIZE,
        "in_channels": IN_CHANNELS,
        "patch_size": args.patch_size,
        "hidden": args.hidden,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "num_classes": NUM_CLASSES,
        "mlp_ratio": args.mlp_ratio,
    }
    model = DiT(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params  "
          f"(patch={args.patch_size}, hidden={args.hidden}, "
          f"depth={args.depth}, heads={args.num_heads})")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loader = get_loader(args.batch_size)

    t0 = time.time()
    step = 0
    running = 0.0
    running_n = 0
    for epoch in range(1, args.epochs + 1):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)

            # CFG label dropout — same as lab 2.2.
            if args.label_dropout > 0:
                mask = torch.rand(B, device=device) < args.label_dropout
                y = torch.where(mask, torch.full_like(y, model.null_class), y)

            t = torch.rand(B, device=device)
            x_t, _, target_v = fm_q_sample(x, t)
            pred = model(x_t, t, y)
            loss = (pred - target_v).pow(2).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += loss.item() * B
            running_n += B
            step += 1
            if step % args.log_every == 0:
                avg = running / running_n
                elapsed = time.time() - t0
                print(f"step {step:6d}  |  {elapsed:6.1f}s  |  "
                      f"epoch {epoch:3d}  |  loss {avg:.4f}")
                running = 0.0
                running_n = 0

    torch.save({"state_dict": model.state_dict(), "config": config}, args.save_path)
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()

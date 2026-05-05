"""Train a tiny flow-matching denoiser on 8 Gaussians, with optional CFG label-dropout.

Run:
    python train.py                          # FM with CFG (default)
    python train.py --label-dropout 0.0      # disable CFG dropout
"""

import argparse
import time

import torch

from data import sample_8gaussians, NUM_CLASSES
from flow import fm_q_sample
from mlp import TimeMLP


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",         type=int,   default=10_000)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--label-dropout", type=float, default=0.1,
                   help="probability of replacing class label with the null "
                        "class during training (CFG). 0 disables CFG.")
    p.add_argument("--save-path",     default="model.pt")
    p.add_argument("--seed",          type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    model = TimeMLP(num_classes=NUM_CLASSES).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x_0, y = sample_8gaussians(args.batch_size, device=device)

        # CFG: drop the class label with probability `label_dropout`.
        if args.label_dropout > 0:
            mask = torch.rand(args.batch_size, device=device) < args.label_dropout
            y = torch.where(mask, torch.full_like(y, model.null_class), y)

        t = torch.rand(args.batch_size, device=device)
        x_t, _, target_v = fm_q_sample(x_0, t)
        pred = model(x_t, t, y)
        loss = (pred - target_v).pow(2).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 1000 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:6d}  |  {elapsed:5.1f}s  |  loss {loss.item():.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "config": {"num_classes": NUM_CLASSES},
    }, args.save_path)
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()

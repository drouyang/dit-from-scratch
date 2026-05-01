"""Train a tiny denoiser on 8 Gaussians, with optional CFG label-dropout.

Default paradigm is **flow matching** (the production training objective for
SD3, FLUX, etc.). Pass `--paradigm ddpm` to train the same MLP with
ε-prediction on a Gaussian Markov chain instead, for direct comparison.

Run:
    python train.py                                    # FM, with CFG
    python train.py --paradigm ddpm                    # DDPM, with CFG
    python train.py --label-dropout 0.0                # disable CFG dropout
"""

import argparse
import time

import torch

from data import sample_8gaussians, NUM_CLASSES
from flow import fm_q_sample, ddpm_q_sample, DDPMSchedule
from mlp import TimeMLP


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paradigm",      choices=["fm", "ddpm"], default="fm")
    p.add_argument("--steps",         type=int,   default=10_000)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--label-dropout", type=float, default=0.1,
                   help="probability of replacing class label with the null "
                        "class during training (CFG). 0 disables CFG.")
    p.add_argument("--ddpm-T",        type=int,   default=100,
                   help="number of DDPM timesteps (only used if --paradigm ddpm)")
    p.add_argument("--save-path",     default=None,
                   help="default: vae_fm.pt or vae_ddpm.pt")
    p.add_argument("--seed",          type=int,   default=0)
    args = p.parse_args()

    if args.save_path is None:
        args.save_path = f"model_{args.paradigm}.pt"

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}, paradigm: {args.paradigm}")

    model = TimeMLP(num_classes=NUM_CLASSES).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    schedule = DDPMSchedule(T=args.ddpm_T).to(device) if args.paradigm == "ddpm" else None

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x_0, y = sample_8gaussians(args.batch_size, device=device)

        # CFG: drop the class label with probability `label_dropout`.
        if args.label_dropout > 0:
            mask = torch.rand(args.batch_size, device=device) < args.label_dropout
            y = torch.where(mask, torch.full_like(y, model.null_class), y)

        if args.paradigm == "fm":
            t = torch.rand(args.batch_size, device=device)
            x_t, _, target_v = fm_q_sample(x_0, t)
            pred = model(x_t, t, y)
            loss = (pred - target_v).pow(2).mean()
        else:  # ddpm
            t_int = torch.randint(0, schedule.T, (args.batch_size,), device=device)
            x_t, noise = ddpm_q_sample(x_0, t_int, schedule)
            t_norm = t_int.float() / schedule.T
            pred = model(x_t, t_norm, y)
            loss = (pred - noise).pow(2).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 1000 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:6d}  |  {elapsed:5.1f}s  |  loss {loss.item():.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "config": {"num_classes": NUM_CLASSES},
        "paradigm": args.paradigm,
        "ddpm_T": args.ddpm_T if args.paradigm == "ddpm" else None,
    }, args.save_path)
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()

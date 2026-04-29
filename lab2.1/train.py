"""Train a VAE on MNIST.

Run: `python train.py`  (~2 minutes on M3 / MPS at default settings)

Reports per-epoch (train and test):
    ELBO   = recon + beta * KL  (the actual loss being minimized, divided by N)
    recon  = per-image reconstruction loss (BCE summed over 28*28 pixels)
    KL     = per-image KL(N(mu, sigma^2) || N(0, I))

A trained VAE has both terms non-trivial: recon high enough to reconstruct
digits, KL high enough that latents are spread out (not all at the origin).
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vae import VAE, vae_loss


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_epoch(model, loader, device, optimizer=None, beta=1.0):
    """Single pass over `loader`. If `optimizer` is None, runs eval mode."""
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = recon_sum = kl_sum = 0.0
    n = 0
    with torch.set_grad_enabled(is_train):
        for x, _ in loader:
            x = x.to(device)
            x_recon_logits, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x, x_recon_logits, mu, logvar, beta=beta)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item()
            recon_sum += recon.item()
            kl_sum += kl.item()
            n += x.size(0)
    return loss_sum / n, recon_sum / n, kl_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dim", type=int,   default=16)
    p.add_argument("--beta",       type=float, default=1.0)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--save-path",  default="vae.pt")
    p.add_argument("--seed",       type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}  |  latent_dim: {args.latent_dim}  |  beta: {args.beta}")

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    model = VAE(latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_recon, tr_kl = run_epoch(model, train_loader, device,
                                             optimizer=optimizer, beta=args.beta)
        te_loss, te_recon, te_kl = run_epoch(model, test_loader,  device,
                                             optimizer=None,      beta=args.beta)
        dt = time.time() - t0
        print(f"epoch {epoch:2d} | {dt:6.1f}s | "
              f"train ELBO {tr_loss:7.2f}  recon {tr_recon:7.2f}  KL {tr_kl:6.2f} | "
              f"test ELBO {te_loss:7.2f}  recon {te_recon:7.2f}  KL {te_kl:6.2f}")

    torch.save({
        "state_dict": model.state_dict(),
        "config":     {"latent_dim": args.latent_dim, "beta": args.beta},
    }, args.save_path)
    print(f"saved weights + config to {args.save_path}")


if __name__ == "__main__":
    main()

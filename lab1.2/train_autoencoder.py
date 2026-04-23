"""Train the CIFAR-10 CNN autoencoder.

High-level flow:
  1. Parse CLI flags.
  2. Pick a compute device (NVIDIA GPU > Apple MPS > CPU).
  3. Load CIFAR-10 as two DataLoaders (train + test).
  4. Build the autoencoder, optimizer, and loss function.
  5. For each epoch (= one full pass through the training set):
       - switch to train() mode, iterate over the train loader,
         do a gradient update per mini-batch.
       - switch to eval() mode, compute test reconstruction loss.
  6. Save the final weight tensors to disk.

Run: `python train.py [flags...]`.  See README for flag reference.
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import Autoencoder


def get_device():
    """Pick the fastest available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute average per-pixel MSE reconstruction loss over `loader`.

    We report per-pixel MSE (loss / total_pixel_count) rather than per-image MSE
    so the number is independent of image resolution — easier to compare across runs.
    """
    model.eval()
    # reduction="sum" accumulates losses across the batch without averaging.
    # We divide by total pixel count at the end for a true per-pixel average.
    criterion = nn.MSELoss(reduction="sum")
    loss_sum = 0.0
    total_pixels = 0
    for x, _ in loader:
        # _ = class labels. Autoencoders are unsupervised — we don't use them.
        x = x.to(device)
        x_hat, _ = model(x)
        loss_sum += criterion(x_hat, x).item()
        # x.numel() = B * C * H * W = B * 3 * 32 * 32 = B * 3072
        total_pixels += x.numel()
    return loss_sum / total_pixels


def main():
    # ---------- Parse CLI flags ----------
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dim", type=int, default=256,
                   help="bottleneck size (try 32, 256, 1024 — see README)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--save-path", default="cnn_autoencoder.pt")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # ---------- Reproducibility + device ----------
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}  |  latent_dim: {args.latent_dim}  |  lr: {args.lr}")

    # ---------- Data ----------
    # CIFAR-10: 60,000 32×32 RGB images across 10 classes (airplane, automobile,
    # bird, cat, deer, dog, frog, horse, ship, truck). 50k train / 10k test.
    #
    # transforms.ToTensor() converts a PIL image in [0, 255] to a float tensor
    # (3, 32, 32) in [0.0, 1.0]. We do NOT apply mean/std normalization here,
    # because the decoder's Sigmoid output is in [0, 1] and the MSE loss works
    # best when targets and predictions live in the same range.
    transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # ---------- Model, optimizer, loss ----------
    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: CNN Autoencoder  |  latent_dim={args.latent_dim}  |  params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # MSELoss = mean squared error = (1/N) Σ (x̂ᵢ - xᵢ)².
    # It penalizes every pixel proportionally to the square of the reconstruction error.
    # This tends to produce slightly blurry reconstructions (MSE rewards the "average"
    # appearance over sharp edges) — a known limitation. VAEs often use MSE + KL;
    # more advanced models use perceptual losses. For now, MSE is the right starting point.
    criterion = nn.MSELoss()

    # ---------- Training loop ----------
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0

        for x, _ in train_loader:
            x = x.to(device)

            # THE FIVE-LINE TRAINING STEP. Identical to lab1.1 — same loop, different model.
            optimizer.zero_grad()          # 1. clear accumulated gradients
            x_hat, _ = model(x)            # 2. forward pass: encode → decode
            loss = criterion(x_hat, x)     # 3. MSE between reconstruction and original
            loss.backward()                # 4. backprop: fill .grad on every parameter
            optimizer.step()               # 5. update: param ← param - lr * (grad-based step)

            # loss is a per-pixel mean (CrossEntropyLoss default was also mean).
            # Multiply by batch size to get the unnormalized sum for this batch.
            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running / seen
        test_loss = evaluate(model, test_loader, device)
        dt = time.time() - t0
        best_loss = min(best_loss, test_loss)
        print(f"epoch {epoch:2d}  |  {dt:5.1f}s  |  train_loss {train_loss:.5f}  "
              f"|  test_loss {test_loss:.5f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"best test_loss: {best_loss:.5f}  |  saved weights to {args.save_path}")


if __name__ == "__main__":
    main()

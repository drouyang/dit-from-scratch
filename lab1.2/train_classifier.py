"""Train the CIFAR-10 CNN classifier.

Same encoder backbone as the autoencoder, but supervised with CrossEntropyLoss
instead of unsupervised MSE. Reports train/test accuracy each epoch.

High-level flow:
  1. Parse CLI flags.
  2. Pick compute device (CUDA → MPS → CPU).
  3. Load CIFAR-10 (normalized for classification).
  4. Build Classifier, optimizer, and CrossEntropyLoss.
  5. Train: one gradient update per mini-batch.
  6. Evaluate: top-1 accuracy on the test set each epoch.
  7. Save the best checkpoint.

Run: `python train_classifier.py [flags...]`
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import Classifier


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (avg cross-entropy loss, top-1 accuracy) over `loader`."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--save-path", default="cnn_classifier.pt")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}  |  latent_dim: {args.latent_dim}  |  lr: {args.lr}")

    # ---------- Data ----------
    # For classification we normalize per-channel to zero mean / unit variance.
    # These are the standard CIFAR-10 mean and std across the training set.
    # Normalization is not needed for MSE reconstruction (the autoencoder uses raw
    # [0,1] pixels), but it stabilizes gradient flow for a classifier because each
    # input dimension now has comparable scale — the linear head's weights don't need
    # to compensate for large differences in channel magnitude.
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        # Random horizontal flip is cheap data augmentation: CIFAR-10 classes are
        # horizontally symmetric (a flipped airplane is still an airplane).
        # This doubles the effective training set size and cuts overfitting.
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(args.data_dir, train=True,  download=True, transform=train_transform)
    test_set  = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # ---------- Model, optimizer, loss ----------
    model = Classifier(latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: CNN Classifier  |  latent_dim={args.latent_dim}  |  params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # CrossEntropyLoss = log-softmax + negative log-likelihood.
    # It takes raw logits (shape B×10) and integer class indices (shape B).
    # The loss for a single example is: -log(softmax(logit[true_class])).
    # A random 10-class classifier scores log(1/10) ≈ 2.3 — watch the loss
    # drop from ~2.3 at epoch 1 to ~0.8–1.0 at convergence.
    criterion = nn.CrossEntropyLoss()

    # ---------- Training loop ----------
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        correct = 0
        seen = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)               # (B, 10) raw class scores
            loss = criterion(logits, y)     # CrossEntropy vs. ground-truth labels
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += x.size(0)

        train_loss = running_loss / seen
        train_acc  = correct / seen
        test_loss, test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0

        print(f"epoch {epoch:2d}  |  {dt:5.1f}s  "
              f"|  train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
              f"|  test_loss {test_loss:.4f}  test_acc {test_acc:.3f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.save_path)

    print(f"best test_acc: {best_acc:.3f}  |  saved weights to {args.save_path}")


if __name__ == "__main__":
    main()

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mlp import MLP


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_optimizer(name, params, lr):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    raise ValueError(f"unknown optimizer: {name}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", default="adam", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, nargs=2, default=[512, 256])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--save-path", default="mlp.pt")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}  |  optimizer: {args.optimizer}  |  lr: {args.lr}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    model = MLP(hidden=tuple(args.hidden), dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: MLP hidden={tuple(args.hidden)} dropout={args.dropout}  |  params: {n_params:,}")

    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        train_loss = running / seen
        test_loss, test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0
        best_acc = max(best_acc, test_acc)
        print(f"epoch {epoch:2d}  |  {dt:5.1f}s  |  train_loss {train_loss:.4f}  "
              f"|  test_loss {test_loss:.4f}  |  test_acc {test_acc * 100:.2f}%")

    torch.save(model.state_dict(), args.save_path)
    print(f"best test_acc: {best_acc * 100:.2f}%  |  saved weights to {args.save_path}")


if __name__ == "__main__":
    main()

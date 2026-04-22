"""Train the MNIST MLP.

High-level flow:
  1. Parse CLI flags.
  2. Pick a compute device (NVIDIA GPU > Apple MPS > CPU).
  3. Load MNIST as two DataLoaders (train + test), with pixel normalization.
  4. Build the model, optimizer, and loss function.
  5. For each epoch (= one full pass through the training set):
       - switch to train() mode, iterate over the train loader,
         do a gradient update per mini-batch.
       - switch to eval() mode, compute test loss + accuracy.
  6. Save the final weight tensors to disk (`mlp.pt`).

Run: `python train.py [flags...]`.  See README for flag reference.
"""

import argparse  # CLI flag parsing (--lr, --epochs, etc.)
import time      # wall-clock timing per epoch

import torch                                        # core PyTorch: tensors + autograd engine
import torch.nn as nn                               # layers and loss functions live under nn
from torch.utils.data import DataLoader             # batching, shuffling, parallel data loading
from torchvision import datasets, transforms        # MNIST dataset + image preprocessing ops

from mlp import MLP                                 # our model, defined in mlp.py


def get_device():
    """Pick the fastest available compute device.

    torch tensors and model weights live ON a specific device; operations only
    work between tensors on the same device. `.to(device)` moves them.
    """
    # CUDA = NVIDIA GPU path. Fastest for training by a wide margin.
    if torch.cuda.is_available():
        return "cuda"
    # MPS = "Metal Performance Shaders", Apple Silicon GPU path.
    # Noticeably slower than CUDA but much faster than CPU for a model this small.
    if torch.backends.mps.is_available():
        return "mps"
    # CPU works for MNIST (model is tiny), but is too slow for anything larger.
    return "cpu"


def build_optimizer(name, params, lr):
    """Construct an optimizer. All three update weights using gradients, but differ in HOW.

    See README section 3 for the conceptual breakdown.

    `params` is what the optimizer will update — we pass `model.parameters()`.
    The optimizer stores references to these tensors; when you call .step()
    it mutates them in place based on their .grad fields.
    """
    name = name.lower()
    if name == "sgd":
        # SGD = Stochastic Gradient Descent. momentum=0.9 adds a running average
        # of past gradients, which smooths noisy updates. Typical default in old papers.
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        # Adam adapts the step size per parameter using running estimates of the
        # gradient's mean and squared-magnitude. Robust to gradient scale → easy to tune.
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        # AdamW = Adam + "decoupled" weight decay (shrinks every weight toward 0 each
        # step, separate from the gradient step). 1e-2 is the community default for transformers.
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    raise ValueError(f"unknown optimizer: {name}")


# @torch.no_grad() disables autograd bookkeeping for everything inside this function.
# Gradients are only needed for training — not for evaluation. Turning autograd off
# here makes eval ~2x faster and avoids building graphs we'd immediately throw away.
@torch.no_grad()
def evaluate(model, loader, device):
    """Compute average loss and top-1 accuracy over `loader`."""
    # eval() does two things: (1) tells Dropout to become a pass-through, and (2)
    # tells BatchNorm (not used here) to freeze its running statistics. Forgetting
    # this call is one of the most common PyTorch bugs — test metrics get noisy.
    model.eval()
    # reduction="sum" returns the total loss across the batch (not averaged).
    # We accumulate into loss_sum and divide by `total` at the end — this way
    # the final number is a true average over the whole test set, regardless
    # of how the batches line up at the boundary.
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        # x: (B, 1, 28, 28) images, y: (B,) integer labels 0..9.
        # Move to device so they match the model's weights.
        x, y = x.to(device), y.to(device)
        logits = model(x)                         # (B, 10) raw class scores
        loss_sum += criterion(logits, y).item()   # .item() pulls a Python float out of a 0-dim tensor
        # argmax(1) picks the highest-scoring class per row → (B,) predicted labels.
        # Compare element-wise to y, cast True→1 via .sum(), convert to Python int.
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)                        # size of this batch (last batch may be smaller)
    # average loss per example,  accuracy in [0, 1]
    return loss_sum / total, correct / total


def main():
    # ---------- Parse CLI flags ----------
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", default="adam", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=1e-3)                 # learning rate
    p.add_argument("--epochs", type=int, default=10)                 # passes over train set
    p.add_argument("--batch-size", type=int, default=128)            # examples per gradient step
    p.add_argument("--hidden", type=int, nargs=2, default=[512, 256])# two hidden-layer widths
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--data-dir", default="./data")                   # where MNIST gets cached
    p.add_argument("--save-path", default="mlp.pt")                  # final weights here
    p.add_argument("--num-workers", type=int, default=2)             # DataLoader worker processes
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # ---------- Reproducibility + device ----------
    # manual_seed seeds PyTorch's RNGs: weight init, dropout masks, data shuffling.
    # Same seed + same code + same hardware → same results.
    # (Different GPUs can still diverge slightly on non-deterministic CUDA kernels —
    # don't chase bit-exact reproducibility across machines.)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}  |  optimizer: {args.optimizer}  |  lr: {args.lr}")

    # ---------- Data ----------
    # transforms.Compose applies each transform in order to every image:
    #   ToTensor:  PIL image in [0,255] → torch tensor (1, 28, 28) in [0.0, 1.0]
    #   Normalize: (x - mean) / std, elementwise. The constants (0.1307, 0.3081)
    #              are the population mean and std of MNIST training pixels.
    # Why normalize? Centering activations near zero keeps the early-layer gradients
    # well-scaled and convergence faster and more stable.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    # Downloads MNIST on first run (~55MB) and caches under args.data_dir.
    # train=True → 60k training examples; train=False → 10k test examples.
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    # DataLoader wraps a Dataset with:
    #   - batching (groups N examples into a single tensor)
    #   - shuffling (reshuffles each epoch; crucial so SGD sees uncorrelated batches)
    #   - parallel loading (num_workers subprocesses prepare batches in background)
    #   - pin_memory (page-locks CPU memory → faster CPU→GPU transfer under CUDA)
    # Test loader: no shuffle (order doesn't matter), larger batch (no gradients → bigger fits in memory).
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # ---------- Model, optimizer, loss ----------
    # .to(device) moves every parameter tensor to GPU/MPS. If you forget this,
    # forward() crashes with "expected all tensors to be on the same device".
    model = MLP(hidden=tuple(args.hidden), dropout=args.dropout).to(device)
    # Each parameter tensor has a .numel() count (total scalars in it). Summing gives the
    # classic "model size" you see reported in papers.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: MLP hidden={tuple(args.hidden)} dropout={args.dropout}  |  params: {n_params:,}")

    # model.parameters() yields every trainable weight tensor; the optimizer holds
    # references and mutates them in place during .step().
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)
    # CrossEntropyLoss = log_softmax + negative log-likelihood, fused and numerically stable.
    # Inputs: (B, num_classes) logits  AND  (B,) integer targets. Not one-hot.
    criterion = nn.CrossEntropyLoss()

    # ---------- Training loop ----------
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # train() turns Dropout ON and puts BatchNorm into batch-statistics mode.
        # Must be called here — evaluate() calls eval() and leaves the model in eval mode.
        model.train()
        t0 = time.time()
        running = 0.0  # accumulator for train loss (un-normalized sum)
        seen = 0       # total examples seen this epoch
        for x, y in train_loader:
            # x: (B, 1, 28, 28)  images
            # y: (B,)             integer labels in [0, 9]
            x, y = x.to(device), y.to(device)

            # THE FIVE-LINE TRAINING STEP.  Every PyTorch model trains with this pattern.
            optimizer.zero_grad()        # 1. clear .grad on every parameter.
                                         #    Grads ACCUMULATE by default — if you skip this,
                                         #    you'd add the new batch's grads on top of the old ones.
            logits = model(x)            # 2. forward pass. Runs MLP.forward() AND builds the autograd
                                         #    graph tracking every op, so backward() knows what to do.
            loss = criterion(logits, y)  # 3. scalar loss (one number, not per-example).
            loss.backward()              # 4. backward pass. Autograd walks the graph in reverse and
                                         #    fills .grad on every parameter that contributed to `loss`.
            optimizer.step()             # 5. apply the update: param ← param - lr * (something based on param.grad).

            # loss.item() syncs with the GPU and returns a Python float. We multiply by batch size
            # because CrossEntropyLoss's default reduction is "mean" — so .item() is already a
            # per-example average for this batch; multiplying undoes that to get a sum we can re-average later.
            running += loss.item() * y.size(0)
            seen += y.size(0)

        train_loss = running / seen
        # Evaluate on the test set. Side effect: leaves the model in eval() mode.
        test_loss, test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0
        best_acc = max(best_acc, test_acc)
        print(f"epoch {epoch:2d}  |  {dt:5.1f}s  |  train_loss {train_loss:.4f}  "
              f"|  test_loss {test_loss:.4f}  |  test_acc {test_acc * 100:.2f}%")

    # state_dict() returns an OrderedDict mapping parameter names (e.g. "net.1.weight")
    # to their tensor values. This is the preferred way to save — small, portable, and
    # doesn't pickle class references (so it survives code refactors). Load later with:
    #   m = MLP(...); m.load_state_dict(torch.load("mlp.pt"))
    torch.save(model.state_dict(), args.save_path)
    print(f"best test_acc: {best_acc * 100:.2f}%  |  saved weights to {args.save_path}")


if __name__ == "__main__":
    # This guard prevents main() from running if someone imports train.py as a module.
    # Also required on macOS/Windows when DataLoader uses num_workers > 0 — the workers
    # re-import this file, and without the guard they'd each start training recursively.
    main()

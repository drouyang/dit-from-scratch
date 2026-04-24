"""Train an attention-only model on the reverse task.

The reverse task: input is a random sequence of tokens `[a, b, c, d]`,
target is the reversed sequence `[d, c, b, a]`. The model has exactly one
attention layer and a linear head — no MLP, no LayerNorm, no residual. If it
succeeds, attention alone is doing the work.

Why this task: it has a visually clean optimal attention pattern. Output
position i must attend to input position (L-1-i), so a fully-trained model's
attention weights form a sharp anti-diagonal. Run visualize.py after training
to see it.

Run: `python train.py`  (≈ 1 min on MPS / CPU)
"""

import argparse
import time

import torch
import torch.nn as nn

from attention import AttentionModel


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_batch(batch_size, seq_len, vocab_size, device):
    """Random sequences and their reversals, generated on the fly.

    The dataset is infinite — there's no train/test split because the
    distribution is uniform over all (vocab_size ** seq_len) sequences and we
    only see a tiny fraction during training. Generalization is the default,
    not something we have to engineer.
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = x.flip(dims=[1])
    return x, y


@torch.no_grad()
def evaluate(model, batch_size, seq_len, vocab_size, device, n_batches=20):
    """Return (avg cross-entropy loss, per-token accuracy) over fresh samples."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    for _ in range(n_batches):
        x, y = sample_batch(batch_size, seq_len, vocab_size, device)
        logits = model(x)                              # (B, L, vocab)
        loss_sum += criterion(logits.reshape(-1, vocab_size), y.reshape(-1)).item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    return loss_sum / n_batches, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len",    type=int,   default=10)
    p.add_argument("--vocab-size", type=int,   default=20)
    p.add_argument("--embed-dim",  type=int,   default=64)
    p.add_argument("--num-heads",  type=int,   default=4)
    p.add_argument("--lr",         type=float, default=3e-3)
    p.add_argument("--steps",      type=int,   default=2000)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--log-every",  type=int,   default=100)
    p.add_argument("--save-path",  default="attention.pt")
    p.add_argument("--seed",       type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}  |  seq_len: {args.seq_len}  |  vocab: {args.vocab_size}  "
          f"|  embed_dim: {args.embed_dim}  |  heads: {args.num_heads}")

    model = AttentionModel(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: AttentionModel  |  params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Random baseline: log(vocab_size). Watch the loss drop from ~log(20)≈3.0
    # toward 0 as attention learns to route output position i to input (L-1-i).
    criterion = nn.CrossEntropyLoss()

    model.train()
    t0 = time.time()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for step in range(1, args.steps + 1):
        x, y = sample_batch(args.batch_size, args.seq_len, args.vocab_size, device)

        optimizer.zero_grad()
        logits = model(x)                                             # (B, L, vocab)
        loss = criterion(logits.reshape(-1, args.vocab_size),         # CE wants (N, C)
                         y.reshape(-1))                                # and (N,)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += (logits.argmax(dim=-1) == y).sum().item()
        running_total += y.numel()

        if step % args.log_every == 0:
            train_loss = running_loss / args.log_every
            train_acc  = running_correct / running_total
            val_loss, val_acc = evaluate(model, args.batch_size, args.seq_len,
                                         args.vocab_size, device)
            dt = time.time() - t0
            print(f"step {step:5d}  |  {dt:5.1f}s  "
                  f"|  train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  "
                  f"|  val_loss {val_loss:.4f}  val_acc {val_acc:.3f}")
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            model.train()

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": args.vocab_size,
            "seq_len":    args.seq_len,
            "embed_dim":  args.embed_dim,
            "num_heads":  args.num_heads,
        },
    }, args.save_path)
    print(f"saved weights + config to {args.save_path}")


if __name__ == "__main__":
    main()

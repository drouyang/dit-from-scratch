"""Train nanoGPT on TinyShakespeare (char-level).

The task: given the previous ~256 characters of Shakespeare plays, predict
the next character. Trained to convergence, the model generates
Shakespeare-flavored text — readable English in iambic-ish rhythm with
proper turn-taking dialogue, even though the model has no semantic
understanding of what it's saying.

What "learning" looks like over training:
    val_loss ≈ 4.2 — random over 65 chars
    val_loss ≈ 3.0 — unigram (letter frequencies)
    val_loss ≈ 2.0 — bigrams + spacing right
    val_loss ≈ 1.5 — words + capitalized speaker names
    val_loss ≈ 1.3 — short phrases, archaic English ("thou", "wherefore")

Run: `python train.py`  (≈ 5–10 min on MPS for the default config)
"""

import argparse
import math
import os
import time
import urllib.request

import torch

from gpt import GPT


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_PATH = "data/input.txt"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_data():
    if not os.path.exists(DATA_PATH):
        print(f"downloading TinyShakespeare to {DATA_PATH} ...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def tokenize(text):
    """Char-level tokenizer: every unique character gets an integer id.

    For TinyShakespeare this gives a 65-token vocab — every printable
    character that appears in the corpus, including newline and a few
    pieces of punctuation. No BPE, no subword units, no external file.
    """
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    return encode, decode, stoi, itos, len(chars)


def get_batch(data, batch_size, block_size, device):
    # Sample `batch_size` random offsets and grab block_size+1 tokens at each.
    # x is tokens[i..i+L-1]; y is tokens[i+1..i+L] — the next-char target at
    # every position (the causal mask makes this safe — see gpt.py).
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device,
                  n_iters=20):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(n_iters)
        for k in range(n_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Linear warmup followed by cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--block-size",   type=int,   default=256)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--n-layer",      type=int,   default=6)
    p.add_argument("--n-head",       type=int,   default=6)
    p.add_argument("--n-embd",       type=int,   default=384)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--max-steps",    type=int,   default=5000)
    p.add_argument("--max-lr",       type=float, default=3e-4)
    p.add_argument("--min-lr",       type=float, default=3e-5)
    p.add_argument("--warmup",       type=int,   default=100)
    p.add_argument("--eval-every",   type=int,   default=500)
    p.add_argument("--sample-every", type=int,   default=1000)
    p.add_argument("--save-path",    default="gpt.pt")
    p.add_argument("--seed",         type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    text = download_data()
    encode, decode, stoi, itos, vocab_size = tokenize(text)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"device: {device}  |  chars: {len(text):,}  |  vocab: {vocab_size}")
    print(f"train tokens: {len(train_data):,}  |  val tokens: {len(val_data):,}")

    model = GPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: GPT  |  params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr,
        betas=(0.9, 0.95), weight_decay=0.1,
    )

    t0 = time.time()
    model.train()
    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.warmup, args.max_steps, args.max_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1:
            losses = estimate_loss(model, train_data, val_data,
                                   args.batch_size, args.block_size, device)
            dt = time.time() - t0
            print(f"step {step:5d}  |  {dt:6.1f}s  |  lr {lr:.2e}  "
                  f"|  train {losses['train']:.4f}  val {losses['val']:.4f}")

        if step % args.sample_every == 0:
            ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
            out = model.generate(ctx, max_new_tokens=200,
                                 temperature=1.0, top_k=40)
            print("--- sample @ step", step, "---")
            print(decode(out[0].cpu().tolist()))
            print("---")

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "block_size": args.block_size,
            "n_layer":    args.n_layer,
            "n_head":     args.n_head,
            "n_embd":     args.n_embd,
            "dropout":    args.dropout,
        },
        "stoi": stoi,
    }, args.save_path)
    print(f"saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()

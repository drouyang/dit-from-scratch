"""Generate text from a trained nanoGPT checkpoint.

Run:
    python sample.py --prompt "ROMEO:" --max-tokens 500
    python sample.py --temperature 0.6 --top-k 40
    python sample.py --temperature 1.5    # spicier, more typos
"""

import argparse

import torch

from gpt import GPT


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        default="gpt.pt")
    p.add_argument("--prompt",      default="\n")
    p.add_argument("--max-tokens",  type=int,   default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k",       type=int,   default=40)
    p.add_argument("--seed",        type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    stoi = ckpt["stoi"]
    itos = {i: c for c, i in stoi.items()}

    model = GPT(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Unknown chars in the prompt fall back to id 0 (newline in this corpus).
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )
    print(decode(out[0].cpu().tolist()))


if __name__ == "__main__":
    main()

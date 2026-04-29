"""Generate text from a pretrained GPT-2 loaded into our `gpt.py` class.

Run:
    python sample.py --model gpt2 --prompt "Once upon a time"
    python sample.py --model gpt2-medium --temperature 0.6 --top-k 40
    python sample.py --model gpt2-xl --prompt "The capital of France is"
"""

import argparse

import tiktoken
import torch

from gpt import GPT, GPT2_CONFIGS


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(GPT2_CONFIGS), default="gpt2")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)

    enc = tiktoken.get_encoding("gpt2")
    model = GPT.from_pretrained(args.model).to(device)
    model.eval()

    idx = torch.tensor([enc.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()

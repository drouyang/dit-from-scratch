"""Side-by-side qualitative comparison of GPT-2 sizes on the same prompts.

Loads each model one at a time (so you don't need to fit all four in memory),
generates a completion for each prompt with the *same* seed, and prints the
results grouped by prompt for easy reading.

Run:
    python compare.py
    python compare.py --models gpt2 gpt2-xl --max-tokens 100
    python compare.py --prompts "The capital of France is" "Once upon a time"
"""

import argparse
import gc

import tiktoken
import torch

from gpt import GPT, GPT2_CONFIGS


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "In quantum mechanics, the wave function",
    "Q: What year did World War II end? A:",
    "def fibonacci(n):\n    ",
]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_one(model, enc, prompt, device, max_tokens, temperature, top_k, seed):
    torch.manual_seed(seed)
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
    )
    return enc.decode(out[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(GPT2_CONFIGS),
                   choices=list(GPT2_CONFIGS))
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = get_device()
    enc = tiktoken.get_encoding("gpt2")

    # results[prompt][model] = completion
    results = {prompt: {} for prompt in args.prompts}

    for model_name in args.models:
        model = GPT.from_pretrained(model_name).to(device)
        model.eval()
        for prompt in args.prompts:
            results[prompt][model_name] = generate_one(
                model, enc, prompt, device,
                args.max_tokens, args.temperature, args.top_k, args.seed,
            )
        # Free this model before loading the next (XL is ~6 GB).
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        if device == "mps":
            torch.mps.empty_cache()

    for prompt in args.prompts:
        print("=" * 78)
        print(f"PROMPT: {prompt!r}")
        print("=" * 78)
        for model_name in args.models:
            print(f"\n--- {model_name} ---")
            print(results[prompt][model_name])
        print()


if __name__ == "__main__":
    main()

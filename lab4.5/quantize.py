"""Produce a quantized variant of WAN-2.1 T2V-1.3B for distribution.

Quantization compresses the transformer weights from bf16 (2 bytes/param)
to int8 (1 byte) or NF4 (~0.5 byte). For Wan-2.1-1.3B that's:

    bf16 baseline:  ~2.6 GB
    int8:            ~1.3 GB    (fits 24 GB GPU comfortably)
    NF4 (4-bit):     ~0.7 GB    (fits on a 12 GB consumer GPU)

The trade-off is a small quality drop and a measurable inference slowdown
(int8/NF4 matmuls aren't always faster than bf16 on modern GPUs — they
*can* be on memory-bound layers, but TFLOPS-bound ones often regress).
What you're buying is *fit*: small-VRAM users can run your model at all.

This script uses bitsandbytes via diffusers' BitsAndBytesConfig — the
canonical path for quantizing diffusion models in HF land. Other options
(torchao, optimum-quanto, GGUF for llama.cpp) all exist; bitsandbytes is
what the diffusers / `pipe.load_lora_weights(...)` ecosystem speaks
natively, so it composes cleanly with your LoRA.

Run:
    # Save an NF4 (4-bit) variant of the transformer + push to HF:
    python quantize.py --quant nf4 \\
                       --output Wan2.1-T2V-1.3B-NF4 \\
                       --push-to-hub your-username/wan-2.1-1.3b-nf4

Requires CUDA — bitsandbytes does not support MPS or CPU. Run on a
rented GPU or skip this path.
"""

import argparse
from pathlib import Path

import torch
from diffusers import BitsAndBytesConfig, WanTransformer3DModel

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quant",        choices=["int8", "nf4"], default="nf4")
    p.add_argument("--output",       default="Wan2.1-T2V-1.3B-quantized",
                   help="local directory to save the quantized transformer")
    p.add_argument("--push-to-hub",  default=None,
                   help="optional HF repo id to upload to, e.g. user/wan-2.1-1.3b-nf4")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("quantize.py requires CUDA — bitsandbytes has no MPS/CPU backend.")

    # Build the quant config. NF4 is the recommended 4-bit format
    # (better than naive int4 on perplexity / FID); int8 is the safest
    # near-lossless choice if you don't need the full 4× compression.
    if args.quant == "nf4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"loading Wan-2.1 T2V-1.3B and quantizing to {args.quant}...")
    transformer = WanTransformer3DModel.from_pretrained(
        WAN_REPO, subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(str(out_dir))
    print(f"saved to {out_dir}/")
    print("note: only the transformer is quantized — VAE and text encoder")
    print("stay in bf16 (they're frozen at inference, no LoRA on top, and")
    print("their compute share is small relative to the transformer).")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        print(f"uploading to https://huggingface.co/{args.push_to_hub}...")
        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.push_to_hub, repo_type="model")
        print("done.")


if __name__ == "__main__":
    main()

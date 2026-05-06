"""Sample from a full-SFT'd Wan-2.1 T2V-1.3B checkpoint.

Drops the fine-tuned transformer weights into a stock WanPipeline. Compare
side-by-side with the base model to actually see what your SFT learned:

    # SFT'd:
    python sample_sft.py --prompt "..." --ckpt runs/my-sft/transformer_step02000.safetensors --out sft.mp4

    # Base WAN, same prompt and seed, for comparison:
    python sample_sft.py --prompt "..." --out base.mp4
"""

import argparse

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from safetensors.torch import load_file


WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",          required=True)
    p.add_argument("--ckpt",            default=None,
                   help="Path to a SFT transformer .safetensors. Omit for base WAN.")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--num-frames",      type=int,   default=49)
    p.add_argument("--steps",           type=int,   default=30)
    p.add_argument("--guidance",        type=float, default=5.0)
    p.add_argument("--height",          type=int,   default=480)
    p.add_argument("--width",           type=int,   default=832)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--out",             default="out.mp4")
    args = p.parse_args()

    print(f"loading {WAN_REPO}...")
    vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_REPO, vae=vae, torch_dtype=torch.bfloat16).to("cuda")

    if args.ckpt:
        print(f"loading SFT checkpoint {args.ckpt}...")
        state = load_file(args.ckpt)
        # `strict=True` would catch a model-architecture mismatch — useful safety
        # net since SFT checkpoints are full state dicts (no missing-key forgiveness
        # like LoRA gives you).
        missing, unexpected = pipe.transformer.load_state_dict(state, strict=True)
        if missing or unexpected:
            print(f"WARNING: {len(missing)} missing, {len(unexpected)} unexpected keys")

    generator = torch.Generator("cuda").manual_seed(args.seed)
    print(f"generating {args.num_frames} frames at {args.width}x{args.height}...")
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height, width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).frames[0]

    export_to_video(output, args.out, fps=16)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

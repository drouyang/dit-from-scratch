"""Generate a video from a text prompt with WAN 2.2 TI2V-5B (diffusers path).

Run:
    python inference_diffusers.py --prompt "a fluffy red panda eating bamboo on a tree branch"

Requires diffusers from source (WanPipeline isn't in stable yet):
    pip install git+https://github.com/huggingface/diffusers

Memory: ~24GB VRAM. Validated on 4090 with offload, A100, H100. Will not run
on M3 — see README for the ComfyUI + GGUF community path if you want laptop
inference.
"""

import argparse

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",          required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--height",          type=int,   default=704)
    p.add_argument("--width",           type=int,   default=1280)
    p.add_argument("--num-frames",      type=int,   default=121,
                   help="121 frames @ 24fps = ~5 seconds")
    p.add_argument("--steps",           type=int,   default=50)
    p.add_argument("--guidance-scale",  type=float, default=5.0)
    p.add_argument("--out",             default="out.mp4")
    p.add_argument("--fps",             type=int,   default=24)
    args = p.parse_args()

    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    # VAE in fp32 for numeric stability; DiT in bf16 (per the model card).
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16,
    ).to("cuda")

    print(f"generating {args.num_frames} frames at {args.width}x{args.height} "
          f"({args.steps} steps, cfg={args.guidance_scale})...")
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    ).frames[0]

    export_to_video(output, args.out, fps=args.fps)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

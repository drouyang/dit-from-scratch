"""Generate a video from a text prompt with WAN 2.1 T2V-1.3B (diffusers path).

Run:
    python inference_diffusers.py --prompt "a fluffy red panda eating bamboo on a tree branch"

`WanPipeline` for the WAN 2.1 family is in stable diffusers (>=0.36) — no
from-source install needed.

Memory: ~12 GB VRAM. Validated on 4090, A10, A100, H100.
"""

import argparse
import warnings

# Diffusers' attention_dispatch.py uses functools.lru_cache, which torch.compile's
# Dynamo flags as a "potential silent-incorrectness" risk on every call. The
# diffusers maintainers are aware; in our use the cached function is just a
# version check, so the warning is noise.
warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
    category=UserWarning,
)
# WAN's diffusers checkpoint is plain bf16/fp32 safetensors — torchao is only
# needed to deserialize torchao-quantized checkpoints, which we don't use.
warnings.filterwarnings(
    "ignore",
    message=r".*Unable to import `torchao` Tensor objects.*",
)

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",          required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--height",          type=int,   default=480)
    p.add_argument("--width",           type=int,   default=832)
    p.add_argument("--num-frames",      type=int,   default=49,
                   help="49 frames @ 16 fps = ~3 seconds")
    p.add_argument("--steps",           type=int,   default=30)
    p.add_argument("--guidance-scale",  type=float, default=5.0)
    p.add_argument("--out",             default="out.mp4")
    p.add_argument("--fps",             type=int,   default=16)
    args = p.parse_args()

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    # VAE in fp32 for numeric stability; DiT in bf16 (per the model card).
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
    )
    pipe = WanPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16,
    ).to("cuda")

    # VAE decode of a (4, T, H, W) video latent at 832×480 × 49 frames
    # otherwise needs ~5 GB on top of the transformer's working set, which
    # OOMs on a 24 GB 4090. Tiling decodes in spatial chunks; tiny wall-clock
    # cost, fits everywhere.
    pipe.vae.enable_tiling()

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

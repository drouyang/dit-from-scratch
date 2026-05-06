"""Generate video from a prompt using Wan-2.1 T2V-1.3B + a LoRA adapter.

Two modes:
    1. With --lora: load WAN base, apply the trained LoRA, generate.
    2. Without --lora: just WAN base — useful as a "before" comparison
       to see what your fine-tune actually changed.

Run:
    python sample_lora.py --prompt "an orange tabby cat on a sofa, cinematic" \\
                          --lora runs/my-style/lora_step02000.safetensors \\
                          --out cat.mp4

    # baseline (no LoRA), same prompt + seed:
    python sample_lora.py --prompt "an orange tabby cat on a sofa, cinematic" \\
                          --out cat_base.mp4
"""

import argparse

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",      required=True)
    p.add_argument("--negative",    default="low quality, blurry, distorted")
    p.add_argument("--lora",        default=None,
                   help="path to a LoRA .safetensors trained by train_lora.py")
    p.add_argument("--lora-scale",  type=float, default=1.0,
                   help="multiplier on the LoRA's contribution; 0 = base, 1 = trained, >1 = exaggerated")
    p.add_argument("--steps",       type=int,   default=30)
    p.add_argument("--guidance",    type=float, default=5.0)
    p.add_argument("--n-frames",    type=int,   default=33)
    p.add_argument("--height",      type=int,   default=480)
    p.add_argument("--width",       type=int,   default=832)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--out",         default="generated.mp4")
    args = p.parse_args()

    device = get_device()
    pipe = WanPipeline.from_pretrained(WAN_REPO, torch_dtype=torch.bfloat16).to(device)

    # Apply the trained LoRA, if provided. `load_lora_weights` calls into
    # peft under the hood — same wrapper-class injection as train_lora.py,
    # except the matrices are read from disk instead of being randomly
    # initialized and trained.
    if args.lora:
        pipe.load_lora_weights(args.lora, adapter_name="trained")
        pipe.set_adapters(["trained"], adapter_weights=[args.lora_scale])
        print(f"loaded LoRA: {args.lora}  (scale {args.lora_scale})")
    else:
        print("running base model (no LoRA)")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_frames=args.n_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).frames[0]

    export_to_video(out, args.out, fps=16)
    print(f"saved {args.out}  ({args.n_frames} frames @ 16fps)")


if __name__ == "__main__":
    main()

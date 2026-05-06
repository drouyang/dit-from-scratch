"""Self-contained Wan-2.1 T2V-1.3B + LoRA sampler, packaged as a ComfyUI node.

Reads the same WAN base + LoRA pair `lab4.2/sample_lora.py` uses, but exposes
the knobs as ComfyUI graph inputs so a non-coder can adjust them visually
and chain into other nodes (upscalers, frame interpolators, savers).

ComfyUI custom-node anatomy:
    INPUT_TYPES — declare graph inputs as a {"required"/"optional": {name: type_spec}} dict.
                  ComfyUI generates the right widget per type ("STRING" → textbox, "INT" → slider, etc.).
    RETURN_TYPES — what this node outputs. We emit "IMAGE" (a frame batch).
    FUNCTION    — name of the method ComfyUI calls per execution. Must match a method below.
    CATEGORY    — node-tree path; users find this node under "WAN".

This is a deliberately minimal example — production nodes split model
loading and LoRA application across separate nodes for composability.
"""

from functools import lru_cache

import numpy as np
import torch

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load WAN once and cache. ComfyUI invokes node methods per-execution,
    but the model should stay in VRAM across runs (~6 GB at bf16)."""
    from diffusers import WanPipeline
    pipe = WanPipeline.from_pretrained(WAN_REPO, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    return pipe


class WanLoraSampler:
    """One-node end-to-end: prompt → optional LoRA → video frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":          ("STRING",  {"multiline": True,
                                                "default": "a cute cat sitting on a sofa"}),
                "negative_prompt": ("STRING",  {"multiline": True,
                                                "default": "low quality, blurry"}),
                "num_frames":      ("INT",     {"default": 33, "min": 9, "max": 81, "step": 4}),
                "height":          ("INT",     {"default": 480, "min": 256, "max": 720, "step": 8}),
                "width":           ("INT",     {"default": 832, "min": 256, "max": 1280, "step": 8}),
                "steps":           ("INT",     {"default": 30, "min": 10, "max": 60}),
                "guidance":        ("FLOAT",   {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "seed":            ("INT",     {"default": 0}),
            },
            "optional": {
                "lora_path":       ("STRING",  {"default": "",
                                                "tooltip": "HF repo id (e.g. user/wan-mystyle-lora) "
                                                           "or a local .safetensors path; empty = base WAN"}),
                "lora_scale":      ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "WAN"

    def generate(self, prompt, negative_prompt, num_frames, height, width,
                 steps, guidance, seed, lora_path="", lora_scale=1.0):
        pipe = _load_pipeline()

        # Reset any LoRA from a prior run, then apply the requested one.
        # Done per-call so users can A/B different LoRAs in the same session.
        pipe.unload_lora_weights()
        if lora_path.strip():
            pipe.load_lora_weights(lora_path, adapter_name="trained")
            pipe.set_adapters(["trained"], adapter_weights=[float(lora_scale)])

        gen = torch.Generator("cuda").manual_seed(int(seed))
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=int(num_frames),
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=gen,
        ).frames[0]   # list of PIL.Image, one per frame

        # ComfyUI IMAGE format: (T, H, W, 3) float32 in [0, 1].
        arr = np.stack([np.asarray(f, dtype=np.float32) / 255.0 for f in out])
        return (torch.from_numpy(arr),)

"""Compare stock diffusers vs `sglang generate` with default flags.

Two runs, side-by-side, same prompt + seed + resolution + step count:

  1. **diffusers**: stock `WanPipeline` — same as lab 4.2's baseline.
  2. **sglang-diffusion**: subprocess out to `sglang generate ...` with
     default flags (FlashAttention 2, hand-written fusions, no CFG / SP / TP).
     Read the README's "technique map" for what those defaults bring.

Why subprocess for SGLang? At the time of writing, SGLang's documented
public surface is the CLI (`sglang generate`) and OpenAI-style HTTP
server (`sglang serve`). A `from sglang import DiffusionEngine` import
isn't documented, so we don't use one — we'd be calling private code
that may move.

Run:
    python benchmark_baseline.py --prompt "a fluffy red panda ..."

Skip diffusers if you've already run lab 4.2's benchmark and know that number:
    python benchmark_baseline.py --skip-diffusers
"""

import argparse
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", message=r".*Unable to import `torchao` Tensor objects.*")
warnings.filterwarnings("ignore", message=r".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=r".*sending unauthenticated requests.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

import torch
torch.set_float32_matmul_precision("high")


WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class RunResult:
    name: str
    load_secs: float = float("nan")
    gen_secs: float = float("nan")
    peak_mem: int = 0
    out_path: str = ""
    skipped: bool = False
    note: str = ""


def fmt_secs(s: float) -> str:
    if not math.isfinite(s):
        return "      —"
    return f"{s:6.1f}s"


def fmt_mem(bytes_: int) -> str:
    if not bytes_:
        return "      —"
    return f"{bytes_ / 1e9:5.2f} GB"


def find_sglang_bin() -> str | None:
    """Locate the sglang executable.

    Prefers the dedicated sglang venv at `lab4.3/.venv-sglang/bin/sglang`
    so this script (running from the shared root .venv) doesn't need
    sglang imported into its own interpreter. Falls back to PATH.
    """
    lab_dir = Path(__file__).resolve().parent
    candidate = lab_dir / ".venv-sglang" / "bin" / "sglang"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("sglang")


# ---------- diffusers baseline ------------------------------------------------

def run_diffusers(args) -> RunResult:
    """Stock WanPipeline, no torch.compile — the simplest reference."""
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    label = "diffusers"
    print(f"\n=== {label} ===")
    t0 = time.time()

    vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_REPO, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.vae.enable_tiling()

    prompt_embeds, neg_embeds = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or "",
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        device=torch.device("cuda"),
    )
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    load_secs = time.time() - t0
    print(f"  load:     {fmt_secs(load_secs)}")

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    generator = torch.Generator("cpu").manual_seed(args.seed)
    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        height=args.height, width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).frames[0]
    gen_secs = time.time() - t1
    peak_mem = torch.cuda.max_memory_allocated()

    out_path = "out_diffusers.mp4"
    export_to_video(output, out_path, fps=args.fps)
    print(f"  generate: {fmt_secs(gen_secs)}")
    print(f"  peak VRAM:{fmt_mem(peak_mem):>10}")

    del pipe, vae, output
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return RunResult(name=label, load_secs=load_secs, gen_secs=gen_secs,
                     peak_mem=peak_mem, out_path=out_path)


# ---------- sglang via subprocess ---------------------------------------------

def run_sglang_baseline(args) -> RunResult:
    """`sglang generate` with default flags."""
    label = "sglang-diffusion (defaults)"
    sglang_bin = find_sglang_bin()
    if sglang_bin is None:
        print(f"\n=== {label} ===  (skipped — sglang not found)")
        print("  set up the sglang venv (from lab4.3/):")
        print("    python3 -m venv .venv-sglang")
        print("    source .venv-sglang/bin/activate")
        print("    pip install --upgrade pip uv")
        print("    uv pip install \"sglang[diffusion]\" --prerelease=allow")
        return RunResult(name=label, skipped=True, note="sglang not found")

    print(f"\n=== {label} ===")
    out_path = "out_sglang.mp4"
    cmd = [
        sglang_bin, "generate",
        "--model-path", WAN_REPO,
        "--prompt", args.prompt,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num-frames", str(args.num_frames),
        "--num-inference-steps", str(args.steps),
        "--guidance-scale", str(args.guidance),
        "--seed", str(args.seed),
        "--output-file-path", out_path,
    ]
    print(f"  $ {' '.join(cmd)}")

    t0 = time.time()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if completed.returncode != 0:
        print(f"  FAILED (exit {completed.returncode}):")
        for line in (completed.stderr.splitlines()[-20:] if completed.stderr else []):
            print(f"    {line}")
        return RunResult(name=label, skipped=True, note=f"exit {completed.returncode}")

    print(f"  wall:     {fmt_secs(wall)}  (includes load + generate)")
    return RunResult(name=label, gen_secs=wall, out_path=out_path,
                     note="wall = load + generate (subprocess can't separate)")


# ---------- main --------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",     default="a fluffy red panda eating bamboo on a tree branch")
    p.add_argument("--height",     type=int,   default=480)
    p.add_argument("--width",      type=int,   default=832)
    p.add_argument("--num-frames", type=int,   default=49)
    p.add_argument("--steps",      type=int,   default=30)
    p.add_argument("--guidance",   type=float, default=5.0)
    p.add_argument("--fps",        type=int,   default=16)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--skip-diffusers", action="store_true",
                   help="Skip diffusers baseline (if you already have the number from lab 4.2)")
    p.add_argument("--negative-prompt", default="")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs CUDA — see README.")

    print(f"prompt: {args.prompt!r}")
    print(f"shape:  {args.num_frames} frames @ {args.width}×{args.height},  steps={args.steps},  cfg={args.guidance},  seed={args.seed}")

    results: list[RunResult] = []
    if not args.skip_diffusers:
        results.append(run_diffusers(args))
    results.append(run_sglang_baseline(args))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'backend':<32} {'load':>8} {'generate':>9} {'peak VRAM':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r.skipped:
            print(f"{r.name:<32} {'(skipped: ' + r.note + ')':>38}")
            continue
        print(f"{r.name:<32} {fmt_secs(r.load_secs):>8} {fmt_secs(r.gen_secs):>9} {fmt_mem(r.peak_mem):>10}")

    base = next((r for r in results
                 if r.name == "diffusers" and not r.skipped and math.isfinite(r.gen_secs)), None)
    if base is not None:
        print()
        for r in results:
            if r is base or r.skipped or not math.isfinite(r.gen_secs):
                continue
            print(f"  speedup vs diffusers:  {base.gen_secs / r.gen_secs:5.2f}×   ({r.name})")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Benchmark stock diffusers `WanPipeline` against SGLang-Diffusion's CLI.

The point of this script is to measure end-to-end wall-clock for the
*same prompt + seed + resolution + step count* on two backends:

  1. **diffusers**: `WanPipeline.from_pretrained(...)` + `.__call__(...)`,
     called as a Python function in this process.
  2. **SGLang-Diffusion**: subprocess out to `sglang generate ...`, parse
     the wall-clock back from process timing.

Why subprocess for SGLang and not a Python import? At the time of writing,
SGLang-Diffusion's documented public surface is the CLI (`sglang generate`)
and the OpenAI-compatible HTTP server (`sglang serve`). A `from sglang import
DiffusionEngine` style import is not officially documented, so we don't use
one — we'd be calling private code that may move.

This is a teaching benchmark, not a production one. It does:
  - same prompt, same seed, same resolution, same num_inference_steps
  - per-run wall clock + peak GPU memory (from torch.cuda.max_memory_allocated)
  - print a small table comparing the two

It does NOT:
  - run multiple trials and report mean ± std (do this yourself for real numbers)
  - separate model-load time from generation time (so first runs include warmup)
  - control for thermal throttling
  - sweep optimization toggles (Cache-DiT, attention backends, ...). The lab's
    README walks through the SGLang flags you can A/B individually with
    `sglang generate --cache-dit-config ...` etc.

Run:
    python benchmark.py --prompt "a fluffy red panda eating bamboo on a tree branch"
"""

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch


WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def fmt_secs(s: float) -> str:
    return f"{s:6.1f}s"


def fmt_mem(bytes_: int) -> str:
    return f"{bytes_ / 1e9:5.2f} GB"


# ---------- diffusers baseline ------------------------------------------------

def run_diffusers(args, *, compile_mode: str | None = None) -> tuple[float, int, str]:
    """Run stock diffusers WanPipeline, optionally with torch.compile.

    compile_mode:
      None              -- pure Python WanPipeline.__call__ (the baseline)
      "default"         -- + torch.compile with Inductor kernel fusion
                           (no CUDA Graphs; CFG-safe with WanPipeline)
      "reduce-overhead" -- + torch.compile with CUDA graph capture
                           (BROKEN with WanPipeline's two-pass CFG: the second
                            transformer call overwrites the first call's output
                            buffer while it's still being read for extrapolation;
                            you'd need to clone the transformer output to use it)
      "max-autotune"    -- + torch.compile with autotune
                           (slowest compile, fastest steady-state; no CUDA Graphs)

    Returns (gen_seconds, peak_gpu_bytes, output_path). For compiled runs we
    do a warmup call before timing so the reported number is steady-state,
    not first-call-with-compile-time.
    """
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    label = "diffusers" if compile_mode is None else f"diffusers + torch.compile({compile_mode})"
    print(f"\n=== {label} ===")
    t0 = time.time()

    vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_REPO, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.vae.enable_tiling()  # avoid VAE-decode OOM at 832x480 × 49 frames on a 24 GB 4090
    print(f"  load:     {fmt_secs(time.time() - t0)}")

    if compile_mode is not None:
        pipe.transformer = torch.compile(pipe.transformer, mode=compile_mode)
        # Warmup pass — first call triggers graph capture / autotune. Use a
        # tiny step count so warmup is short relative to the real run.
        print(f"  warming up torch.compile (this is the slow part — ~30-60s)...")
        tw = time.time()
        _ = pipe(
            prompt=args.prompt,
            height=args.height, width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=2,                  # just enough to compile
            guidance_scale=args.guidance,
            generator=torch.Generator("cuda").manual_seed(0),
        )
        print(f"  warmup:   {fmt_secs(time.time() - tw)}")

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    generator = torch.Generator("cuda").manual_seed(args.seed)
    output = pipe(
        prompt=args.prompt,
        height=args.height, width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).frames[0]
    gen_secs = time.time() - t1
    peak_mem = torch.cuda.max_memory_allocated()

    suffix = "" if compile_mode is None else f"_compile_{compile_mode}"
    out_path = f"out_diffusers{suffix}.mp4"
    export_to_video(output, out_path, fps=args.fps)
    print(f"  generate: {fmt_secs(gen_secs)} (steady-state, post-warmup)")
    print(f"  peak VRAM:{fmt_mem(peak_mem):>10}")
    print(f"  saved {out_path}")

    # Tear down so the next backend has the GPU to itself.
    del pipe, vae, output
    torch.cuda.empty_cache()
    return gen_secs, peak_mem, out_path


# ---------- SGLang via subprocess ---------------------------------------------

def run_sglang(args) -> tuple[float, int, str]:
    """Drive SGLang via the documented `sglang generate` CLI.

    Memory measurement here is approximate — we can't use
    torch.cuda.max_memory_allocated across a subprocess, so we time the
    whole thing and trust SGLang's own profiling for VRAM.
    """
    if shutil.which("sglang") is None:
        print("\n=== sglang ===  (skipped — `sglang` not in PATH)")
        print("  install with:  uv pip install \"sglang[diffusion]\" --prerelease=allow")
        return float("nan"), 0, ""

    print("\n=== sglang ===")
    out_path = "out_sglang.mp4"
    cmd = [
        "sglang", "generate",
        "--model-path", WAN_REPO,
        "--prompt", args.prompt,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num-frames", str(args.num_frames),
        "--num-inference-steps", str(args.steps),
        "--guidance-scale", str(args.guidance),
        "--seed", str(args.seed),
        "--save-output", out_path,
    ]
    # The exact flag names above are documented in
    # https://sgl-project.github.io/diffusion/api/cli.html — but flag names do
    # drift across SGLang releases. If `sglang generate --help` shows
    # something different, adjust accordingly.
    print(f"  $ {' '.join(cmd)}")

    t0 = time.time()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if completed.returncode != 0:
        print(f"  FAILED (exit {completed.returncode}):")
        print(completed.stderr.splitlines()[-20:] if completed.stderr else "  (no stderr)")
        return float("nan"), 0, ""

    print(f"  wall:     {fmt_secs(wall)}  (includes load + generate, can't separate)")
    print(f"  saved {out_path}")
    return wall, 0, out_path


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
    p.add_argument("--skip-diffusers", action="store_true")
    p.add_argument("--skip-compile",   action="store_true",
                   help="Skip the diffusers + torch.compile run (~1-2 min extra)")
    p.add_argument("--skip-sglang",    action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs CUDA — see README compute notes.")

    print(f"prompt: {args.prompt!r}")
    print(f"shape:  {args.num_frames} frames @ {args.width}×{args.height},  steps={args.steps},  cfg={args.guidance},  seed={args.seed}")

    diff_secs = comp_secs = sg_secs = float("nan")
    diff_mem  = comp_mem  = sg_mem  = 0
    if not args.skip_diffusers:
        diff_secs, diff_mem, _ = run_diffusers(args)
    if not args.skip_compile:
        comp_secs, comp_mem, _ = run_diffusers(args, compile_mode="default")
    if not args.skip_sglang:
        sg_secs, sg_mem, _ = run_sglang(args)

    print("\n=== comparison ===")
    print(f"{'backend':<28} {'wall':>10}  {'peak VRAM':>12}")
    print(f"{'-'*52}")
    print(f"{'diffusers':<28} {fmt_secs(diff_secs):>10}  {fmt_mem(diff_mem) if diff_mem else '         —':>12}")
    print(f"{'diffusers + torch.compile':<28} {fmt_secs(comp_secs):>10}  {fmt_mem(comp_mem) if comp_mem else '         —':>12}")
    print(f"{'sglang-diffusion':<28} {fmt_secs(sg_secs):>10}  {'(see sglang logs)':>12}")

    if diff_secs == diff_secs:  # finite
        if comp_secs == comp_secs:
            print(f"\ntorch.compile speedup: {diff_secs / comp_secs:.2f}×  (vs diffusers baseline)")
        if sg_secs == sg_secs:
            print(f"sglang        speedup: {diff_secs / sg_secs:.2f}×  (vs diffusers baseline)")


if __name__ == "__main__":
    main()

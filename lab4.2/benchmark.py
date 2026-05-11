"""Benchmark stock diffusers `WanPipeline` against `torch.compile`.

Three runs, side-by-side, same prompt + seed + resolution + step count:

  1. **diffusers baseline**: `WanPipeline.from_pretrained(...)` + `.__call__(...)`
  2. **diffusers + torch.compile(default)**: same pipeline, transformer
     wrapped with `torch.compile(mode="default")` — Inductor kernel fusion,
     ~1.5–1.7× steady-state speedup, ~30–60 s warmup.
  3. **diffusers + torch.compile(max-autotune)**: same as (2) but
     `mode="max-autotune"` — autotunes Triton kernels for ~5–10% extra speedup,
     5–10 minutes warmup. Skip via `--skip-max-autotune` for fast iteration.

This is the diffusers-only inference benchmark. The SGLang side of the
comparison lives in lab4.3 (`benchmark_baseline.py`, `benchmark_kernels.py`,
`benchmark_parallel.py`), which exercises a separate inference runtime.

It does:
  - same prompt, same seed, same resolution, same num_inference_steps
  - per-run wall clock + peak GPU memory (from torch.cuda.max_memory_allocated)
  - print a small table comparing the three

It does NOT:
  - run multiple trials and report mean ± std (do this yourself for real numbers)
  - separate model-load time from generation time (so first runs include warmup)
  - control for thermal throttling

Run:
    python benchmark.py --prompt "a fluffy red panda eating bamboo on a tree branch"
    python benchmark.py --skip-max-autotune   # faster iteration; skips 5-10min compile
"""

import argparse
import logging
import math
import time
import warnings
from dataclasses import dataclass

# Same suppression block as lab4.1/inference_diffusers.py — these warnings are
# noise for our use of WAN's public diffusers checkpoint.
warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Unable to import `torchao` Tensor objects.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*local_dir_use_symlinks.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*sending unauthenticated requests.*",
)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
# torchao import nag is emitted via diffusers' own logger (logger.warning),
# not warnings.warn, so the regex filter above doesn't catch it. Lower the
# diffusers logger to ERROR.
logging.getLogger("diffusers").setLevel(logging.ERROR)

import torch

# Enable TF32 on Ampere+ GPUs — ~2× speedup on fp32 matmuls (the VAE is fp32
# in this pipeline; the transformer is bf16 and unaffected). Quality loss is
# imperceptible for video frames.
torch.set_float32_matmul_precision("high")


WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class RunResult:
    """Per-backend timing record. Any field can be NaN ("not measured")
    or 0 ("not applicable"). The summary at the end of main() reads from
    these — no scraping of stdout required."""
    name: str
    load_secs: float = float("nan")
    warmup_secs: float = float("nan")
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


# ---------- diffusers, optionally compiled ------------------------------------

def run_diffusers(args, *, compile_mode: str | None = None) -> RunResult:
    """Run stock diffusers WanPipeline, optionally with torch.compile.

    compile_mode:
      None              -- pure Python WanPipeline.__call__ (the baseline)
      "default"         -- + torch.compile with Inductor kernel fusion
                           (no CUDA Graphs; CFG-safe with WanPipeline)
      "reduce-overhead" -- + torch.compile with CUDA graph capture
                           (BROKEN with WanPipeline's two-pass CFG: the second
                            transformer call overwrites the first call's output
                            buffer while it's still being read for extrapolation;
                            see README for workarounds)
      "max-autotune"    -- + torch.compile with Triton autotune
                           (slowest compile, fastest steady-state; no CUDA Graphs)

    For compiled runs we do a warmup call before timing so the reported number
    is steady-state, not first-call-with-compile-time.
    """
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    label = "diffusers" if compile_mode is None else f"diffusers + torch.compile({compile_mode})"
    print(f"\n=== {label} ===")
    t0 = time.time()

    vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_REPO, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.vae.enable_tiling()  # avoid VAE-decode OOM at 832x480 × 49 frames on a 24 GB 4090

    # Pre-encode the prompts and offload umT5-XXL (~11 GB) to CPU before
    # sampling. Without this, the compile-mode run OOMs on a 24 GB 4090 — the
    # text encoder weights sit unused on cuda:0 while the compiled
    # transformer's activations + Inductor caches need every spare GB.
    # We do it for the baseline too so peak-VRAM rows are comparable.
    prompt_embeds, neg_embeds = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=getattr(args, "negative_prompt", "") or "",
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        device=torch.device("cuda"),
    )
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    load_secs = time.time() - t0
    print(f"  load:     {fmt_secs(load_secs)}")

    warmup_secs = float("nan")
    if compile_mode is not None:
        pipe.transformer = torch.compile(pipe.transformer, mode=compile_mode)
        compile_msg = "(~30-60s warmup)" if compile_mode == "default" else "(~5-10min warmup — Triton autotune)"
        print(f"  warming up torch.compile {compile_msg}...")
        tw = time.time()
        _ = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            height=args.height, width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=2,                  # just enough to compile
            guidance_scale=args.guidance,
            generator=torch.Generator("cpu").manual_seed(0),
        )
        warmup_secs = time.time() - tw
        print(f"  warmup:   {fmt_secs(warmup_secs)}")

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

    suffix = "" if compile_mode is None else f"_compile_{compile_mode}"
    out_path = f"out_diffusers{suffix}.mp4"
    export_to_video(output, out_path, fps=args.fps)
    print(f"  generate: {fmt_secs(gen_secs)} (steady-state, post-warmup)")
    print(f"  peak VRAM:{fmt_mem(peak_mem):>10}")
    print(f"  saved {out_path}")

    # Tear down so the next backend has the GPU to itself. With torch.compile
    # we also need to flush Dynamo's compiled-artifact cache and gc Python-side
    # references; otherwise the next backend OOMs even after empty_cache().
    del pipe, vae, output
    if compile_mode is not None:
        torch._dynamo.reset()
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return RunResult(
        name=label,
        load_secs=load_secs,
        warmup_secs=warmup_secs,
        gen_secs=gen_secs,
        peak_mem=peak_mem,
        out_path=out_path,
    )


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
    p.add_argument("--skip-diffusers",    action="store_true",
                   help="Skip the stock diffusers baseline (e.g. you only want to time compile)")
    p.add_argument("--skip-compile",      action="store_true",
                   help="Skip the diffusers + torch.compile(default) run")
    p.add_argument("--skip-max-autotune", action="store_true",
                   help="Skip the torch.compile(max-autotune) run (~5-10min compile)")
    p.add_argument("--negative-prompt", default="",
                   help="Negative prompt text. Empty string uses the model's default.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs CUDA — see README compute notes.")

    print(f"prompt: {args.prompt!r}")
    print(f"shape:  {args.num_frames} frames @ {args.width}×{args.height},  steps={args.steps},  cfg={args.guidance},  seed={args.seed}")

    results: list[RunResult] = []
    if not args.skip_diffusers:
        results.append(run_diffusers(args))
    if not args.skip_compile:
        results.append(run_diffusers(args, compile_mode="default"))
    if not args.skip_max_autotune:
        results.append(run_diffusers(args, compile_mode="max-autotune"))

    # Summary block printed at the very end. Pulls from the RunResult records,
    # so even if intermediate stdout is cluttered (progress bars, dynamo notes),
    # this final block is clean and self-contained.
    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"prompt: {args.prompt!r}")
    print(f"shape:  {args.num_frames} frames @ {args.width}×{args.height},  steps={args.steps},  cfg={args.guidance},  seed={args.seed}")
    print()
    header = f"{'backend':<42} {'load':>8} {'warmup':>8} {'generate':>9} {'peak VRAM':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r.skipped:
            print(f"{r.name:<42} {'(skipped: ' + r.note + ')':>38}")
            continue
        print(
            f"{r.name:<42} "
            f"{fmt_secs(r.load_secs):>8} "
            f"{fmt_secs(r.warmup_secs):>8} "
            f"{fmt_secs(r.gen_secs):>9} "
            f"{fmt_mem(r.peak_mem):>10}"
        )

    # Speedups vs the plain-diffusers baseline (steady-state generate seconds).
    base = next(
        (r for r in results
         if r.name == "diffusers" and not r.skipped and math.isfinite(r.gen_secs)),
        None,
    )
    if base is not None:
        print()
        for r in results:
            if r is base or r.skipped or not math.isfinite(r.gen_secs):
                continue
            print(f"  speedup vs diffusers:  {base.gen_secs / r.gen_secs:5.2f}×   ({r.name})")

    # Output paths, so you don't have to grep stdout to find the videos.
    saved = [r for r in results if r.out_path]
    if saved:
        print()
        for r in saved:
            print(f"  output: {r.out_path:<40} ({r.name})")
    print("=" * 88)


if __name__ == "__main__":
    main()

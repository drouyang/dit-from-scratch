"""Sweep SGLang attention backends on the same prompt + seed.

Five configurations, all routed through `sglang generate` as subprocesses
against the lab's sglang venv:

  - fa             FlashAttention 2 (default)
  - _flash_3_hub   FlashAttention 3 (Hopper async — skipped on non-Hopper)
  - sage           SageAttention (INT8 Q/K/V — small quality drop, FLOPs win)
  - xformers       xFormers memory-efficient attention
  - native         PyTorch SDPA (no backend hint — usually dispatches to FA2)

All produce identical-enough output to be visually comparable; only Sage is
mathematically non-identical (INT8 attention rounds).

Run:
    python benchmark_kernels.py --prompt "a fluffy red panda ..."
    python benchmark_kernels.py --skip-fa3   # if you don't have an H100
"""

import argparse
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

BACKENDS = [
    ("fa",            "FlashAttention 2"),
    ("_flash_3_hub",  "FlashAttention 3 (Hopper)"),
    ("sage",          "SageAttention (INT8)"),
    ("xformers",      "xFormers"),
    ("native",        "PyTorch SDPA (auto)"),
]


@dataclass
class RunResult:
    backend: str
    label: str
    wall_secs: float = float("nan")
    out_path: str = ""
    skipped: bool = False
    note: str = ""


def fmt_secs(s: float) -> str:
    if not math.isfinite(s):
        return "      —"
    return f"{s:6.1f}s"


def find_sglang_bin() -> str | None:
    lab_dir = Path(__file__).resolve().parent
    candidate = lab_dir / ".venv-sglang" / "bin" / "sglang"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("sglang")


def run_backend(args, sglang_bin: str, backend: str, label: str) -> RunResult:
    print(f"\n=== {label} ({backend}) ===")
    out_path = f"out_sglang_{backend.replace('_', '')}.mp4"
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
        "--attention-backend", backend,
        "--output-file-path", out_path,
    ]
    print(f"  $ {' '.join(cmd)}")

    t0 = time.time()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if completed.returncode != 0:
        print(f"  FAILED (exit {completed.returncode}):")
        for line in (completed.stderr.splitlines()[-15:] if completed.stderr else []):
            print(f"    {line}")
        return RunResult(backend=backend, label=label, skipped=True,
                         note=f"exit {completed.returncode}")

    print(f"  wall:     {fmt_secs(wall)}  (includes load + generate)")
    return RunResult(backend=backend, label=label, wall_secs=wall, out_path=out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",     default="a fluffy red panda eating bamboo on a tree branch")
    p.add_argument("--height",     type=int,   default=480)
    p.add_argument("--width",      type=int,   default=832)
    p.add_argument("--num-frames", type=int,   default=49)
    p.add_argument("--steps",      type=int,   default=30)
    p.add_argument("--guidance",   type=float, default=5.0)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--skip-fa3",   action="store_true",
                   help="Skip FlashAttention 3 (non-Hopper GPUs; falls back to FA2 anyway)")
    p.add_argument("--skip-sage",  action="store_true",
                   help="Skip SageAttention")
    args = p.parse_args()

    sglang_bin = find_sglang_bin()
    if sglang_bin is None:
        raise SystemExit("sglang not found. Set up lab4.3/.venv-sglang/ per the README.")

    print(f"prompt: {args.prompt!r}")
    print(f"shape:  {args.num_frames} frames @ {args.width}×{args.height},  steps={args.steps}")

    results: list[RunResult] = []
    for backend, label in BACKENDS:
        if backend == "_flash_3_hub" and args.skip_fa3:
            continue
        if backend == "sage" and args.skip_sage:
            continue
        results.append(run_backend(args, sglang_bin, backend, label))

    print("\n" + "=" * 80)
    print("SUMMARY  (kernel sweep, all sglang)")
    print("=" * 80)
    header = f"{'attention backend':<32} {'wall':>9}"
    print(header)
    print("-" * len(header))

    # Find the fastest non-skipped result for relative speedup column.
    finite = [r for r in results if not r.skipped and math.isfinite(r.wall_secs)]
    fastest = min((r.wall_secs for r in finite), default=None)

    for r in results:
        if r.skipped:
            print(f"{r.label + ' (' + r.backend + ')':<32} {'(skipped: ' + r.note + ')':>40}")
            continue
        rel = ""
        if fastest is not None and r.wall_secs != fastest:
            rel = f"   ({r.wall_secs / fastest:.2f}× slower than fastest)"
        elif fastest is not None:
            rel = "   (fastest)"
        print(f"{r.label + ' (' + r.backend + ')':<32} {fmt_secs(r.wall_secs):>9}{rel}")
    print("=" * 80)


if __name__ == "__main__":
    main()

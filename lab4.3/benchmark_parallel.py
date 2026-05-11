"""Sweep SGLang parallelism configurations on the same prompt + seed.

Six configurations, auto-skipping any that need more GPUs than available:

  - 1 GPU baseline       no parallelism flags
  - CFG-parallel         --enable-cfg-parallel    (2 GPUs)
  - Ulysses              --sp-degree N --ulysses-degree N --ring-degree 1
  - Ring                 --sp-degree N --ulysses-degree 1 --ring-degree N
  - USP hybrid (4 GPU)   --sp-degree 4 --ulysses-degree 2 --ring-degree 2
  - Tensor parallel      --tp-size N

All configurations should produce identical output for the same seed
(parallelism is mathematically equivalent to single-GPU). Differences are
wall clock and per-GPU peak memory.

Run:
    python benchmark_parallel.py --prompt "a fluffy red panda ..."
"""

import argparse
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class Config:
    label: str
    min_gpus: int
    flags: list[str]
    out_suffix: str


CONFIGS = [
    Config("baseline (1 GPU)",     1, [], "1gpu"),
    Config("CFG-parallel (2 GPU)", 2, ["--enable-cfg-parallel"], "cfgp"),
    Config("Ulysses (2 GPU)",      2,
           ["--sp-degree", "2", "--ulysses-degree", "2", "--ring-degree", "1"], "ulysses2"),
    Config("Ring (2 GPU)",         2,
           ["--sp-degree", "2", "--ulysses-degree", "1", "--ring-degree", "2"], "ring2"),
    Config("Ulysses (4 GPU)",      4,
           ["--sp-degree", "4", "--ulysses-degree", "4", "--ring-degree", "1"], "ulysses4"),
    Config("Ring (4 GPU)",         4,
           ["--sp-degree", "4", "--ulysses-degree", "1", "--ring-degree", "4"], "ring4"),
    Config("USP hybrid (4 GPU)",   4,
           ["--sp-degree", "4", "--ulysses-degree", "2", "--ring-degree", "2"], "usp4"),
    Config("tensor-parallel (2 GPU)", 2, ["--tp-size", "2"], "tp2"),
]


@dataclass
class RunResult:
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


def num_gpus() -> int:
    """nvidia-smi -L count. Doesn't import torch (sglang runs in its own venv)."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


def run_config(args, sglang_bin: str, cfg: Config) -> RunResult:
    print(f"\n=== {cfg.label} ===")
    out_path = f"out_sglang_{cfg.out_suffix}.mp4"
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
        *cfg.flags,
    ]
    print(f"  $ {' '.join(cmd)}")

    t0 = time.time()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if completed.returncode != 0:
        print(f"  FAILED (exit {completed.returncode}):")
        for line in (completed.stderr.splitlines()[-15:] if completed.stderr else []):
            print(f"    {line}")
        return RunResult(label=cfg.label, skipped=True, note=f"exit {completed.returncode}")

    print(f"  wall:     {fmt_secs(wall)}  (includes load + generate)")
    return RunResult(label=cfg.label, wall_secs=wall, out_path=out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",     default="a fluffy red panda eating bamboo on a tree branch")
    p.add_argument("--height",     type=int,   default=480)
    p.add_argument("--width",      type=int,   default=832)
    p.add_argument("--num-frames", type=int,   default=49)
    p.add_argument("--steps",      type=int,   default=30)
    p.add_argument("--guidance",   type=float, default=5.0)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    sglang_bin = find_sglang_bin()
    if sglang_bin is None:
        raise SystemExit("sglang not found. Set up lab4.3/.venv-sglang/ per the README.")

    n_gpus = num_gpus()
    print(f"available GPUs (from nvidia-smi -L): {n_gpus}")
    print(f"prompt: {args.prompt!r}")
    print(f"shape:  {args.num_frames} frames @ {args.width}×{args.height},  steps={args.steps}")

    results: list[RunResult] = []
    for cfg in CONFIGS:
        if cfg.min_gpus > n_gpus:
            print(f"\n=== {cfg.label} ===  (skipped — needs {cfg.min_gpus} GPUs, have {n_gpus})")
            results.append(RunResult(label=cfg.label, skipped=True,
                                      note=f"needs {cfg.min_gpus} GPUs"))
            continue
        results.append(run_config(args, sglang_bin, cfg))

    print("\n" + "=" * 80)
    print("SUMMARY  (parallelism sweep, all sglang)")
    print("=" * 80)
    header = f"{'configuration':<32} {'wall':>9}"
    print(header)
    print("-" * len(header))

    baseline = next((r for r in results
                     if r.label.startswith("baseline") and not r.skipped
                     and math.isfinite(r.wall_secs)), None)

    for r in results:
        if r.skipped:
            print(f"{r.label:<32} {'(skipped: ' + r.note + ')':>40}")
            continue
        rel = ""
        if baseline is not None and r is not baseline:
            rel = f"   ({baseline.wall_secs / r.wall_secs:.2f}× speedup over baseline)"
        elif baseline is not None and r is baseline:
            rel = "   (baseline)"
        print(f"{r.label:<32} {fmt_secs(r.wall_secs):>9}{rel}")
    print("=" * 80)


if __name__ == "__main__":
    main()

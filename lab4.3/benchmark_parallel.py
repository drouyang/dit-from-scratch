"""A/B one SGLang parallelism configuration per invocation.

Single config per invocation, lab-4.2-style. Pass the sglang parallelism flags
directly — semantic names, not opaque preset strings:

    python benchmark_parallel.py                                    # 1 GPU baseline
    python benchmark_parallel.py --enable-cfg-parallel              # CFG-parallel, 2 GPUs
    python benchmark_parallel.py --ulysses-degree 4 --ring-degree 1 # pure Ulysses, 4 GPUs
    python benchmark_parallel.py --ulysses-degree 1 --ring-degree 4 # pure Ring, 4 GPUs
    python benchmark_parallel.py --ulysses-degree 2 --ring-degree 2 # USP hybrid, 4 GPUs
    python benchmark_parallel.py --tp-size 2                        # tensor-parallel, 2 GPUs

`--sp-degree` is derived from `ulysses_degree * ring_degree` when either is set,
so you don't repeat yourself. Pre-flight check fails fast if the requested
GPU count is more than `nvidia-smi -L` reports.

Each invocation runs `sglang generate` twice (cold + warm subprocess) and
reports `config | wall (cold) | wall (warm) | peak VRAM (per GPU)`. Peak VRAM
is the max across visible GPUs sampled by nvidia-smi during the warm run.

Why subprocess and not HTTP? Same reason as benchmark_kernels.py — the A/B
is about steady-state wall time, not about warm/serve dynamics.
"""

import argparse
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class RunResult:
    label: str
    cold_secs: float = float("nan")
    warm_secs: float = float("nan")
    peak_mem_gb_smi: float = float("nan")


def fmt_secs(s: float) -> str:
    if not math.isfinite(s):
        return "      —"
    return f"{s:6.1f}s"


def fmt_mem_gb(gb: float) -> str:
    if not math.isfinite(gb):
        return "      —"
    return f"{gb:5.2f} GB"


# ---------- nvidia-smi peak-VRAM polling --------------------------------------

class VramPoller(threading.Thread):
    """Track peak `nvidia-smi --query-gpu=memory.used` across all visible GPUs."""
    def __init__(self, interval_secs: float = 0.5):
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self._interval = interval_secs
        self._peak_mib_per_gpu: dict[int, int] = {}
        self._available = shutil.which("nvidia-smi") is not None

    def run(self) -> None:
        if not self._available:
            return
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=3,
                )
                for line in out.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    idx_s, mib_s = (p.strip() for p in line.split(","))
                    idx, mib = int(idx_s), int(mib_s)
                    self._peak_mib_per_gpu[idx] = max(
                        self._peak_mib_per_gpu.get(idx, 0), mib)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    FileNotFoundError, ValueError):
                pass
            self._stop.wait(self._interval)

    def stop_and_peak_gb(self) -> float:
        """Return the largest per-GPU peak observed (GB)."""
        self._stop.set()
        self.join(timeout=5)
        if not self._available or not self._peak_mib_per_gpu:
            return float("nan")
        return max(self._peak_mib_per_gpu.values()) / 1024.0


def find_sglang_bin() -> str | None:
    lab_dir = Path(__file__).resolve().parent
    candidate = lab_dir / ".venv-sglang" / "bin" / "sglang"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("sglang")


def num_visible_gpus() -> int:
    """`nvidia-smi -L` count. No torch import (sglang lives in its own venv)."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"],
                             capture_output=True, text=True, check=True)
        return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


def parallelism_flags_and_label(args) -> tuple[list[str], str, int]:
    """Translate semantic CLI args into sglang flags + a one-line label + min GPUs."""
    flags: list[str] = []
    parts: list[str] = []
    min_gpus = 1

    sp_required = 1
    if args.ulysses_degree > 1 or args.ring_degree > 1:
        sp_required = args.ulysses_degree * args.ring_degree
        flags += ["--sp-degree", str(sp_required),
                  "--ulysses-degree", str(args.ulysses_degree),
                  "--ring-degree", str(args.ring_degree)]
        if args.ulysses_degree > 1 and args.ring_degree == 1:
            parts.append(f"Ulysses-{args.ulysses_degree}")
        elif args.ring_degree > 1 and args.ulysses_degree == 1:
            parts.append(f"Ring-{args.ring_degree}")
        else:
            parts.append(f"USP {args.ulysses_degree}×{args.ring_degree}")

    cfg_required = 1
    if args.enable_cfg_parallel:
        flags += ["--enable-cfg-parallel"]
        cfg_required = 2
        parts.append("CFG-parallel")

    tp_required = 1
    if args.tp_size > 1:
        flags += ["--tp-size", str(args.tp_size)]
        tp_required = args.tp_size
        parts.append(f"TP-{args.tp_size}")

    if not parts:
        label = "baseline (1 GPU)"
    else:
        label = " + ".join(parts)

    min_gpus = max(sp_required, cfg_required, tp_required)
    return flags, label, min_gpus


def run_once(sglang_bin: str, args, parallel_flags: list[str],
             out_path: str, track_vram: bool) -> float:
    """Run `sglang generate` once. Returns wall_secs. Peak VRAM via the poller."""
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
        *parallel_flags,
    ]
    print(f"  $ {' '.join(cmd)}")

    t0 = time.time()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if completed.returncode != 0:
        print(f"  FAILED (exit {completed.returncode}):")
        for line in (completed.stderr.splitlines()[-15:] if completed.stderr else []):
            print(f"    {line}")
        raise SystemExit(f"sglang generate exited {completed.returncode}")
    return wall


def main():
    p = argparse.ArgumentParser()
    # Semantic parallelism flags (pass-through to sglang generate):
    p.add_argument("--enable-cfg-parallel", action="store_true",
                   help="CFG parallel: cond + uncond on separate GPUs (needs 2 GPUs)")
    p.add_argument("--ulysses-degree", type=int, default=1,
                   help="Ulysses (head-shard via all-to-all) degree")
    p.add_argument("--ring-degree", type=int, default=1,
                   help="Ring (sequence-shard via streaming K/V) degree")
    p.add_argument("--tp-size", type=int, default=1,
                   help="Tensor parallel size (sharded linear weights)")
    # Generation params:
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

    flags, label, min_gpus = parallelism_flags_and_label(args)
    available = num_visible_gpus()
    if min_gpus > available:
        raise SystemExit(
            f"{label} needs {min_gpus} GPUs but nvidia-smi -L reports {available}.")

    suffix = label.lower().replace(" ", "").replace("+", "_").replace("(", "").replace(")", "").replace("×", "x")
    out_path = f"out_sglang_{suffix}.mp4"

    print(f"config:  {label}  ({min_gpus} GPU{'s' if min_gpus > 1 else ''})")
    print(f"prompt:  {args.prompt!r}")
    print(f"shape:   {args.num_frames} frames @ {args.width}×{args.height},  steps={args.steps}")

    print(f"\n=== cold run ===")
    cold_secs = run_once(sglang_bin, args, flags, out_path, track_vram=False)
    print(f"  wall:    {fmt_secs(cold_secs)}")

    print(f"\n=== warm run ===")
    vram = VramPoller(); vram.start()
    try:
        warm_secs = run_once(sglang_bin, args, flags, out_path, track_vram=True)
    finally:
        peak_gb = vram.stop_and_peak_gb()
    print(f"  wall:    {fmt_secs(warm_secs)}")
    print(f"  peak VRAM (per GPU, max): {fmt_mem_gb(peak_gb)}  (nvidia-smi)")

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    header = (f"{'config':<32} {'cold':>9} {'warm':>9} {'peak VRAM/GPU':>14}")
    print(header)
    print("-" * len(header))
    print(f"{label:<32} {fmt_secs(cold_secs):>9} {fmt_secs(warm_secs):>9} {fmt_mem_gb(peak_gb):>14}")
    print("=" * 80)
    print("Note: speedup vs baseline requires running both invocations and")
    print("      comparing the warm columns.")


if __name__ == "__main__":
    main()

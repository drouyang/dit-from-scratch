"""A/B one SGLang attention backend per invocation.

Single config per invocation, lab-4.2-style. Each invocation runs
`sglang generate` *twice* as separate subprocesses:

  - cold: fresh process, kernels may need recompiling
  - warm: second process, hits sglang's cubin disk cache from the first run

Reported as one row of `config | wall (cold) | wall (warm) | speedup (warm) | peak VRAM`.
Peak VRAM is polled from `nvidia-smi` for the lifetime of the warm run.

Why subprocess and not HTTP like benchmark_baseline.py? The kernel A/B
only cares about steady-state generation time, not load. Subprocess is
simpler — no server lifecycle to manage. `wall` here is load + generate
combined, but since every row pays the same load cost, the differences
between rows are still the attention-backend differences.

Run one backend at a time:
    python benchmark_kernels.py --attention-backend fa
    python benchmark_kernels.py --attention-backend _flash_3_hub
    python benchmark_kernels.py --attention-backend sage
    python benchmark_kernels.py --attention-backend xformers
    python benchmark_kernels.py --attention-backend native
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

VALID_BACKENDS = ["fa", "_flash_3_hub", "sage", "xformers", "native"]


@dataclass
class RunResult:
    backend: str
    cold_secs: float = float("nan")
    warm_secs: float = float("nan")
    peak_mem_gb_smi: float = float("nan")
    skipped: bool = False
    note: str = ""


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
        self._peak_mib = 0
        self._available = shutil.which("nvidia-smi") is not None

    def run(self) -> None:
        if not self._available:
            return
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=3,
                )
                mibs = [int(x.strip()) for x in out.stdout.splitlines() if x.strip()]
                if mibs:
                    self._peak_mib = max(self._peak_mib, max(mibs))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    FileNotFoundError, ValueError):
                pass
            self._stop.wait(self._interval)

    def stop_and_peak_gb(self) -> float:
        self._stop.set()
        self.join(timeout=5)
        if not self._available or self._peak_mib == 0:
            return float("nan")
        return self._peak_mib / 1024.0


def find_sglang_bin() -> str | None:
    lab_dir = Path(__file__).resolve().parent
    candidate = lab_dir / ".venv-sglang" / "bin" / "sglang"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("sglang")


def run_once(sglang_bin: str, args, out_path: str,
             track_vram: bool) -> tuple[float, float]:
    """Run `sglang generate` once. Returns (wall_secs, peak_gb_or_nan).

    `peak_gb_or_nan` is only populated when `track_vram` is True.
    """
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
        "--attention-backend", args.attention_backend,
        "--output-file-path", out_path,
    ]
    print(f"  $ {' '.join(cmd)}")

    vram = VramPoller() if track_vram else None
    if vram is not None:
        vram.start()
    t0 = time.time()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    peak_gb = vram.stop_and_peak_gb() if vram is not None else float("nan")

    if completed.returncode != 0:
        print(f"  FAILED (exit {completed.returncode}):")
        for line in (completed.stderr.splitlines()[-15:] if completed.stderr else []):
            print(f"    {line}")
        raise SystemExit(f"sglang generate exited {completed.returncode}")

    return wall, peak_gb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attention-backend", required=True, choices=VALID_BACKENDS,
                   help="One of: " + ", ".join(VALID_BACKENDS))
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

    print(f"backend: {args.attention_backend}")
    print(f"prompt:  {args.prompt!r}")
    print(f"shape:   {args.num_frames} frames @ {args.width}×{args.height},  steps={args.steps}")

    out_path = f"out_sglang_{args.attention_backend.replace('_', '')}.mp4"

    print(f"\n=== cold run ===")
    cold_secs, _ = run_once(sglang_bin, args, out_path, track_vram=False)
    print(f"  wall:    {fmt_secs(cold_secs)}")

    print(f"\n=== warm run ===")
    warm_secs, peak_gb = run_once(sglang_bin, args, out_path, track_vram=True)
    print(f"  wall:    {fmt_secs(warm_secs)}")
    print(f"  peak VRAM: {fmt_mem_gb(peak_gb)}  (nvidia-smi, warm run only)")

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    header = (f"{'config':<32} {'cold':>9} {'warm':>9} {'peak VRAM':>10}")
    print(header)
    print("-" * len(header))
    print(f"{'--attention-backend ' + args.attention_backend:<32} "
          f"{fmt_secs(cold_secs):>9} {fmt_secs(warm_secs):>9} {fmt_mem_gb(peak_gb):>10}")
    print("=" * 80)
    print("Note: speedup vs `--attention-backend fa` requires running both invocations")
    print("      and comparing the warm columns.")


if __name__ == "__main__":
    main()

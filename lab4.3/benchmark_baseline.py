"""Compare stock diffusers vs sglang-diffusion (HTTP server) on the same prompt + seed.

Two rows, same six columns as lab 4.2's benchmark.py:

  config | model load | first call | second call | speedup (second) | peak VRAM

  1. **diffusers** (in-process) — same WanPipeline as lab 4.2, called twice
     in a row to capture the cold/warm split. Both should match (no JIT here);
     the second-call number is the comparable steady state.
  2. **sglang-diffusion** (HTTP) — spawn `sglang serve` once, send two POST
     requests, time each. Server stays resident, so load / first / second
     separate cleanly. Peak VRAM tracked by polling `nvidia-smi`.

Why HTTP, not subprocess `sglang generate`? Subprocess `generate` is one-shot:
it loads, generates, exits — you get one wall number that conflates load and
generation. `sglang serve` keeps the model resident across requests, which is
what we need for the lab-4.2 column schema.

Why not `from sglang import DiffusionEngine`? Not documented as public API
as of writing. The CLI (`generate`) and HTTP server (`serve`) are.

If your sglang version uses a different HTTP route than the default below,
override with: SGLANG_ENDPOINT=/your/path python benchmark_baseline.py

Run:
    python benchmark_baseline.py                    # sglang row only; reuses lab 4.2's diffusers numbers
    python benchmark_baseline.py --with-diffusers   # re-measure diffusers on this hardware too
"""

import argparse
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
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


WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# sglang serve HTTP defaults. Override either via env var if your sglang
# build uses a different route.
SGLANG_PORT = int(os.environ.get("SGLANG_PORT", "30000"))
SGLANG_ENDPOINT = os.environ.get("SGLANG_ENDPOINT", "/v1/videos/generations")


@dataclass
class RunResult:
    name: str
    load_secs: float = float("nan")
    first_secs: float = float("nan")
    second_secs: float = float("nan")
    # Either of these may be set depending on which row this is:
    peak_mem_bytes: int = 0            # torch.cuda.max_memory_allocated (diffusers)
    peak_mem_gb_smi: float = float("nan")  # nvidia-smi polling (sglang)
    skipped: bool = False
    note: str = ""


def fmt_secs(s: float) -> str:
    if not math.isfinite(s):
        return "      —"
    return f"{s:6.1f}s"


def fmt_mem(r: RunResult) -> str:
    if r.peak_mem_bytes:
        return f"{r.peak_mem_bytes / 1e9:5.2f} GB"
    if math.isfinite(r.peak_mem_gb_smi):
        return f"{r.peak_mem_gb_smi:5.2f} GB"
    return "      —"


# ---------- nvidia-smi peak-VRAM polling (for the sglang row) ----------------

class VramPoller(threading.Thread):
    """Poll `nvidia-smi --query-gpu=memory.used` in a background thread.

    Tracks the peak observed memory.used across all visible GPUs since
    `start()`. Use for the sglang row where we can't reach into the worker's
    torch state.
    """
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


# ---------- diffusers (in-process) row ---------------------------------------

def run_diffusers(args) -> RunResult:
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video
    import torch
    import gc

    label = "diffusers"
    print(f"\n=== {label} (in-process) ===")
    t0 = time.time()

    vae = AutoencoderKLWan.from_pretrained(
        WAN_REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(
        WAN_REPO, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.vae.enable_tiling()

    prompt_embeds, neg_embeds = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or "",
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        device=torch.device("cuda"),
    )
    # Same trick as lab 4.2: delete (don't .to('cpu')) so pipe.device falls
    # through to the transformer's cuda device.
    del pipe.text_encoder
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    load_secs = time.time() - t0
    print(f"  load:   {fmt_secs(load_secs)}")

    def one_call() -> tuple[float, list, int]:
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        generator = torch.Generator("cpu").manual_seed(args.seed)
        out = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            height=args.height, width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).frames[0]
        return time.time() - t, out, torch.cuda.max_memory_allocated()

    first_secs, frames, peak1 = one_call()
    print(f"  first:  {fmt_secs(first_secs)}")
    second_secs, _, peak2 = one_call()
    print(f"  second: {fmt_secs(second_secs)}")
    peak = max(peak1, peak2)
    print(f"  peak VRAM: {peak / 1e9:5.2f} GB")

    export_to_video(frames, "out_diffusers.mp4", fps=args.fps)

    del pipe, vae
    gc.collect()
    torch.cuda.empty_cache()

    return RunResult(name=label, load_secs=load_secs,
                     first_secs=first_secs, second_secs=second_secs,
                     peak_mem_bytes=peak)


# ---------- sglang serve + HTTP row -------------------------------------------

def find_sglang_bin() -> str | None:
    lab_dir = Path(__file__).resolve().parent
    candidate = lab_dir / ".venv-sglang" / "bin" / "sglang"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which("sglang")


def wait_for_server(port: int, proc: subprocess.Popen, timeout_secs: float) -> bool:
    """Return True once `sglang serve` accepts HTTP, False on timeout / death."""
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                try:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=2)
                    return True
                except urllib.error.HTTPError as e:
                    # No /health route; once TCP is up, assume ready.
                    if e.code in (404, 405):
                        return True
                except (urllib.error.URLError, socket.timeout):
                    pass
        except (ConnectionRefusedError, socket.timeout, OSError):
            pass
        time.sleep(1.0)
    return False


def post_generate(port: int, endpoint: str, payload: dict, timeout_secs: float) -> int:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
        resp.read()  # drain
        return resp.status


def run_sglang(args) -> RunResult:
    label = "sglang (defaults)"
    sglang_bin = find_sglang_bin()
    if sglang_bin is None:
        print(f"\n=== {label} ===  (skipped — sglang not found)")
        print("  set up the sglang venv first (from lab4.3/):")
        print("    python3 -m venv .venv-sglang")
        print("    source .venv-sglang/bin/activate")
        print("    pip install --upgrade pip uv")
        print("    uv pip install 'sglang[diffusion]' --prerelease=allow")
        return RunResult(name=label, skipped=True, note="sglang not found")

    print(f"\n=== {label} (sglang serve + HTTP) ===")
    cmd = [
        sglang_bin, "serve",
        "--model-path", WAN_REPO,
        "--port", str(SGLANG_PORT),
    ]
    print(f"  $ {' '.join(cmd)}")

    vram = VramPoller(); vram.start()

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT, text=True)
    try:
        t0 = time.time()
        if not wait_for_server(SGLANG_PORT, proc, timeout_secs=900):
            vram.stop_and_peak_gb()
            return RunResult(name=label, skipped=True,
                             note="server did not become ready in 15 min")
        load_secs = time.time() - t0
        print(f"  load:   {fmt_secs(load_secs)}  (sglang serve ready)")

        payload = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt or "",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance,
            "seed": args.seed,
        }

        try:
            t1 = time.time()
            post_generate(SGLANG_PORT, SGLANG_ENDPOINT, payload, timeout_secs=1800)
            first_secs = time.time() - t1
            print(f"  first:  {fmt_secs(first_secs)}")

            t2 = time.time()
            post_generate(SGLANG_PORT, SGLANG_ENDPOINT, payload, timeout_secs=1800)
            second_secs = time.time() - t2
            print(f"  second: {fmt_secs(second_secs)}")
        except urllib.error.HTTPError as e:
            body = e.read()[:500].decode(errors="replace")
            print(f"  HTTP {e.code} on {SGLANG_ENDPOINT}: {body}")
            print(f"  Override the route with: SGLANG_ENDPOINT=/your/path python benchmark_baseline.py")
            return RunResult(name=label, skipped=True, note=f"HTTP {e.code}")
        except (urllib.error.URLError, socket.timeout) as e:
            print(f"  request failed: {e}")
            return RunResult(name=label, skipped=True, note=f"request failed: {e}")

        peak_gb = vram.stop_and_peak_gb()
        print(f"  peak VRAM: {peak_gb:5.2f} GB  (nvidia-smi polling)")

        return RunResult(name=label, load_secs=load_secs,
                         first_secs=first_secs, second_secs=second_secs,
                         peak_mem_gb_smi=peak_gb)
    finally:
        vram.stop_and_peak_gb()  # idempotent
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


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
    p.add_argument("--with-diffusers", action="store_true",
                   help="Also run the diffusers row (default: skipped — use lab 4.2's number)")
    p.add_argument("--negative-prompt", default="")
    args = p.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs CUDA — see README.")

    print(f"prompt: {args.prompt!r}")
    print(f"shape:  {args.num_frames} frames @ {args.width}×{args.height},"
          f"  steps={args.steps},  cfg={args.guidance},  seed={args.seed}")

    results: list[RunResult] = []
    if args.with_diffusers:
        results.append(run_diffusers(args))
    results.append(run_sglang(args))

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    header = (f"{'config':<28} {'load':>8} {'first':>8} {'second':>8} "
              f"{'speedup':>9} {'peak VRAM':>10}")
    print(header)
    print("-" * len(header))

    # If --with-diffusers, the in-process row is the baseline. Otherwise compare
    # the sglang row against lab 4.2's measured single-4090 second-call number
    # (5.8 s load, 77 s first, 77 s second, 20.5 GB peak) and label the row so
    # the reader knows where the reference came from.
    LAB42_REF_SECOND = 77.0
    baseline = next((r for r in results
                     if r.name == "diffusers" and not r.skipped
                     and math.isfinite(r.second_secs)), None)
    if baseline is None:
        print(f"{'diffusers (lab 4.2 ref)':<28} {'5.8s':>8} {'77.0s':>8} {'77.0s':>8} "
              f"{'1.00×':>9} {'20.50 GB':>10}")

    for r in results:
        if r.skipped:
            print(f"{r.name:<28} {'(skipped: ' + r.note + ')':>62}")
            continue
        if baseline is not None and r is not baseline and math.isfinite(r.second_secs):
            speedup = f"{baseline.second_secs / r.second_secs:5.2f}×"
        elif baseline is not None and r is baseline:
            speedup = "1.00×"
        elif math.isfinite(r.second_secs):
            speedup = f"{LAB42_REF_SECOND / r.second_secs:5.2f}×"
        else:
            speedup = ""
        print(f"{r.name:<28} {fmt_secs(r.load_secs):>8} {fmt_secs(r.first_secs):>8} "
              f"{fmt_secs(r.second_secs):>8} {speedup:>9} {fmt_mem(r):>10}")
    print("=" * 92)


if __name__ == "__main__":
    main()

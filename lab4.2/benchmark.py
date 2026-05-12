"""Diffusers inference benchmark — one config per invocation.

Four orthogonal optimization knobs (`--compile`, `--aot`, `--sdpa-backend`,
`--offload`), each tunable independently. Pass `--help` for the full surface.
The README composes invocations into four named experiments:

  Exp 1  torch.compile modes        --compile {default,max-autotune-no-cudagraphs}
  Exp 2  AOT cold-start vs JIT      --aot save / --aot load
  Exp 3  SDPA backend selection     --sdpa-backend {flash,efficient,cudnn,math}
  Exp 4  Offload trade-off          --offload {model,sequential}

One invocation runs **two inference calls back-to-back** and times each.
That lets a single config tell two stories:

  first call  — includes JIT compile cost (for --compile), AOT load cost
                (for --aot load), or just steady-state cost (baseline).
  second call — steady-state per-call latency, no startup overhead.

The big AOT-vs-JIT story is the first-call column; the steady-state column
should match between compile-JIT and AOT-load.

Wall-clock reload between invocations is ~45 s (mostly VAE + text encoder
loading from HuggingFace cache). Accepted as the cost of orthogonal flags.

Run examples:
    python benchmark.py                                  # baseline
    python benchmark.py --compile default                # JIT compile
    python benchmark.py --aot save                       # one-time export
    python benchmark.py --aot load                       # cold-start = AOT load
    python benchmark.py --sdpa-backend flash
    python benchmark.py --offload model
"""

import argparse
import logging
import math
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo detected a call to a `functools\.lru_cache`-wrapped function.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", message=r".*Unable to import `torchao` Tensor objects.*")
warnings.filterwarnings("ignore", message=r".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=r".*sending unauthenticated requests.*")
# PyTorch pytree internals still call the deprecated isinstance(treespec, LeafSpec)
# pattern from inside copyreg/pickle paths; fires during AOT load.
warnings.filterwarnings("ignore", message=r".*`isinstance\(treespec, LeafSpec\)` is deprecated.*", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

import torch
torch.set_float32_matmul_precision("high")

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@dataclass
class Result:
    """One invocation's record. The two timing columns are the headline."""
    config: str
    model_load_secs: float = float("nan")
    first_call_secs: float = float("nan")
    second_call_secs: float = float("nan")
    peak_mem: int = 0
    out_path: str = ""
    note: str = ""


def fmt_secs(s: float) -> str:
    if not math.isfinite(s):
        return "      —"
    return f"{s:6.1f}s"


def fmt_mem(bytes_: int) -> str:
    if not bytes_:
        return "      —"
    return f"{bytes_ / 1e9:5.2f} GB"


# ---------- SDPA backend wrapper ---------------------------------------------

@contextmanager
def sdpa_backend(name: str):
    """Force a specific SDPA backend via `torch.nn.attention.sdpa_kernel`.

    `auto` is a no-op (PyTorch picks). Other choices restrict the dispatch.
    """
    if name == "auto":
        yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = {
        "flash":     SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn":     SDPBackend.CUDNN_ATTENTION,
        "math":      SDPBackend.MATH,
    }[name]
    with sdpa_kernel(backend):
        yield


# ---------- pipeline loading + optimization application ----------------------

def build_pipeline(args):
    """Load WanPipeline with the requested optimizations applied.

    Returns (pipe, prompt_embeds, neg_embeds, model_load_secs).
    """
    from diffusers import AutoencoderKLWan, WanPipeline

    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(WAN_REPO, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    pipe.vae.enable_tiling()

    # Pre-encode prompts on cuda. The pipeline doesn't need the text encoder
    # again after this — we pass prompt_embeds=... to pipe() directly.
    prompt_embeds, neg_embeds = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or "",
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        device=torch.device("cuda"),
    )

    # Free umT5-XXL (~11 GB) — but only when we're NOT using diffusers' own
    # offload helpers. Those helpers hook every module in the pipeline,
    # including the text encoder, and will offload it themselves; deleting it
    # would break their accounting and leave the transformer + VAE on cuda
    # full-time (muting the offload win).
    #
    # The setattr-None pattern (rather than .cpu()) is load-bearing: diffusers'
    # DiffusionPipeline.device walks self.config.keys() and returns the device
    # of the first nn.Module it finds. If text_encoder is on CPU, pipe.device
    # returns "cpu", prepare_latents allocates noise on CPU, the transformer
    # (cuda) Conv3d(cuda weight, cpu input) finds no cuDNN kernel, falls back
    # to aten::slow_conv3d_forward (CPU-only), and errors. Setting it to None
    # makes pipe.device skip past it to the transformer's cuda device.
    if args.offload not in ("model", "sequential"):
        import gc
        del pipe.text_encoder
        pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()

    # Apply --offload AFTER prompt encoding (offload helpers want a fully-
    # constructed pipeline). `enable_*_cpu_offload` re-hooks the model — and
    # the text encoder must still be present for the helper to move it to
    # CPU itself.
    if args.offload == "model":
        pipe.enable_model_cpu_offload()
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()

    # Apply --compile if requested. AOT load handled in main() (different shape).
    if args.compile and args.compile != "none":
        pipe.transformer = torch.compile(pipe.transformer, mode=args.compile)

    model_load_secs = time.time() - t0
    return pipe, prompt_embeds, neg_embeds, model_load_secs


def run_inference(pipe, prompt_embeds, neg_embeds, args, *, seed):
    """One inference call with the given seed. Returns (frames, secs)."""
    t0 = time.time()
    with sdpa_backend(args.sdpa_backend):
        output = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            height=args.height, width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).frames[0]
    return output, time.time() - t0


# ---------- AOT export + load ------------------------------------------------

def run_aot_save(args) -> Result:
    """Compile the transformer JIT, run inference to warm the cache, then
    export the warmed graph to a .pt2 package.

    The exact AOT API surface depends on the PyTorch version. We use
    `torch.export.export` + `torch._inductor.aoti_compile_and_package`
    (PyTorch ≥ 2.4 stable). Earlier / later versions may need adjustment;
    see https://pytorch.org/tutorials/recipes/torch_export_aoti_python.html.
    """
    from diffusers.utils import export_to_video

    print(f"\n=== --aot save  (→ {args.aot_path}) ===")
    args.compile = "default"  # AOT export goes through Inductor; need JIT first
    pipe, prompt_embeds, neg_embeds, model_load_secs = build_pipeline(args)
    print(f"  model load: {fmt_secs(model_load_secs)}")

    # Warm: run one inference to fully trace the compiled transformer.
    print("  warming transformer (one JIT compile pass)...")
    tw = time.time()
    _, _ = run_inference(pipe, prompt_embeds, neg_embeds, args, seed=0)
    print(f"  warmup:     {fmt_secs(time.time() - tw)}")

    # Now export the warmed transformer.
    print(f"  exporting to {args.aot_path}...")
    try:
        # Inspect one real call's args to build example_inputs (shape-stable).
        # The transformer's forward signature in current diffusers Wan:
        #   forward(hidden_states, timestep, encoder_hidden_states,
        #           encoder_attention_mask=None, return_dict=True)
        latent_shape = (
            1, pipe.transformer.config.in_channels,
            (args.num_frames - 1) // 4 + 1,
            args.height // 8, args.width // 8,
        )
        example_inputs = (
            torch.randn(latent_shape, dtype=torch.bfloat16, device="cuda"),
            torch.tensor([500.0], device="cuda"),
            prompt_embeds.to(torch.bfloat16),
        )
        exported = torch.export.export(
            pipe.transformer._orig_mod if hasattr(pipe.transformer, "_orig_mod")
            else pipe.transformer,
            args=example_inputs,
        )
        torch._inductor.aoti_compile_and_package(exported, package_path=args.aot_path)
        print(f"  saved {args.aot_path}")
    except Exception as e:
        print(f"  AOT export failed: {e}")
        print(f"  (the AOT API moves between PyTorch versions; check the docs)")
        return Result(config="aot-save", note=f"export failed: {e!s}",
                      model_load_secs=model_load_secs)

    return Result(
        config="aot-save",
        model_load_secs=model_load_secs,
        out_path=args.aot_path,
        note=f"exported to {args.aot_path}; use --aot load to benchmark",
    )


def run_aot_load(args) -> Result:
    """Load a previously-exported .pt2 and run two inferences. Compares
    first-call latency (the AOT win) against steady-state (matches JIT)."""
    from diffusers.utils import export_to_video

    print(f"\n=== --aot load  ({args.aot_path}) ===")
    # Reuse build_pipeline so the text_encoder=None handling stays in one
    # place. AOT load doesn't compile the transformer (the .pt2 already is
    # the compiled artifact), so force --compile=none for build_pipeline.
    args_for_build = argparse.Namespace(**{**vars(args), "compile": "none"})
    pipe, prompt_embeds, neg_embeds, model_load_secs = build_pipeline(args_for_build)

    # Load the AOT package and swap into the pipeline's transformer slot.
    # The loaded callable mimics transformer.__call__ but skips JIT compile.
    try:
        loaded = torch._inductor.aoti_load_package(args.aot_path)
    except Exception as e:
        print(f"  AOT load failed: {e}")
        return Result(config="aot-load", note=f"load failed: {e!s}")

    # Wrap so pipe.transformer(...) calls the loaded module.
    class _AOTWrapper:
        def __init__(self, loaded_module, ref_module):
            self._loaded = loaded_module
            self.config = ref_module.config
            self.dtype = ref_module.dtype
            self.device = ref_module.device
        def __call__(self, **kwargs):
            out = self._loaded(kwargs["hidden_states"], kwargs["timestep"],
                                kwargs["encoder_hidden_states"])
            return (out,) if kwargs.get("return_dict") is False else type(
                "Out", (), {"sample": out})()
    pipe.transformer = _AOTWrapper(loaded, pipe.transformer)
    print(f"  model load: {fmt_secs(model_load_secs)}  (includes .pt2 load)")

    torch.cuda.reset_peak_memory_stats()
    print("  first call (cold)...")
    _, first = run_inference(pipe, prompt_embeds, neg_embeds, args, seed=args.seed)
    print(f"  first:      {fmt_secs(first)}")

    print("  second call (warm)...")
    output, second = run_inference(pipe, prompt_embeds, neg_embeds, args, seed=args.seed + 1)
    print(f"  second:     {fmt_secs(second)}")

    peak_mem = torch.cuda.max_memory_allocated()
    out_path = "out_diffusers_aot_load.mp4"
    export_to_video(output, out_path, fps=args.fps)
    print(f"  peak VRAM:  {fmt_mem(peak_mem)}")
    print(f"  saved {out_path}")

    return Result(
        config="aot-load",
        model_load_secs=model_load_secs,
        first_call_secs=first,
        second_call_secs=second,
        peak_mem=peak_mem,
        out_path=out_path,
    )


# ---------- non-AOT path (one invocation, two inference calls) ---------------

def run_invocation(args) -> Result:
    from diffusers.utils import export_to_video

    config = _config_label(args)
    print(f"\n=== {config} ===")
    pipe, prompt_embeds, neg_embeds, model_load_secs = build_pipeline(args)
    print(f"  model load: {fmt_secs(model_load_secs)}")

    torch.cuda.reset_peak_memory_stats()

    # First call — includes torch.compile JIT cost (if --compile) or just
    # one cold inference (baseline). Either way, this is the "first request
    # served from a fresh process" number.
    print("  first call (cold)...")
    _, first = run_inference(pipe, prompt_embeds, neg_embeds, args, seed=args.seed)
    print(f"  first:      {fmt_secs(first)}")

    # Second call — steady state.
    print("  second call (warm)...")
    output, second = run_inference(pipe, prompt_embeds, neg_embeds, args, seed=args.seed + 1)
    print(f"  second:     {fmt_secs(second)}")

    peak_mem = torch.cuda.max_memory_allocated()
    out_path = _output_path(args)
    export_to_video(output, out_path, fps=args.fps)
    print(f"  peak VRAM:  {fmt_mem(peak_mem)}")
    print(f"  saved {out_path}")

    return Result(
        config=config,
        model_load_secs=model_load_secs,
        first_call_secs=first,
        second_call_secs=second,
        peak_mem=peak_mem,
        out_path=out_path,
    )


def _config_label(args) -> str:
    """Short human-readable label summarizing the optimization combo."""
    parts = []
    if args.compile and args.compile != "none":
        parts.append(f"compile={args.compile}")
    if args.sdpa_backend != "auto":
        parts.append(f"sdpa={args.sdpa_backend}")
    if args.offload != "none":
        parts.append(f"offload={args.offload}")
    return "diffusers" + (" (" + ", ".join(parts) + ")" if parts else " (baseline)")


def _output_path(args) -> str:
    parts = ["out_diffusers"]
    if args.compile and args.compile != "none":
        parts.append(f"compile-{args.compile}")
    if args.sdpa_backend != "auto":
        parts.append(f"sdpa-{args.sdpa_backend}")
    if args.offload != "none":
        parts.append(f"offload-{args.offload}")
    return "_".join(parts) + ".mp4"


# ---------- main --------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Generation knobs.
    p.add_argument("--prompt",     default="a fluffy red panda eating bamboo on a tree branch")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--height",     type=int,   default=480)
    p.add_argument("--width",      type=int,   default=832)
    p.add_argument("--num-frames", type=int,   default=49)
    p.add_argument("--steps",      type=int,   default=30)
    p.add_argument("--guidance",   type=float, default=5.0)
    p.add_argument("--fps",        type=int,   default=16)
    p.add_argument("--seed",       type=int,   default=42)

    # Four orthogonal optimization knobs.
    p.add_argument("--compile",
                   choices=["none", "default", "reduce-overhead",
                            "max-autotune", "max-autotune-no-cudagraphs"],
                   default="none",
                   help="torch.compile mode for the transformer. "
                        "Note: reduce-overhead and max-autotune enable CUDA "
                        "Graphs, which alias buffers across WanPipeline's "
                        "two-pass CFG and raise RuntimeError. Use "
                        "max-autotune-no-cudagraphs for the autotune win.")
    p.add_argument("--aot", choices=["none", "save", "load"], default="none",
                   help="AOT export (save) or load a previously-exported .pt2")
    p.add_argument("--aot-path", default="wan_transformer.pt2",
                   help="Path for AOT save/load")
    p.add_argument("--sdpa-backend", choices=["auto", "flash", "efficient", "cudnn", "math"],
                   default="auto",
                   help="Force a specific SDPA backend for attention")
    p.add_argument("--offload", choices=["none", "model", "sequential"], default="none",
                   help="diffusers CPU-offload mode")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs CUDA — see README compute notes.")

    print(f"prompt: {args.prompt!r}")
    print(f"shape:  {args.num_frames} frames @ {args.width}×{args.height},  steps={args.steps},  cfg={args.guidance},  seed={args.seed}")

    # Dispatch on --aot first since it's the only flag that changes the
    # whole flow (save = export and exit; load = swap transformer for .pt2).
    if args.aot == "save":
        result = run_aot_save(args)
    elif args.aot == "load":
        result = run_aot_load(args)
    else:
        result = run_invocation(args)

    print("\n" + "=" * 84)
    print("RESULT")
    print("=" * 84)
    header = f"{'config':<44} {'load':>8} {'first':>8} {'second':>8} {'peak VRAM':>10}"
    print(header)
    print("-" * len(header))
    print(
        f"{result.config:<44} "
        f"{fmt_secs(result.model_load_secs):>8} "
        f"{fmt_secs(result.first_call_secs):>8} "
        f"{fmt_secs(result.second_call_secs):>8} "
        f"{fmt_mem(result.peak_mem):>10}"
    )
    if result.note:
        print(f"\n  note: {result.note}")
    if result.out_path:
        print(f"\n  output: {result.out_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()

"""Full SFT (supervised fine-tuning) of Wan-2.1 T2V-1.3B on a video-caption set.

This lab's hands-on centerpiece. *Every* base parameter is trainable — no
LoRA adapter, no parameter-efficient tricks. The trade-off vs lab 4.4:

                     LoRA (lab 4.4)        Full SFT (this lab)
                     ─────────────         ──────────────────
    trainable params  ~5 M (rank 16)        ~1.3 B
    optimizer state   ~40 MB                ~2.6 GB (8-bit Adam) / 10.4 GB (fp32)
    artifact size     ~140 MB               ~2.6 GB
    overfits at       ~5k steps             ~500 steps on a tiny dataset
    quality ceiling   rank-r capped         unbounded (in principle)

The flow-matching forward pass is identical to lab 4.4. What's new is
the memory machinery that makes 1.3B-parameter SFT fit a 4090:

  1. **8-bit AdamW** (bitsandbytes) — Adam's m and v moments stored at int8
     instead of fp32. Saves ~7.8 GB on a 1.3B-param model. Quality cost
     is <1% in practice. Required for the 4090 path.
  2. **Gradient checkpointing** — recompute forward activations on the
     backward pass instead of caching them. ~30% wall-clock hit; gigabytes
     of activation memory recovered. Required.
  3. **bf16 throughout** — params, grads, and forward activations all in
     bf16. No fp32 master weights (some production setups keep them for
     stability; we skip for VRAM headroom).

Compute budget: this fits on a single 4090 (24 GB) at 256×256 × 17 frames,
batch 1, grad-accum 8. Wall clock ~10–14 hours for 2000 steps (vs ~6–8 for
the LoRA in lab 4.4 — full SFT is slower per step because every parameter
gets a gradient). 4× 4090 with DDP cuts that ~4× via accelerate.

Run:
    accelerate launch train_sft.py \\
        --data-root data/ \\
        --output-dir runs/my-sft/ \\
        --steps 2000
"""

import argparse
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKLWan, WanPipeline
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5EncoderModel

from data import VideoCaptionDataset, collate

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",       required=True)
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--steps",           type=int,   default=2000)
    p.add_argument("--batch-size",      type=int,   default=1)
    p.add_argument("--grad-accum",      type=int,   default=8)
    # SFT learning rate is ~10× *lower* than LoRA's. LoRA params are
    # randomly initialized and tiny so they tolerate a hot LR; SFT
    # updates the pretrained weights, which need gentle nudges or you
    # destroy what the base learned during pretraining.
    p.add_argument("--lr",              type=float, default=1e-5)
    p.add_argument("--n-frames",        type=int,   default=17)
    p.add_argument("--height",          type=int,   default=256)
    p.add_argument("--width",           type=int,   default=256)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Required to fit a 4090. Disable on H100/A100 if you have headroom.")
    p.add_argument("--save-every",      type=int,   default=500)
    p.add_argument("--seed",            type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum,
    )
    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Frozen pretrained components: VAE + text encoder. Same as lab 4.4.
    accelerator.print("loading Wan-VAE...")
    vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval()
    vae.requires_grad_(False)

    accelerator.print("loading umT5 text encoder...")
    tokenizer = AutoTokenizer.from_pretrained(WAN_REPO, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        WAN_REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    # ── The transformer is the trainable part — everything unfrozen. This is
    # the only structural difference vs lab 4.4's LoRA loop.
    accelerator.print("loading WAN transformer (fully trainable)...")
    pipe = WanPipeline.from_pretrained(WAN_REPO, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer
    transformer.requires_grad_(True)
    transformer.train()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    n_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    accelerator.print(f"trainable params: {n_trainable / 1e9:.2f}B  (full transformer)")

    # ── 8-bit AdamW. The line below is the difference between "fits a 4090"
    # and "OOMs on a 4090". For 1.3B params, fp32 AdamW state is ~10 GB;
    # 8-bit AdamW state is ~2.6 GB.
    optim = bnb.optim.AdamW8bit(
        transformer.parameters(),
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999),
    )

    # ── Data.
    ds = VideoCaptionDataset(
        args.data_root, n_frames=args.n_frames,
        height=args.height, width=args.width,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, drop_last=True,
    )

    # ── Hand off to accelerate (DDP across GPUs, mixed precision, grad accum).
    transformer, optim, loader = accelerator.prepare(transformer, optim, loader)
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)

    # ── Training loop. Identical math to lab 4.4 — only the optimizer's reach
    # is different (every parameter, not just the adapter).
    step = 0
    while step < args.steps:
        for videos, captions in loader:
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    z_0 = vae.encode(videos.to(torch.bfloat16)).latent_dist.sample()
                    z_0 = z_0 * vae.config.scaling_factor

                    toks = tokenizer(
                        captions, padding="max_length", truncation=True,
                        max_length=512, return_tensors="pt",
                    ).to(accelerator.device)
                    text_embeds = text_encoder(**toks).last_hidden_state

                # Flow matching forward: x_t = (1-t)·x_0 + t·noise; target = noise - x_0.
                noise = torch.randn_like(z_0)
                t = torch.rand(z_0.size(0), device=z_0.device, dtype=z_0.dtype)
                t_b = t.view(-1, *([1] * (z_0.dim() - 1)))
                z_t = (1 - t_b) * z_0 + t_b * noise
                target_v = noise - z_0

                pred = transformer(
                    hidden_states=z_t,
                    timestep=t * 1000,
                    encoder_hidden_states=text_embeds,
                    encoder_attention_mask=toks.attention_mask,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(pred.float(), target_v.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                if step % 50 == 0 or step == 1:
                    accelerator.print(f"step {step:5d}  loss {loss.item():.4f}")
                if step % args.save_every == 0 or step == args.steps:
                    save_transformer(transformer, args.output_dir, step, accelerator)
                if step >= args.steps:
                    break

    accelerator.print("done.")


def save_transformer(transformer, output_dir, step, accelerator):
    """Save the *full* transformer state dict.

    Unlike lab 4.4's LoRA save (~140 MB), this is the entire transformer
    (~2.6 GB at bf16). That's the whole price of full SFT — your
    distribution artifact is much heavier.
    """
    if not accelerator.is_main_process:
        return
    transformer = accelerator.unwrap_model(transformer)
    state = {k: v.contiguous() for k, v in transformer.state_dict().items()}
    out = Path(output_dir) / f"transformer_step{step:05d}.safetensors"
    save_file(state, str(out))
    bytes_total = sum(t.numel() * t.element_size() for t in state.values())
    accelerator.print(f"saved {out}  ({len(state)} tensors, {bytes_total / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()

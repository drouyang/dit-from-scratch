"""LoRA fine-tune Wan-2.1 T2V-1.3B on a tiny video-caption dataset.

This is the hands-on centerpiece of lab 4.2 — what every other post-training
flavor (full SFT, DPO, distillation) is contrasted against in the README.

The high-level flow is identical to lab 3.2's text-to-image training loop;
the only differences are dimensional (3D VAE, 3D patchify, 3D RoPE inside
WAN — all transparent to us here, we just call the diffusers wrapper) and
parametric (we train ~50–200 MB of LoRA adapter weights instead of all
1.3B base parameters).

End-to-end:
    video (B, 3, T, H, W) ──Wan-VAE.encode──► z_0 (B, C, T', H', W')
    caption (str)         ──umT5.encode    ──► (text_embeds, mask)
    sample t ∈ [0, 1]
    z_t = (1 - t) · z_0  +  t · noise               (flow matching forward)
    pred = transformer_with_lora(z_t, t, text_embeds, mask)
    loss = MSE(pred, noise - z_0)                   (velocity target)
    loss.backward()
    optimizer.step()    # only LoRA params get updated; base stays frozen

Read the README first — it covers what LoRA actually is, which target
modules to wrap, and how to choose rank / alpha / lr.

Compute budget: defaults are sized for a single 4090 (24 GB) with bf16,
gradient checkpointing, 17 frames @ 256×256, batch 1, grad-accum 8.
The same config scales linearly across multiple 4090s via accelerate's
DDP — 4× 4090 cuts wall-clock by ~4×. See the README's "Hardware"
section to scale up to higher resolution / more frames on H100 / A100.

Run:
    accelerate launch train_lora.py \\
        --data-root data/ \\
        --output-dir runs/my-style/ \\
        --rank 16 \\
        --steps 2000
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline
from peft import LoraConfig, get_peft_model_state_dict
from peft.utils import set_peft_model_state_dict
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, UMT5EncoderModel

from data import VideoCaptionDataset, collate

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# WAN's transformer is a stack of attention + FFN blocks. The canonical
# LoRA targets are the linear layers inside attention (q, k, v projections
# + the output proj) and inside the MLP (gate/up/down projections).
# Adapting these touches the most "model behavior" per added parameter.
LORA_TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",     # attention projections
    "ffn.net.0.proj", "ffn.net.2",          # MLP gate/up + down projections
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",       required=True)
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--steps",           type=int,   default=2000)
    p.add_argument("--batch-size",      type=int,   default=1)
    p.add_argument("--grad-accum",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--rank",            type=int,   default=16)
    p.add_argument("--alpha",           type=int,   default=16,
                   help="LoRA scaling factor; effective scale = alpha/rank")
    # Defaults sized for a single 4090 (24 GB) with gradient checkpointing.
    # 17 frames = 4*4 + 1 — matches Wan-VAE's 4× temporal compression pattern.
    p.add_argument("--n-frames",        type=int,   default=17)
    p.add_argument("--height",          type=int,   default=256)
    p.add_argument("--width",           type=int,   default=256)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Trade recompute for VRAM; required to fit a 4090. "
                        "Disable with --no-gradient-checkpointing on H100/A100.")
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

    # ── Load WAN pieces individually (vs WanPipeline.from_pretrained which
    # would also pull a scheduler we don't need at training time). The VAE
    # and text encoder stay frozen — same pattern as lab 3.2.
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

    accelerator.print("loading WAN transformer (the trainable part)...")
    pipe = WanPipeline.from_pretrained(WAN_REPO, torch_dtype=torch.bfloat16)
    transformer = pipe.transformer
    # Important: freeze the base. peft will add and unfreeze the LoRA params next.
    transformer.requires_grad_(False)

    # ── Apply LoRA. peft wraps every target module with LoRA(A, B) such that
    # forward = base(x) + (alpha/rank) * B @ A @ x. Only A and B get gradients.
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=LORA_TARGET_MODULES,
        init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_config)
    transformer.train()

    # Gradient checkpointing trades activation memory for recompute. With
    # PEFT-wrapped models, also flip on input_require_grads so the recompute
    # graph extends through the (frozen) base — without this, autograd thinks
    # the base has nothing to backprop and the LoRA grads come back zero.
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        transformer.enable_input_require_grads()

    # Sanity: only LoRA params should be trainable.
    n_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in transformer.parameters())
    accelerator.print(
        f"trainable params: {n_trainable / 1e6:.2f}M  "
        f"({100 * n_trainable / n_total:.2f}% of {n_total / 1e9:.2f}B base)"
    )

    # ── Optimizer sees only LoRA params (everything else has requires_grad=False).
    optim = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
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

    # ── Hand off to accelerate (handles device placement, mixed precision,
    # and gradient accumulation under the hood).
    transformer, optim, loader = accelerator.prepare(transformer, optim, loader)
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)

    # ── Training loop.
    step = 0
    while step < args.steps:
        for videos, captions in loader:
            with accelerator.accumulate(transformer):
                # Encode video → 3D latent.
                with torch.no_grad():
                    z_0 = vae.encode(videos.to(torch.bfloat16)).latent_dist.sample()
                    z_0 = z_0 * vae.config.scaling_factor

                # Encode caption → per-token text embeddings + attention mask.
                with torch.no_grad():
                    toks = tokenizer(
                        captions, padding="max_length", truncation=True,
                        max_length=512, return_tensors="pt",
                    ).to(accelerator.device)
                    text_embeds = text_encoder(**toks).last_hidden_state

                # Flow-matching forward: x_t = (1-t)·x_0 + t·noise; target = noise - x_0.
                noise = torch.randn_like(z_0)
                t = torch.rand(z_0.size(0), device=z_0.device, dtype=z_0.dtype)
                t_b = t.view(-1, *([1] * (z_0.dim() - 1)))
                z_t = (1 - t_b) * z_0 + t_b * noise
                target_v = noise - z_0

                # WAN's transformer forward expects timestep ∈ [0, 1000].
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
                    accelerator.clip_grad_norm_(
                        [p for p in transformer.parameters() if p.requires_grad], 1.0,
                    )
                optim.step()
                optim.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                if step % 50 == 0 or step == 1:
                    accelerator.print(f"step {step:5d}  loss {loss.item():.4f}")
                if step % args.save_every == 0 or step == args.steps:
                    save_lora(transformer, args.output_dir, step, accelerator)
                if step >= args.steps:
                    break

    accelerator.print("done.")


def save_lora(transformer, output_dir, step, accelerator):
    """Save *only* the LoRA adapter weights as safetensors.

    A LoRA file is much smaller than the base model — typically 50–200 MB
    for rank 16, vs ~5 GB for the full bf16 transformer. That's the whole
    point: shareable, swappable, stackable.
    """
    if not accelerator.is_main_process:
        return
    transformer = accelerator.unwrap_model(transformer)
    state = get_peft_model_state_dict(transformer)
    out = Path(output_dir) / f"lora_step{step:05d}.safetensors"
    save_file(state, str(out))
    accelerator.print(f"saved {out}  ({len(state)} tensors, "
                      f"{sum(t.numel() for t in state.values()) * 2 / 1e6:.1f} MB at bf16)")


if __name__ == "__main__":
    main()

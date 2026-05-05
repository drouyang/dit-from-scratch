"""Sample digits from a trained DiT.

Run:
    python sample.py                                   # 8 samples per class, cfg=4
    python sample.py --cfg-scale 2.0  --steps 20       # weaker CFG, fewer steps
    python sample.py --class-id 7 --n-per-class 16     # only sevens
"""

import argparse

import torch
from torchvision.utils import save_image

from data import IMAGE_SIZE, IN_CHANNELS, NUM_CLASSES, denormalize
from dit import DiT
from flow import fm_euler_sample


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         default="dit.pt")
    p.add_argument("--steps",        type=int,   default=50)
    p.add_argument("--cfg-scale",    type=float, default=4.0,
                   help="1.0 = no CFG; > 1.0 = stronger conditioning")
    p.add_argument("--class-id",     type=int,   default=None,
                   help="if set, sample only this class. Otherwise round-robin all 10")
    p.add_argument("--n-per-class",  type=int,   default=8)
    p.add_argument("--save",         default="samples.png")
    p.add_argument("--seed",         type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DiT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if args.class_id is not None:
        classes = torch.full((args.n_per_class,), args.class_id,
                             device=device, dtype=torch.long)
    else:
        classes = torch.arange(NUM_CLASSES, device=device).repeat_interleave(args.n_per_class)

    x = fm_euler_sample(
        model, classes.size(0), n_steps=args.steps,
        shape=(IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
        classes=classes, cfg_scale=args.cfg_scale, device=device,
    )
    x = denormalize(x).clamp(0, 1).cpu()
    save_image(x, args.save, nrow=args.n_per_class, padding=2)
    print(f"saved {args.save}  "
          f"({tuple(x.shape)}, cfg={args.cfg_scale}, steps={args.steps})")


if __name__ == "__main__":
    main()

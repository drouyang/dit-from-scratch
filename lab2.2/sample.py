"""Sample from a trained model. Picks the right sampler based on the
checkpoint's saved paradigm (flow matching → Euler ODE; DDPM → ancestral).

Run:
    python sample.py                                    # 8 samples per class
    python sample.py --ckpt model_ddpm.pt
    python sample.py --cfg-scale 3.0  --steps 10
    python sample.py --class-id 3 --n-per-class 32
"""

import argparse

import torch

from data import NUM_CLASSES
from flow import fm_euler_sample, ddpm_sample, DDPMSchedule
from mlp import TimeMLP


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         default="model_fm.pt")
    p.add_argument("--steps",        type=int,   default=50)
    p.add_argument("--cfg-scale",    type=float, default=1.0,
                   help="1.0 = no CFG; > 1.0 = stronger conditioning")
    p.add_argument("--class-id",     type=int,   default=None,
                   help="if set, sample only this class. Otherwise round-robin all 8")
    p.add_argument("--n-per-class",  type=int,   default=8)
    p.add_argument("--seed",         type=int,   default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TimeMLP(num_classes=ckpt["config"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if args.class_id is not None:
        classes = torch.full((args.n_per_class,), args.class_id, device=device, dtype=torch.long)
    else:
        classes = torch.arange(NUM_CLASSES, device=device).repeat_interleave(args.n_per_class)

    paradigm = ckpt["paradigm"]
    if paradigm == "fm":
        x = fm_euler_sample(
            model, classes.size(0), n_steps=args.steps, dim=2,
            classes=classes, cfg_scale=args.cfg_scale, device=device,
        )
    else:
        T = ckpt.get("ddpm_T", 100)
        sched = DDPMSchedule(T=T).to(device)
        x = ddpm_sample(
            model, sched, classes.size(0), dim=2,
            classes=classes, cfg_scale=args.cfg_scale, device=device,
        )

    print(f"paradigm={paradigm}  steps={args.steps}  cfg={args.cfg_scale}")
    print("class | (x, y)")
    for c, p in zip(classes.cpu().tolist(), x.cpu().tolist()):
        print(f"  {c}   | ({p[0]:+.3f}, {p[1]:+.3f})")


if __name__ == "__main__":
    main()

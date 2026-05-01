"""Tiny MLP for 2-D diffusion / flow-matching toys.

Inputs:
    x:  (B, 2) — current point
    t:  (B,)   — time in [0, 1]
    c:  (B,)   — class index in {0, ..., num_classes}, where `num_classes` is
                 the *null* class used for unconditional training (the label-
                 dropout slot for CFG).

Output:
    v:  (B, 2) — predicted velocity (flow matching) or noise epsilon (DDPM)

The network doesn't know which target it's predicting — that's decided by
the loss in `train.py`. Same architecture, different supervision signal.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_time_embed(t, dim=128):
    """Sinusoidal embedding for continuous t in [0, 1]. (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, num_classes, hidden=256, time_dim=128, class_dim=128):
        super().__init__()
        # +1 for the null class used by CFG label-dropout.
        self.num_classes = num_classes
        self.null_class = num_classes
        self.class_embed = nn.Embedding(num_classes + 1, class_dim)

        self.t_proj = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.x_proj = nn.Linear(2, hidden)

        self.fc = nn.Sequential(
            nn.Linear(hidden + time_dim + class_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.time_dim = time_dim

    def forward(self, x, t, c):
        t_emb = self.t_proj(sinusoidal_time_embed(t, self.time_dim))
        x_emb = self.x_proj(x)
        c_emb = self.class_embed(c)
        h = torch.cat([x_emb, t_emb, c_emb], dim=-1)
        return self.fc(h)

"""Flow matching forward process and Euler ODE sampler — adapted from lab 2.2/3.1.

The forward process and sampler are paradigm-agnostic: they don't care whether
the model conditions on a class label or a text embedding. Only the function
signature of the model call changes.
"""

import torch


def fm_q_sample(x_0, t, noise=None):
    """Forward process: x_t = (1 - t) * x_0 + t * noise.

    Identical to lab 2.2 / 3.1. Returns (x_t, noise, target_velocity).
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    t = t.view(-1, *([1] * (x_0.dim() - 1)))
    x_t = (1 - t) * x_0 + t * noise
    target_v = noise - x_0
    return x_t, noise, target_v


@torch.no_grad()
def fm_euler_sample(model, n_samples, n_steps, shape, *,
                    text_tokens, text_pooled, text_mask,
                    cfg_scale=1.0, device="cpu", return_trajectory=False):
    """Sample by Euler ODE integration with optional CFG.

    Args:
        model: DiT (text-conditioned)
        n_samples: batch size
        n_steps: number of Euler steps
        shape: latent shape per sample, e.g., (4, 8, 8) for SD-VAE @ 64x64
        text_tokens, text_pooled, text_mask: from CLIPTextEncoder.encode()
        cfg_scale: 1.0 = pure conditional; > 1.0 = stronger guidance
    """
    x = torch.randn(n_samples, *shape, device=device)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    traj = [x.clone()] if return_trajectory else None

    no_null = torch.zeros(n_samples, dtype=torch.bool, device=device)
    yes_null = torch.ones(n_samples, dtype=torch.bool, device=device)

    for i in range(n_steps):
        t = torch.full((n_samples,), ts[i].item(), device=device)
        if cfg_scale == 1.0:
            v = model(x, t, text_tokens, text_pooled, text_mask, force_null=no_null)
        else:
            v_cond = model(x, t, text_tokens, text_pooled, text_mask, force_null=no_null)
            v_uncond = model(x, t, text_tokens, text_pooled, text_mask, force_null=yes_null)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        dt = ts[i + 1] - ts[i]
        x = x + dt * v
        if return_trajectory:
            traj.append(x.clone())

    if return_trajectory:
        return x, torch.stack(traj)
    return x

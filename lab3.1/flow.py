"""Flow matching forward process and Euler ODE sampler.

Reused verbatim from lab 2.2 with one change: `fm_euler_sample` now takes a
`shape` tuple instead of a scalar `dim`, so the sampler works on (B, C, H, W)
images rather than (B, 2) toy points. Everything else — the forward
interpolation, the velocity supervision target, the CFG extrapolation — is
identical to the 2-D toy.

That's the whole promise of the curriculum: you wrote the loss and the
sampler once in lab 2.2, and you don't touch them again here. The only thing
that changes between labs 2.2, 3.1, 3.2 is what `model(x, t, c)` is — an MLP
on 2-D points (2.2), a DiT on pixel images (here), or a DiT on VAE latents
(3.2). The training loop and the sampler stay the same line for line.
"""

import torch


def fm_q_sample(x_0, t, noise=None):
    """Forward process: sample x_t along the straight line from data to noise.

        x_t  =  (1 - t) * x_0  +  t * noise

    The velocity field at every t is the *constant* `noise - x_0` (rectified
    flow): the path is straight, so the direction never changes.

    Works on any shape (B, ...). For the 2-D toy in lab 2.2 that was (B, 2);
    here it's (B, C, H, W). The math is the same — only the shape varies.

    Args:
        x_0:   (B, ...) data point
        t:     (B,) time in [0, 1]
        noise: (B, ...) optional; sampled from N(0, I) if not provided

    Returns:
        x_t:      (B, ...) the noisy sample at time t
        noise:    (B, ...) the noise that was used
        target_v: (B, ...) the supervision target = noise - x_0
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    t = t.view(-1, *([1] * (x_0.dim() - 1)))
    x_t = (1 - t) * x_0 + t * noise
    target_v = noise - x_0
    return x_t, noise, target_v


@torch.no_grad()
def fm_euler_sample(model, n_samples, n_steps, shape, classes,
                    cfg_scale=1.0, device="cpu", return_trajectory=False):
    """Sample by Euler integration of the learned ODE  dx/dt = v(x, t, c).

    Starts from x ~ N(0, I) at t=1 and integrates backward to t=0.

    Args:
        model:      predicts velocity v(x, t, c)
        n_samples:  batch size
        n_steps:    number of Euler steps. FM works in very few steps because
                    the path is straight; production-ish quality is reached
                    with N=20–50 here.
        shape:      tuple, the shape of one sample. (C, H, W) for images,
                    (D,) for the 2-D toy.
        classes:    (n_samples,) class indices in [0, num_classes)
        cfg_scale:  1.0 = no CFG; > 1.0 = stronger conditioning
        return_trajectory: if True, also return all intermediate x's
    """
    x = torch.randn(n_samples, *shape, device=device)
    null = torch.full_like(classes, model.null_class)

    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    traj = [x.clone()] if return_trajectory else None

    for i in range(n_steps):
        t = torch.full((n_samples,), ts[i].item(), device=device)
        if cfg_scale == 1.0:
            v = model(x, t, classes)
        else:
            # CFG: extrapolate between conditional and unconditional.
            # Two model evaluations per step, same as lab 2.2.
            v_cond = model(x, t, classes)
            v_uncond = model(x, t, null)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        dt = ts[i + 1] - ts[i]  # negative (going from 1 to 0)
        x = x + dt * v
        if return_trajectory:
            traj.append(x.clone())

    if return_trajectory:
        return x, torch.stack(traj)
    return x

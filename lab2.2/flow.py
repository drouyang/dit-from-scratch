"""Forward processes and samplers for Flow Matching and (briefly) DDPM.

Flow Matching = production. Predict velocity along a straight line between
data and noise; sample by Euler ODE integration.

DDPM = historical. Predict noise added through a Gaussian Markov chain;
sample by ancestral steps. Included here for direct comparison — the same
MLP can be trained with either supervision signal, see the contrast in
sample quality vs step count.
"""

import torch


# ─── Flow Matching (Rectified Flow) ─────────────────────────────────────────


def fm_q_sample(x_0, t, noise=None):
    """Forward process: sample x_t along the straight line from data to noise.

        x_t  =  (1 - t) * x_0  +  t * noise

    The velocity field at every t is the *constant* `noise - x_0` (rectified
    flow): the path is straight, so the direction never changes.

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
def fm_euler_sample(model, n_samples, n_steps, dim, classes,
                    cfg_scale=1.0, device="cpu", return_trajectory=False):
    """Sample by Euler integration of the learned ODE  dx/dt = v(x, t, c).

    Starts from x ~ N(0, I) at t=1 and integrates backward to t=0.

    Args:
        model:      predicts velocity v(x, t, c)
        n_samples:  batch size
        n_steps:    number of Euler steps (production: 10–50; FM works fine
                    with very few steps because the path is straight)
        dim:        dimensionality of x (2 for the toy)
        classes:    (n_samples,) class indices in [0, num_classes)
        cfg_scale:  1.0 = no CFG; > 1.0 = stronger conditioning
        return_trajectory: if True, return all intermediate x's
    """
    x = torch.randn(n_samples, dim, device=device)
    null = torch.full_like(classes, model.null_class)

    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    traj = [x.clone()] if return_trajectory else None

    for i in range(n_steps):
        t = torch.full((n_samples,), ts[i].item(), device=device)
        if cfg_scale == 1.0:
            v = model(x, t, classes)
        else:
            # CFG: extrapolate between conditional and unconditional.
            v_cond = model(x, t, classes)
            v_uncond = model(x, t, null)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        dt = ts[i + 1] - ts[i]  # negative (going from 1 to 0)
        x = x + dt * v
        if return_trajectory:
            traj.append(x.clone())

    if return_trajectory:
        return x, torch.stack(traj)  # (n_steps+1, n_samples, dim)
    return x


# ─── DDPM (for comparison) ──────────────────────────────────────────────────


def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear β schedule. (Cosine is marginally better; linear is fine here.)"""
    return torch.linspace(beta_start, beta_end, T)


class DDPMSchedule:
    """Pre-computes the DDPM noise schedule constants."""

    def __init__(self, T=100):
        self.T = T
        betas = make_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = alpha_bars.sqrt()
        self.sqrt_one_minus_alpha_bars = (1 - alpha_bars).sqrt()

    def to(self, device):
        for k in ("betas", "alphas", "alpha_bars",
                  "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"):
            setattr(self, k, getattr(self, k).to(device))
        return self


def ddpm_q_sample(x_0, t, schedule, noise=None):
    """Forward process for DDPM:

        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * noise

    Closed-form (no need to step through the Markov chain explicitly).
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    t_long = t.long()
    sqrt_ab = schedule.sqrt_alpha_bars[t_long].view(-1, *([1] * (x_0.dim() - 1)))
    sqrt_1_ab = schedule.sqrt_one_minus_alpha_bars[t_long].view(-1, *([1] * (x_0.dim() - 1)))
    x_t = sqrt_ab * x_0 + sqrt_1_ab * noise
    return x_t, noise


@torch.no_grad()
def ddpm_sample(model, schedule, n_samples, dim, classes,
                cfg_scale=1.0, device="cpu"):
    """Ancestral DDPM sampling: T denoising steps."""
    T = schedule.T
    null = torch.full_like(classes, model.null_class)
    x = torch.randn(n_samples, dim, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        # Model expects continuous t in [0, 1].
        t_norm = t_batch.float() / T

        if cfg_scale == 1.0:
            eps = model(x, t_norm, classes)
        else:
            eps_cond = model(x, t_norm, classes)
            eps_uncond = model(x, t_norm, null)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        alpha_bar_t = schedule.alpha_bars[t]
        coef = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
        mean = (x - coef * eps) / alpha_t.sqrt()
        if t > 0:
            sigma = beta_t.sqrt()
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean

    return x

"""Conditional flow-matching loss (linear / "rectified flow" / CondOT path).

Path:     x_t = (1 - t) * x0 + t * x1,     x0 ~ N(0, I) (noise),  x1 ~ data
Target:   u_t = d x_t / dt
Loss:     E_{t, x0, x1} || v_theta(x_t, t) - u_t ||^2
"""
import torch


def interpolate(x0, x1, t):
    """x_t on the straight path. t has shape (B,); broadcast it over the other dims of x."""
    t = t.view(-1, *([1] * (x0.dim() - 1)))  # broadcast
    x_t = (1-t)*x0 + t*x1
    return x_t


def target_velocity(x0, x1):
    """u_t for the straight path."""
    u_t = x1 - x0
    return u_t


def fm_loss(model, x1, generator=None):
    """Monte-Carlo FM loss for a batch of data x1 (B, ...).

    Steps: sample x0 ~ N(0,I) like x1, sample t ~ U[0,1] of shape (B,),
    build x_t and u_t with the helpers above, return the mean squared error (scalar).
    Use `generator` for torch.randn / torch.rand if given (reproducibility).
    """
    B = x1.shape[0]
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator) # normal law
    t = torch.rand(B, device=x1.device, dtype=x1.dtype, generator=generator) # uniform law

    x_t = interpolate(x0, x1, t)
    u_t = target_velocity(x0, x1)

    error = torch.nn.functional.mse_loss(model(x_t, t), u_t)
    return error
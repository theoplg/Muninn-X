"""Conditional flow-matching loss (linear / "rectified flow" / CondOT path).

>>> YOU IMPLEMENT THIS (GUIDE.md, step 1) <<<

Path:     x_t = (1 - t) * x0 + t * x1,     x0 ~ N(0, I) (noise),  x1 ~ data
Target:   u_t = d x_t / dt
Loss:     E_{t, x0, x1} || v_theta(x_t, t) - u_t ||^2
"""
import torch


def interpolate(x0, x1, t):
    """x_t on the straight path. t has shape (B,); broadcast it over the other dims of x."""
    # TODO(you)
    raise NotImplementedError


def target_velocity(x0, x1):
    """u_t for the straight path (does it depend on t?)."""
    # TODO(you)
    raise NotImplementedError


def fm_loss(model, x1, generator=None):
    """Monte-Carlo FM loss for a batch of data x1 (B, ...).

    Steps: sample x0 ~ N(0,I) like x1, sample t ~ U[0,1] of shape (B,),
    build x_t and u_t with the helpers above, return the mean squared error (scalar).
    Use `generator` for torch.randn / torch.rand if given (reproducibility).
    """
    # TODO(you)
    raise NotImplementedError

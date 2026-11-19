"""2D toy target distributions (the "data" x1 for flow matching)."""
import math

import torch


def eight_gaussians(n, scale=4.0, std=0.5, generator=None):
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = scale * torch.stack([angles.cos(), angles.sin()], dim=1)
    idx = torch.randint(0, 8, (n,), generator=generator)
    return centers[idx] + std * torch.randn(n, 2, generator=generator)


def moons(n, noise=0.1, generator=None):
    n1 = n // 2
    n2 = n - n1
    t1 = math.pi * torch.rand(n1, generator=generator)
    t2 = math.pi * torch.rand(n2, generator=generator)
    upper = torch.stack([t1.cos(), t1.sin()], dim=1)
    lower = torch.stack([1 - t2.cos(), 0.5 - t2.sin()], dim=1)
    x = torch.cat([upper, lower], dim=0)
    x = x + noise * torch.randn(n, 2, generator=generator)
    x = (x - torch.tensor([0.5, 0.25])) * 2.0  # roughly centred, unit-ish scale
    return x[torch.randperm(n, generator=generator)]


def checkerboard(n, scale=2.0, generator=None):
    """4x4 checkerboard on [-4, 4]^2 (FFJORD version, scaled): 8 filled squares of side 2."""
    x1 = 4 * torch.rand(n, generator=generator) - 2                     # column in [-2, 2)
    x2_ = torch.rand(n, generator=generator) - 2 * torch.randint(0, 2, (n,), generator=generator).float()
    x2 = x2_ + (torch.floor(x1) % 2)                                     # shift odd columns by one cell
    return scale * torch.stack([x1, x2], dim=1)


DATASETS = {
    "8gaussians": eight_gaussians,
    "moons": moons,
    "checkerboard": checkerboard,
}


def sample_toy(name, n, generator=None, device=None):
    """Sample n points. With `device` (and no generator) the points are generated directly on it."""
    if name not in DATASETS:
        raise KeyError(f"unknown toy dataset {name!r}; choose from {list(DATASETS)}")
    if device is not None and generator is None:
        with torch.device(device):
            return DATASETS[name](n)
    x = DATASETS[name](n, generator=generator)
    return x if device is None else x.to(device)

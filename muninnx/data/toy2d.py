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


def checkerboard(n, generator=None):
    x1 = 8 * torch.rand(n, generator=generator) - 4
    x2_ = 2 * torch.rand(n, generator=generator) - 2 + 2 * torch.randint(0, 2, (n,), generator=generator)
    x2 = x2_ + (torch.floor(x1 / 2) % 2) * 2 - 2
    return torch.stack([x1, x2 + 1], dim=1)


DATASETS = {
    "8gaussians": eight_gaussians,
    "moons": moons,
    "checkerboard": checkerboard,
}


def sample_toy(name, n, generator=None):
    if name not in DATASETS:
        raise KeyError(f"unknown toy dataset {name!r}; choose from {list(DATASETS)}")
    return DATASETS[name](n, generator=generator)

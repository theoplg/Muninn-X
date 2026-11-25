"""View a trajectory backbone (B, H, D) as a model on flat vectors (B, H*D).

All the point-based Muninn code (solvers, controller, calibration, extrapolation policy, AB2/RK4
samplers) is dimension-agnostic on (B, D) tensors, so wrapping is enough to reuse it on trajectories.
Distances must still use the paper's trajectory metric: d = ||a - b||_F / sqrt(H), i.e.
trajectory_distance(wrapper.unflat(a), wrapper.unflat(b)) or flat_distance below.
"""
import math

import torch
from torch import nn


class FlatWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.H, self.dim = model.H, model.dim

    def unflat(self, x):
        return x.view(-1, self.H, self.dim)

    def forward(self, x, t):
        return self.m(self.unflat(x), t).flatten(1)

    def probe(self, x, t):
        return self.m.probe(self.unflat(x), t)


def flat_distance(a, b, H):
    """Paper metric on flat trajectories: ||a - b||_F / sqrt(H), per sample."""
    return (a - b).flatten(1).norm(dim=1) / math.sqrt(H)

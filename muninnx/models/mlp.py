"""Time-conditioned MLP velocity network v_theta(x, t) for low-dimensional toys.

Structure:  embed_input(x, t) -> stem -> body -> head
`stem` is the first block: it is what the Muninn probe looks at (paper's "Stem-1").
"""
import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: (B,)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = 1000.0 * t[:, None] * freqs[None]  # t in [0,1] -> scale up
        return torch.cat([args.sin(), args.cos()], dim=1)


def as_batch_time(t, batch, device):
    """Accept a python float, 0-d tensor or (B,) tensor and return a (B,) tensor."""
    t = torch.as_tensor(t, dtype=torch.float32, device=device)
    if t.ndim == 0:
        t = t.expand(batch)
    return t


class VelocityMLP(nn.Module):
    def __init__(self, dim=2, hidden=256, depth=4, t_dim=64):
        super().__init__()
        self.dim = dim
        self.t_embed = SinusoidalEmbedding(t_dim)
        self.stem = nn.Sequential(nn.Linear(dim + t_dim, hidden), nn.SiLU())
        body = []
        for _ in range(depth - 1):
            body += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(hidden, dim)

    def embed_input(self, x, t):
        t = as_batch_time(t, x.shape[0], x.device)
        return torch.cat([x, self.t_embed(t)], dim=1)

    def forward(self, x, t):
        h = self.stem(self.embed_input(x, t))
        return self.head(self.body(h))

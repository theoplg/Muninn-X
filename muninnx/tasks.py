"""Task layer: one object that makes every experiment script backbone- and data-agnostic.

A Task bundles a velocity model acting on FLAT states (B, D), the noise shape, and the paper's
deviation metric d(a, b) = ||a - b||_F / sqrt(H). Point-based code (solvers, Muninn controller,
calibration, extrapolation policies, FewSteps samplers) then runs unchanged on:
  - 2D toys        (VelocityMLP checkpoints, D = 2, H = 1)
  - trajectory toy (muninnx.models.traj_backbones checkpoints, D = H * dim)
  - later Maze2D   (same interface)
"""
from dataclasses import dataclass

import torch
from torch.utils.flop_counter import FlopCounterMode

from muninnx.models.flat import FlatWrapper, flat_distance
from muninnx.models.mlp import VelocityMLP
from muninnx.models.traj_backbones import load_traj_model
from muninnx.utils.misc import load_checkpoint


@dataclass
class Task:
    model: torch.nn.Module  # flat velocity model: (B, D), t -> (B, D)
    D: int
    H: int
    name: str

    def noise(self, n, device, generator=None):
        return torch.randn(n, self.D, device=device, generator=generator)

    def dist(self, a, b):
        return flat_distance(a, b, self.H)

    @torch.no_grad()
    def probe_cost(self, n=256):
        """Probe cost as a fraction of one forward pass (FLOP count)."""
        dev = next(self.model.parameters()).device
        g = torch.Generator(device=dev).manual_seed(12345)  # private RNG: must not shift the global stream
        x, t = self.noise(n, dev, generator=g), torch.rand(n, device=dev, generator=g)
        flops = []
        for fn in (lambda: self.model.probe(x, t) if hasattr(self.model, "probe")
                   else self.model.stem(self.model.embed_input(x, t)), lambda: self.model(x, t)):
            with FlopCounterMode(display=False) as fc:
                fn()
            flops.append(fc.get_total_flops())
        return flops[0] / flops[1]


def load_task(ckpt, device):
    cfg = torch.load(ckpt, map_location="cpu")["cfg"]
    if "arch" in cfg:  # trajectory backbone
        model, cfg = load_traj_model(ckpt, device)
        return Task(FlatWrapper(model).eval(), D=model.H * model.dim, H=model.H, name=f"traj_{cfg['arch']}")
    model, cfg = load_checkpoint(ckpt, VelocityMLP, device)
    return Task(model.eval(), D=cfg["model"]["dim"], H=1, name=f"toy_{cfg.get('dataset', 'unknown')}")

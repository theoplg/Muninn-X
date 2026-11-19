"""Measure how much the sampling flow amplifies a small perturbation (tests the K_k = 1 assumption).

    python scripts/flow_sensitivity.py --ckpt runs/toy_checkerboard/model.pt

For several steps k, perturb x_{k+1} by a random vector of norm delta, integrate the
full Euler chain to the end and report G_k = ||x_N' - x_N|| / delta (quantiles over samples).
The Euler bound assumes G_k = 1 (prod_{j>k} K_j with K_j = 1).
"""
import argparse

import torch

from muninnx.models.mlp import VelocityMLP
from muninnx.solvers.euler import EulerSolver
from muninnx.utils.misc import get_device, load_checkpoint, seed_all


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 20, 40, 60, 80, 90, 94])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    model, _ = load_checkpoint(args.ckpt, VelocityMLP, device)
    S = EulerSolver(args.n_steps)

    x = torch.randn(args.n, 2, device=device)
    xs = [x]
    for k in range(args.n_steps):
        x = S.step(x, model(x, S.t(k)), k)
        xs.append(x)
    x_N = x

    qs = torch.tensor([0.5, 0.9, 0.99], device=device)
    print("  k  median    p90    p99     max   (G_k = ||dx_N|| / ||dx_{k+1}||)")
    for k in args.ks:
        u = torch.randn(args.n, 2, device=device)
        y = xs[k + 1] + args.delta * u / u.norm(dim=1, keepdim=True)
        for j in range(k + 1, args.n_steps):
            y = S.step(y, model(y, S.t(j)), j)
        g = (y - x_N).norm(dim=1) / args.delta
        q = torch.quantile(g, qs)
        print(f"{k:3d} {q[0]:7.2f} {q[1]:6.2f} {q[2]:6.2f} {g.max():7.1f}")


if __name__ == "__main__":
    main()

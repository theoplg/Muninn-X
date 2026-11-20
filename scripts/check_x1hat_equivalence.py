"""Check that reusing x1_hat = x + (1-t) v is identical to reusing v (Euler, straight FM path).

    python scripts/check_x1hat_equivalence.py --ckpt runs/toy_checkerboard/model.pt

Runs the SAME random reuse pattern with both caches in float64 and prints the max difference
of the final samples. Analytically: Euler on dx/dt = (x1_hat - x)/(1-t) with x1_hat fixed
telescopes to x_k = x_j + (t_k - t_j) v_j, i.e. one big Euler step with the old velocity.
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
    p.add_argument("--p-recompute", type=float, default=0.3)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_all(0)
    dev = get_device(args.device)
    model, _ = load_checkpoint(args.ckpt, VelocityMLP, dev)
    model = model.double()
    N, S = args.n_steps, EulerSolver(args.n_steps)
    x0 = torch.randn(args.n, 2, device=dev, dtype=torch.float64)
    rec = torch.rand(N, args.n, device=dev) < args.p_recompute
    rec[0] = True
    rec[-5:] = True

    xv, xx = x0.clone(), x0.clone()
    for k in range(N):
        t, dt, r = S.t(k), S.dt(k), rec[k][:, None]
        v_new = model(xv, t)
        vc = v_new if k == 0 else torch.where(r, v_new, vc)
        xv = xv + dt * vc
        x1_new = xx + (1 - t) * model(xx, t)
        x1c = x1_new if k == 0 else torch.where(r, x1_new, x1c)
        xx = xx + dt * (x1c - xx) / (1 - t)
    print(f"max |x_N(v reuse) - x_N(x1_hat reuse)| = {(xv - xx).abs().max().item():.3e}")
    print(f"mean ||x_N||                           = {xv.norm(dim=1).mean().item():.3f}")


if __name__ == "__main__":
    main()

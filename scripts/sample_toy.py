"""Full-compute Euler sampling from a trained toy model + plots.

    python scripts/sample_toy.py --ckpt runs/toy_8gaussians/model.pt --n-steps 100
"""
import argparse
from pathlib import Path

import torch

from muninnx.data.toy2d import sample_toy
from muninnx.models.mlp import VelocityMLP
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver
from muninnx.utils.misc import get_device, load_checkpoint, seed_all
from muninnx.utils.plot import plot_paths, scatter_compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, nargs="+", default=[100, 10, 3])
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    model, cfg = load_checkpoint(args.ckpt, VelocityMLP, device)
    out_dir = Path(args.ckpt).parent

    x0 = torch.randn(args.n, 2, device=device)
    panels = {"data": sample_toy(cfg["dataset"], args.n)}
    for n_steps in args.n_steps:
        x1, traj = sample_full(model, EulerSolver(n_steps), x0, return_traj=True)
        panels[f"Euler N={n_steps}"] = x1
        if n_steps == max(args.n_steps):
            plot_paths(traj, out_dir / "paths.png")
    scatter_compare(panels, out_dir / "samples.png")
    print(f"wrote {out_dir / 'samples.png'} and {out_dir / 'paths.png'}")


if __name__ == "__main__":
    main()

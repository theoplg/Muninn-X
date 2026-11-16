"""First experiment (E0): Muninn-FM on a 2D toy, Euler solver.

    python scripts/run_muninn_toy.py --ckpt runs/toy_8gaussians/model.pt

Pipeline: calibrate once -> for each (eta, alpha): fit U(s), sample with the
controller, compare to the full chain from the SAME noise, compare to plain
Euler with fewer steps at matched NFE ("FewSteps" baseline).
"""
import argparse
from pathlib import Path

import torch

from muninnx.cache.calibrate import ConformalUpperBound, collect_calibration_pairs
from muninnx.cache.controller import MuninnConfig, cache_steps, muninn_sample
from muninnx.eval.paired import summarize, trajectory_distance
from muninnx.models.mlp import VelocityMLP
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver
from muninnx.utils.misc import get_device, load_checkpoint, save_json, seed_all
from muninnx.utils.plot import plot_lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-cal", type=int, default=500)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--etas", type=float, nargs="+", default=[0.05, 0.15, 0.30])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--forbid", type=int, default=5, help="forbidden prefix = suffix length")
    p.add_argument("--max-lag", type=int, default=10, help="max consecutive reuses of one velocity")
    p.add_argument("--omega", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    model, _ = load_checkpoint(args.ckpt, VelocityMLP, device)
    solver = EulerSolver(args.n_steps)
    out_dir = Path(args.ckpt).parent / "muninn"

    base_cfg = MuninnConfig(eta=0.0, alpha=0.0, omega=args.omega,
                            forbid_prefix=args.forbid, forbid_suffix=args.forbid, max_lag=args.max_lag)
    steps = cache_steps(args.n_steps, base_cfg)

    # ---- calibration (independent noise from the test set) ----
    x0_cal = torch.randn(args.n_cal, 2, device=device)
    pairs = collect_calibration_pairs(model, solver, x0_cal, steps, max_lag=args.max_lag, omega=args.omega)
    lag1 = pairs["lag"] == 1
    ks = sorted(set(pairs["k"].tolist()))
    mean_s = [pairs["s"][lag1 & (pairs["k"] == k)].mean().item() for k in ks]
    mean_e = [pairs["eps"][lag1 & (pairs["k"] == k)].mean().item() for k in ks]
    plot_lines(ks, {"mean s (lag 1)": mean_s, "mean eps (lag 1)": mean_e}, out_dir / "calibration_vs_step.png",
               xlabel="step k", title="probe score and reuse error along the chain")
    lags = sorted(set(pairs["lag"].tolist()))
    plot_lines(lags, {"mean eps": [pairs["eps"][pairs["lag"] == l].mean().item() for l in lags]},
               out_dir / "error_vs_lag.png", xlabel="staleness (steps)", title="reuse error vs staleness")

    # ---- test set, paired with the full chain ----
    x0_test = torch.randn(args.n_test, 2, device=device)
    x_full = sample_full(model, solver, x0_test)

    results = []
    for alpha in args.alphas:
        alpha_step = alpha / len(steps)
        bound = ConformalUpperBound().fit(pairs["s"], pairs["eps"], alpha_step, seed=args.seed)
        for eta in args.etas:
            cfg = MuninnConfig(eta=eta, alpha=alpha, omega=args.omega,
                               forbid_prefix=args.forbid, forbid_suffix=args.forbid, max_lag=args.max_lag)
            x_cached, info = muninn_sample(model, solver, x0_test, bound, cfg)
            d = trajectory_distance(x_full, x_cached)
            row = {"method": "muninn", "eta": eta, "alpha": alpha,
                   **summarize(d, eta, info["nfe"], args.n_steps),
                   # does the per-sample certificate hold? (d <= sum of c_hat over reused steps)
                   "bound_holds_frac": (d <= info["bound_used"] + 1e-6).float().mean().item()}

            # FewSteps baseline at matched (mean) NFE
            n_few = max(1, round(info["nfe"].float().mean().item()))
            x_few = sample_full(model, EulerSolver(n_few), x0_test)
            d_few = trajectory_distance(x_full, x_few)
            few = {"method": "fewsteps", "eta": eta, "alpha": alpha, "n_steps": n_few,
                   **summarize(d_few, eta, torch.full_like(info["nfe"], n_few), args.n_steps)}
            results += [row, few]
            print(f"alpha={alpha:.2f} eta={eta:.2f} | Muninn: NFE {row['nfe_mean']:.1f} "
                  f"E[d] {row['E_d']:.4f} p_viol {row['p_viol']:.3f} | "
                  f"FewSteps(N={n_few}): E[d] {few['E_d']:.4f} p_viol {few['p_viol']:.3f}")

    save_json({"args": vars(args), "results": results}, out_dir / "results.json")
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

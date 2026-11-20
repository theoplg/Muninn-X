"""E4: certified FewSteps vs Muninn at the same (eta, alpha).

    python scripts/certified_fewsteps.py --ckpt runs/toy_checkerboard/model.pt

Both methods get the same calibration budget (n_cal chains from independent noise) and are
evaluated on the same test noise against the full N-step chain.
Writes <ckpt dir>/certified_fewsteps/results.json.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import torch

from muninnx.baselines.fewsteps import fewsteps_distances, select_n_steps
from muninnx.cache.calibrate import ConformalUpperBound, collect_calibration_pairs
from muninnx.cache.controller import MuninnConfig, cache_steps, muninn_sample
from muninnx.eval.paired import summarize, trajectory_distance
from muninnx.models.mlp import VelocityMLP
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver
from muninnx.utils.misc import get_device, load_checkpoint, save_json, seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-cal", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=50000)
    p.add_argument("--etas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--delta", type=float, default=0.05, help="LTT confidence level")
    p.add_argument("--forbid", type=int, default=5)
    p.add_argument("--max-lag", type=int, default=10)
    p.add_argument("--omega", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    model, _ = load_checkpoint(args.ckpt, VelocityMLP, device)
    N = args.n_steps
    solver = EulerSolver(N)
    candidates = list(range(1, N + 1))

    # ---- calibration (shared noise for both methods) ----
    x0_cal = torch.randn(args.n_cal, 2, device=device)
    d_cal = fewsteps_distances(model, x0_cal, N, candidates)
    base = MuninnConfig(eta=0.0, alpha=0.0, omega=args.omega,
                        forbid_prefix=args.forbid, forbid_suffix=args.forbid, max_lag=args.max_lag)
    steps = cache_steps(N, base)
    pairs = collect_calibration_pairs(model, solver, x0_cal, steps, max_lag=args.max_lag, omega=args.omega)

    # ---- test ----
    x0 = torch.randn(args.n_test, 2, device=device)
    x_full = sample_full(model, solver, x0)
    d_test = {}  # lazily computed FewSteps test distances

    def few_test(n):
        if n not in d_test:
            d_test[n] = trajectory_distance(x_full, sample_full(model, EulerSolver(n), x0))
        return d_test[n]

    rows = []
    for alpha in args.alphas:
        bound = ConformalUpperBound().fit(pairs["s"], pairs["eps"], alpha / len(steps), seed=args.seed)
        for eta in args.etas:
            cfg = replace(base, eta=eta, alpha=alpha)
            x, info = muninn_sample(model, solver, x0, bound, cfg)
            d = trajectory_distance(x_full, x)
            r = {"method": "muninn", "eta": eta, "alpha": alpha, **summarize(d, eta, info["nfe"], N)}
            rows.append(r)
            line = f"a={alpha:.2f} eta={eta:<5} | Muninn NFE {r['nfe_mean']:5.1f} E[d] {r['E_d']:.4f} pv {r['p_viol']:.4f}"
            for method in ("conformal", "ltt"):
                n_sel = select_n_steps(d_cal, eta, alpha, method, args.delta)
                dn = few_test(n_sel)
                r = {"method": f"certfew_{method}", "eta": eta, "alpha": alpha, "n_steps": n_sel,
                     **summarize(dn, eta, torch.full((args.n_test,), n_sel), N)}
                rows.append(r)
                line += f" | {method}: N'={n_sel:3d} E[d] {r['E_d']:.4f} pv {r['p_viol']:.4f}"
            print(line)

    out = Path(args.ckpt).parent / "certified_fewsteps" / "results.json"
    save_json({"args": vars(args), "results": rows}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

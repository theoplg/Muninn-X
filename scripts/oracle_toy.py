"""Oracle experiment: can velocity reuse beat FewSteps at all?

    python scripts/oracle_toy.py --ckpt runs/toy_checkerboard/model.pt

For a sweep of eta, compares at matched mean NFE:
  - Muninn          (score + conformal bound U(s), the real method)
  - Oracle, max_lag (true reuse error, same lag cap as Muninn)
  - Oracle, no cap  (true reuse error, unlimited consecutive reuse)
  - FewSteps        (uniform Euler with N' steps, full curve N' = 2..N)
Distances d are always to the full N-step chain from the same noise.
Writes <ckpt dir>/oracle/results.json and pareto.png (E[d] vs mean NFE).
"""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from muninnx.cache.calibrate import ConformalUpperBound, collect_calibration_pairs
from muninnx.cache.controller import MuninnConfig, cache_steps, muninn_sample
from muninnx.cache.oracle import oracle_sample
from muninnx.eval.paired import summarize, trajectory_distance
from muninnx.models.mlp import VelocityMLP
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver
from muninnx.utils.misc import get_device, load_checkpoint, save_json, seed_all

# categorical slots 1-4 (validated palette), fixed order; markers give a second encoding
STYLE = {
    "fewsteps":      dict(color="#2a78d6", marker="o", label="FewSteps (uniform Euler)"),
    "muninn":        dict(color="#eb6834", marker="s", label="Muninn (U(s), max_lag)"),
    "oracle_lag":    dict(color="#1baf7a", marker="^", label="Oracle (true ε, max_lag)"),
    "oracle_nolag":  dict(color="#eda100", marker="D", label="Oracle (true ε, no lag cap)"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-cal", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=50000)
    p.add_argument("--etas", type=float, nargs="+",
                   default=[0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 0.5, 1.0, 2.0])
    p.add_argument("--alpha", type=float, default=0.05)
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
    out_dir = Path(args.ckpt).parent / "oracle"

    base = MuninnConfig(eta=0.0, alpha=args.alpha, omega=args.omega,
                        forbid_prefix=args.forbid, forbid_suffix=args.forbid, max_lag=args.max_lag)
    steps = cache_steps(N, base)

    x0_cal = torch.randn(args.n_cal, 2, device=device)
    pairs = collect_calibration_pairs(model, solver, x0_cal, steps, max_lag=args.max_lag, omega=args.omega)
    bound = ConformalUpperBound().fit(pairs["s"], pairs["eps"], args.alpha / len(steps), seed=args.seed)

    x0 = torch.randn(args.n_test, 2, device=device)
    x_full = sample_full(model, solver, x0)

    rows = []
    for n_few in range(2, N + 1):
        d = trajectory_distance(x_full, sample_full(model, EulerSolver(n_few), x0))
        rows.append({"method": "fewsteps", "n_steps": n_few,
                     **summarize(d, float("inf"), torch.full((args.n_test,), n_few), N)})

    for eta in args.etas:
        cfg = replace(base, eta=eta)
        runs = {
            "muninn": lambda: muninn_sample(model, solver, x0, bound, cfg),
            "oracle_lag": lambda: oracle_sample(model, solver, x0, cfg),
            "oracle_nolag": lambda: oracle_sample(model, solver, x0, replace(cfg, max_lag=N)),
        }
        for name, fn in runs.items():
            x, info = fn()
            d = trajectory_distance(x_full, x)
            row = {"method": name, "eta": eta, **summarize(d, eta, info["nfe"], N),
                   "bound_holds_frac": (d <= info["bound_used"] + 1e-6).float().mean().item()}
            rows.append(row)
            print(f"eta={eta:<5} {name:13s} NFE {row['nfe_mean']:6.1f}  E[d] {row['E_d']:.4f}  "
                  f"p_viol {row['p_viol']:.4f}  bound_holds {row['bound_holds_frac']:.4f}")

    save_json({"args": vars(args), "results": rows}, out_dir / "results.json")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, st in STYLE.items():
        r = sorted((x for x in rows if x["method"] == name), key=lambda x: x["nfe_mean"])
        r = [x for x in r if x["E_d"] > 0]
        ax.plot([x["nfe_mean"] for x in r], [x["E_d"] for x in r], lw=2, ms=6 if name != "fewsteps" else 0,
                **st)
    ax.set_yscale("log")
    ax.set_xlabel("mean NFE (probe cost not counted)")
    ax.set_ylabel("E[d] to the full N=%d chain" % N)
    ax.set_title(f"{Path(args.ckpt).parent.name}: error vs compute")
    ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=150)
    print(f"wrote {out_dir / 'results.json'} and {out_dir / 'pareto.png'}")


if __name__ == "__main__":
    main()

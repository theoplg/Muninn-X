"""E5: can adaptive EXTRAPOLATION (instead of reuse) beat FewSteps?

    python scripts/extrap_oracle_toy.py --ckpt runs/toy_checkerboard/model.pt

Methods (all on the same test noise):
  - FewSteps Euler / AB2 / Heun on uniform grids (NFE = N', N', 2N')
  - Oracle reuse        : E2 oracle, reuse v_cache (true error decides)
  - Oracle extrapolate  : linear extrapolation from the last two computed velocities (true error decides),
                          with max_lag and without lag cap
Two references:
  - "euler": the full Euler N-step chain (what the Muninn certificate targets)
  - "exact": RK4 with --n-ref steps (near-exact ODE solution = actual sample quality)
Writes <ckpt dir>/extrap_oracle/results.json and pareto.png.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from muninnx.baselines.fewsteps import sample_ab2, sample_heun, sample_rk4
from muninnx.cache.controller import MuninnConfig
from muninnx.cache.oracle import oracle_extrap_sample, oracle_sample
from muninnx.eval.paired import trajectory_distance
from muninnx.models.mlp import VelocityMLP
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver
from muninnx.utils.misc import get_device, load_checkpoint, save_json, seed_all

# validated categorical palette, fixed order; marker + line style as second encoding
STYLE = {
    "few_euler":         dict(color="#2a78d6", marker="o", ls="-",  label="FewSteps Euler"),
    "few_ab2":           dict(color="#eb6834", marker="s", ls="-",  label="FewSteps AB2"),
    "few_heun":          dict(color="#1baf7a", marker="^", ls="-",  label="FewSteps Heun"),
    "oracle_reuse":      dict(color="#eda100", marker="D", ls="--", label="Oracle reuse (max_lag 10)"),
    "oracle_extrap":     dict(color="#e87ba4", marker="v", ls="--", label="Oracle extrapolate (max_lag 10)"),
    "oracle_extrap_nolag": dict(color="#008300", marker="P", ls="--", label="Oracle extrapolate (no cap)"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-ref", type=int, default=400, help="RK4 steps for the exact reference")
    p.add_argument("--n-test", type=int, default=50000)
    p.add_argument("--etas", type=float, nargs="+",
                   default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    p.add_argument("--forbid", type=int, default=5)
    p.add_argument("--max-lag", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    model, _ = load_checkpoint(args.ckpt, VelocityMLP, device)
    N = args.n_steps
    solver = EulerSolver(N)

    x0 = torch.randn(args.n_test, 2, device=device)
    refs = {"euler": sample_full(model, solver, x0), "exact": sample_rk4(model, args.n_ref, x0)}
    euler_vs_exact = trajectory_distance(refs["exact"], refs["euler"]).mean().item()
    print(f"E[d(Euler-{N}, exact)] = {euler_vs_exact:.4f}")

    rows = []

    def record(method, nfe, x, **extra):
        r = {"method": method, "nfe_mean": float(nfe), **extra}
        for name, ref in refs.items():
            d = trajectory_distance(ref, x).cpu()
            r[f"E_d_{name}"] = float(d.mean())
            r[f"p95_d_{name}"] = float(d.quantile(0.95))
        rows.append(r)
        return r

    for n in sorted(set([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100])):
        record("few_euler", n, sample_full(model, EulerSolver(n), x0), n_steps=n)
        record("few_ab2", n, sample_ab2(model, n, x0), n_steps=n)
        if n <= 50:
            record("few_heun", 2 * n, sample_heun(model, n, x0), n_steps=n)

    base = MuninnConfig(eta=0.0, alpha=0.05, forbid_prefix=args.forbid, forbid_suffix=args.forbid,
                        max_lag=args.max_lag)
    for eta in args.etas:
        cfg = replace(base, eta=eta)
        runs = {
            "oracle_reuse": lambda: oracle_sample(model, solver, x0, cfg),
            "oracle_extrap": lambda: oracle_extrap_sample(model, solver, x0, cfg),
            "oracle_extrap_nolag": lambda: oracle_extrap_sample(model, solver, x0, replace(cfg, max_lag=N)),
        }
        for name, fn in runs.items():
            x, info = fn()
            r = record(name, info["nfe"].float().mean().item(), x, eta=eta)
            print(f"eta={eta:<6} {name:20s} NFE {r['nfe_mean']:6.1f}  "
                  f"E[d] vs Euler {r['E_d_euler']:.4f}  vs exact {r['E_d_exact']:.4f}")

    out_dir = Path(args.ckpt).parent / "extrap_oracle"
    save_json({"args": vars(args), "euler_vs_exact": euler_vs_exact, "results": rows}, out_dir / "results.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    titles = {"euler": f"vs full Euler-{N} chain (certificate target)",
              "exact": f"vs exact ODE (RK4-{args.n_ref}) = sample quality"}
    for ax, ref in zip(axes, ("euler", "exact")):
        for name, st in STYLE.items():
            r = sorted((x for x in rows if x["method"] == name), key=lambda x: x["nfe_mean"])
            r = [x for x in r if x[f"E_d_{ref}"] > 0 and x["nfe_mean"] <= N]
            ax.plot([x["nfe_mean"] for x in r], [x[f"E_d_{ref}"] for x in r], lw=2, ms=5, **st)
        if ref == "exact":
            ax.axhline(euler_vs_exact, color="#888888", lw=1, ls=":")
            ax.text(N, euler_vs_exact, f" Euler-{N} itself", va="bottom", ha="right", fontsize=8, color="#555555")
        ax.set_yscale("log")
        ax.set_xlabel("mean NFE (probe cost not counted)")
        ax.set_title(titles[ref], fontsize=10)
        ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("E[d]")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{Path(args.ckpt).parent.name}: reuse vs extrapolation vs FewSteps")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto.png", dpi=150)
    print(f"wrote {out_dir / 'results.json'} and {out_dir / 'pareto.png'}")


if __name__ == "__main__":
    main()

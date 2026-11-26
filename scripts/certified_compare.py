"""E8: end-to-end certified Muninn vs certified FewSteps (Euler) vs certified FewSteps (AB2).

    python scripts/certified_compare.py --ckpt runs/toy_checkerboard/model.pt

Every method is certified the SAME way: on calibration noise, fixed-sequence testing picks the most
aggressive configuration with P(d > eta) <= alpha (conformal, and LTT with delta), then it is
evaluated on fresh test noise.
  - FewSteps Euler / AB2: configuration = number of steps N' (tested from large to small).
  - Muninn: configuration = the controller's internal budget lambda (tested from 0 upwards).
    U(s) is fitted on separate noise and is only a heuristic here: validity comes from calibration.
Two references for d: the exact ODE (RK4, --n-ref steps) and the full Euler N-step chain.
Writes <ckpt dir>/certified_compare/results.json and certified_nfe.png.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from muninnx.baselines.fewsteps import fixed_sequence_select, sample_ab2, sample_rk4
from muninnx.cache.calibrate import ConformalUpperBound, collect_calibration_pairs
from muninnx.cache.controller import MuninnConfig, cache_steps, muninn_sample
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver
from muninnx.tasks import load_task
from muninnx.utils.misc import get_device, save_json, seed_all

STYLE = {
    "euler":  dict(color="#2a78d6", marker="o", ls="-", label="certified FewSteps Euler"),
    "ab2":    dict(color="#eb6834", marker="s", ls="-", label="certified FewSteps AB2"),
    "muninn": dict(color="#1baf7a", marker="^", ls="-", label="certified Muninn (NFE)"),
    "muninn_probe": dict(color="#1baf7a", marker="^", ls="--", label="certified Muninn (NFE + probe)"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-ref", type=int, default=400)
    p.add_argument("--n-fit", type=int, default=2000, help="chains used to fit U(s)")
    p.add_argument("--n-cal", type=int, default=4000, help="chains used for certification")
    p.add_argument("--n-test", type=int, default=50000)
    p.add_argument("--etas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5,
                            0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    p.add_argument("--forbid", type=int, default=5)
    p.add_argument("--max-lag", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-steps", type=int, default=400, help="largest FewSteps N' tried")
    args = p.parse_args()

    seed_all(args.seed)
    dev = get_device(args.device)
    task = load_task(args.ckpt, dev)
    model = task.model
    N = args.n_steps
    solver = EulerSolver(N)
    c_probe = task.probe_cost()
    candidates = [n for n in list(range(1, 101)) + list(range(110, 401, 10)) if n <= args.max_steps]  # N', safest = largest

    x0 = {s: task.noise(n, dev) for s, n in
          (("fit", args.n_fit), ("cal", args.n_cal), ("test", args.n_test))}
    refs = {s: {"exact": sample_rk4(model, args.n_ref, x0[s]), "euler": sample_full(model, solver, x0[s])}
            for s in ("cal", "test")}

    # Muninn's score bound (fitted on its own noise; only a heuristic for the decisions)
    base = MuninnConfig(eta=0.0, alpha=0.05, forbid_prefix=args.forbid, forbid_suffix=args.forbid,
                        max_lag=args.max_lag)
    steps = cache_steps(N, base)
    pairs = collect_calibration_pairs(model, solver, x0["fit"], steps, max_lag=args.max_lag)
    bound = ConformalUpperBound().fit(pairs["s"], pairs["eps"], 0.05 / len(steps), seed=args.seed)

    def run(method, param, split):
        if method == "euler":
            return sample_full(model, EulerSolver(param), x0[split]), float(param)
        if method == "ab2":
            return sample_ab2(model, param, x0[split]), float(param)
        x, info = muninn_sample(model, solver, x0[split], bound, replace(base, eta=param))
        return x, info["nfe"].float().mean().item()

    # Configurations ordered safest -> most aggressive, PER REFERENCE (the order must be fixed in
    # advance, not chosen from data). Against the Euler-N reference the safest Euler config is N'=N
    # (d = 0), so larger N' are excluded; against the exact ODE, larger N' is always safer.
    desc = sorted(candidates, reverse=True)
    configs = {
        "exact": {"euler": desc, "ab2": desc, "muninn": sorted(args.lambdas)},
        "euler": {"euler": [n for n in desc if n <= N], "ab2": desc, "muninn": sorted(args.lambdas)},
    }
    # keep only distances (a few numbers per plan), never the sampled states: on trajectories the states of
    # ~300 configurations x n_cal plans would take several GB of GPU memory
    d_by_cfg = {}
    for ref_cfg in configs.values():
        for m, params in ref_cfg.items():
            for prm in params:
                if (m, prm) not in d_by_cfg:
                    x = run(m, prm, "cal")[0]
                    d_by_cfg[(m, prm)] = {r: task.dist(refs["cal"][r], x).cpu() for r in ("exact", "euler")}
                    del x
    d_cal = {ref: {m: [d_by_cfg[(m, prm)][ref] for prm in params] for m, params in ref_cfg.items()}
             for ref, ref_cfg in configs.items()}

    test_cache = {}

    def evaluate(m, prm, ref, eta):
        if (m, prm) not in test_cache:  # cache distances to both references, not the sampled states
            x, nfe_ = run(m, prm, "test")
            test_cache[(m, prm)] = ({r: task.dist(refs["test"][r], x) for r in ("exact", "euler")}, nfe_)
            del x
        dd, nfe = test_cache[(m, prm)]
        d = dd[ref]
        return {"nfe": nfe, "E_d": d.mean().item(), "p_viol": (d > eta).float().mean().item()}

    rows = []
    for ref in ("exact", "euler"):
        for alpha in args.alphas:
            for eta in args.etas:
                line = f"ref={ref:5s} a={alpha:.2f} eta={eta:<5}"
                for m, params in configs[ref].items():
                    for method in ("conformal", "ltt"):
                        i = fixed_sequence_select(d_cal[ref][m], eta, alpha, method, args.delta)
                        row = {"ref": ref, "alpha": alpha, "eta": eta, "method": m, "test": method,
                               "param": None if i is None else params[i]}
                        if i is not None:
                            row.update(evaluate(m, params[i], ref, eta))
                            if m == "muninn":
                                row["nfe_with_probe"] = row["nfe"] + c_probe * N
                        rows.append(row)
                        if method == "conformal":
                            line += (f" | {m}: infeasible" if i is None else
                                     f" | {m}: {'N' if m != 'muninn' else 'λ'}={params[i]} "
                                     f"NFE {row['nfe']:5.1f} pv {row['p_viol']:.3f}")
                print(line)

    out_dir = Path(args.ckpt).parent / "certified_compare"
    save_json({"args": vars(args), "probe_cost": c_probe, "results": rows}, out_dir / "results.json")
    print(f"probe cost = {c_probe:.3f} forward passes")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, ref in zip(axes, ("exact", "euler")):
        sel = [r for r in rows if r["ref"] == ref and r["alpha"] == 0.05 and r["test"] == "conformal"
               and r["param"] is not None]
        for key, st in STYLE.items():
            m = "muninn" if key.startswith("muninn") else key
            pts = sorted((r for r in sel if r["method"] == m), key=lambda r: r["eta"])
            y = [r["nfe_with_probe"] if key == "muninn_probe" else r["nfe"] for r in pts]
            ax.plot([r["eta"] for r in pts], y, lw=2, ms=6, **st)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("tolerance η  (certified: P(d > η) ≤ 0.05)")
        ax.set_title({"exact": f"d to the exact ODE (RK4-{args.n_ref})",
                      "euler": f"d to the full Euler-{N} chain"}[ref], fontsize=10)
        ax.grid(True, which="both", color="#e5e5e5", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("mean NFE needed (lower is better)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{Path(args.ckpt).parent.name}: compute needed for the same certified guarantee")
    fig.tight_layout()
    fig.savefig(out_dir / "certified_nfe.png", dpi=150)
    print(f"wrote {out_dir / 'results.json'} and {out_dir / 'certified_nfe.png'}")


if __name__ == "__main__":
    main()

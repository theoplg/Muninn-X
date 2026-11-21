"""E9: can per-sample adaptive extrapolation beat certified AB2, even as an oracle?

    python scripts/g_aware_oracle.py --ckpt runs/toy_checkerboard/model.pt

Policies (oracle_extrap_ab2_sample, AB2 base grid with N steps, extrapolate-or-compute per sample):
  plain-threshold, plain-budget : cost = h ||e_k||
  G-threshold,     G-budget     : cost = h ||G_{k+1->N} e_k||  (G along the AB2-N reference chain)
Every policy and certified FewSteps AB2 are certified the same way (E8): fixed-sequence testing of the
knob on calibration chains against the exact ODE, then evaluated on fresh test chains.
NFE counts only computed steps; the oracle's free evaluations and G are NOT counted (best case).
Writes <ckpt dir>/g_aware_oracle/results.json and certified_nfe.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from muninnx.analysis.flow import flow_jacobians
from muninnx.baselines.fewsteps import bonferroni_select, fixed_sequence_select, sample_ab2, sample_rk4
from muninnx.cache.oracle import oracle_extrap_ab2_sample
from muninnx.eval.paired import trajectory_distance
from muninnx.models.mlp import VelocityMLP
from muninnx.utils.misc import get_device, load_checkpoint, save_json, seed_all

POLICIES = {  # name: (rule, G-aware?)
    "plain_threshold": ("threshold", False),
    "plain_budget": ("budget", False),
    "G_threshold": ("threshold", True),
    "G_budget": ("budget", True),
}
STYLE = {
    "ab2":             dict(color="#2a78d6", marker="o", ls="-",  label="certified FewSteps AB2"),
    "plain_threshold": dict(color="#eb6834", marker="s", ls="--", label="oracle extrap, cost h‖e‖, threshold"),
    "plain_budget":    dict(color="#1baf7a", marker="^", ls="--", label="oracle extrap, cost h‖e‖, budget"),
    "G_threshold":     dict(color="#eda100", marker="D", ls="-",  label="oracle extrap, cost h‖Ge‖, threshold"),
    "G_budget":        dict(color="#e87ba4", marker="v", ls="-",  label="oracle extrap, cost h‖Ge‖, budget"),
}


@torch.no_grad()
def ab2_states(model, x0, n):
    """States x_0..x_{n-1} of the AB2-n chain (for the flow Jacobians)."""
    h, x, v_prev, xs = 1.0 / n, x0, None, []
    for k in range(n):
        xs.append(x)
        v = model(x, k * h)
        x = x + h * (v if v_prev is None else 1.5 * v - 0.5 * v_prev)
        v_prev = v
    return xs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-ref", type=int, default=400)
    p.add_argument("--n-cal", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=50000)
    p.add_argument("--etas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2])
    p.add_argument("--budgets", type=float, nargs="+",
                   default=[0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--max-lag", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_all(args.seed)
    dev = get_device(args.device)
    model, _ = load_checkpoint(args.ckpt, VelocityMLP, dev)
    N, h = args.n_steps, 1.0 / args.n_steps
    x0 = {"cal": torch.randn(args.n_cal, 2, device=dev), "test": torch.randn(args.n_test, 2, device=dev)}
    ref = {s: sample_rk4(model, args.n_ref, x0[s]) for s in x0}
    G = {s: flow_jacobians(model, ab2_states(model, x0[s], N), [k * h for k in range(N)], h) for s in x0}

    candidates = sorted(set(range(2, 101)) | set(range(110, 401, 10)), reverse=True)  # AB2 N', safest first
    knobs = {name: (args.thresholds if rule == "threshold" else args.budgets) for name, (rule, _) in POLICIES.items()}

    def run(method, prm, split):
        if method == "ab2":
            return sample_ab2(model, prm, x0[split]), float(prm)
        rule, g_aware = POLICIES[method]
        x, nfe = oracle_extrap_ab2_sample(model, x0[split], N, prm, rule, G[split] if g_aware else None,
                                          args.max_lag)
        return x, nfe.float().mean().item()

    configs = {"ab2": candidates, **{name: sorted(k) for name, k in knobs.items()}}
    d_cal, nfe_cal = {}, {}
    for m, params in configs.items():
        d_cal[m], nfe_cal[m] = [], []
        for prm in params:
            x, nfe = run(m, prm, "cal")
            d_cal[m].append(trajectory_distance(ref["cal"], x).cpu())
            nfe_cal[m].append(nfe)

    cache, rows = {}, []
    for eta in args.etas:
        line = f"eta={eta:<6}"
        for m, params in configs.items():
            for test in ("conformal", "ltt", "bonferroni"):
                i = (bonferroni_select(d_cal[m], nfe_cal[m], eta, args.alpha, args.delta) if test == "bonferroni"
                     else fixed_sequence_select(d_cal[m], eta, args.alpha, test, args.delta))
                row = {"eta": eta, "method": m, "test": test, "param": None if i is None else params[i]}
                if i is not None:
                    if (m, params[i]) not in cache:
                        cache[(m, params[i])] = run(m, params[i], "test")
                    x, nfe = cache[(m, params[i])]
                    d = trajectory_distance(ref["test"], x)
                    row.update(nfe=nfe, E_d=d.mean().item(), p_viol=(d > eta).float().mean().item())
                rows.append(row)
                if test == "bonferroni":
                    line += f" | {m}: " + ("inf" if i is None else f"{row['nfe']:5.1f} (pv {row['p_viol']:.3f})")
        print(line)

    # uncertified Pareto view on calibration data: E[d] vs NFE for every configuration
    pareto = {m: [{"param": prm, "nfe": n, "E_d": d.mean().item(), "p95_d": d.quantile(0.95).item()}
                  for prm, n, d in zip(configs[m], nfe_cal[m], d_cal[m])] for m in configs}

    out_dir = Path(args.ckpt).parent / "g_aware_oracle"
    save_json({"args": vars(args), "results": rows, "pareto_cal": pareto}, out_dir / "results.json")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for m, st in STYLE.items():
        pts = sorted((r for r in rows if r["method"] == m and r["test"] == "bonferroni" and r["param"] is not None),
                     key=lambda r: r["eta"])
        ax[0].plot([r["eta"] for r in pts], [r["nfe"] for r in pts], lw=2, ms=6, **st)
        q = sorted((r for r in pareto[m] if r["nfe"] <= N), key=lambda r: r["nfe"])
        ax[1].plot([r["nfe"] for r in q], [r["p95_d"] for r in q], lw=2, ms=4, **st)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(f"tolerance η  (certified: P(d > η) ≤ {args.alpha}, LTT + Bonferroni)")
    ax[0].set_ylabel("mean NFE needed (lower is better)")
    ax[0].set_title("compute for the same certified guarantee", fontsize=10)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("mean NFE"); ax[1].set_ylabel("p95 of d to the exact ODE")
    ax[1].set_title("uncertified view: error tail vs compute (calibration chains)", fontsize=10)
    for a in ax:
        a.grid(True, which="both", color="#e5e5e5", lw=0.8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{Path(args.ckpt).parent.name}: adaptive extrapolation oracles vs certified AB2")
    fig.tight_layout()
    fig.savefig(out_dir / "certified_nfe.png", dpi=150)
    print(f"wrote {out_dir / 'results.json'} and {out_dir / 'certified_nfe.png'}")


if __name__ == "__main__":
    main()

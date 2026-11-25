"""E14: is the Muninn certificate gap generic? Same checks on DDPM / DDIM (the paper's samplers) and FM-Euler.

    python scripts/e14_diffusion_check.py

For each sampler (DDPM and DDIM on a VP epsilon-model, Euler on the FM velocity model; checkerboard):
 1. Oracle reuse chain (MRR cache, reuse iff L_t * ||out(tau_full_t) - cache|| <= remaining budget,
    max_lag 10, forbidden 5/5), paired with the full chain (same noise, same sampler noise xi).
    Per sample:  d = ||tau_0^full - tau~_0||
      B_claim = sum over REUSE steps of L_t ||out(tau_full_t) - used_t||   (the paper's certificate)
      B_all   = sum over ALL steps   of L_t ||out(tau_full_t) - used_t||   (valid by the paper's Eq. 1)
 2. Slack of the paper's weights: true flow Jacobian ||A_{N-1} ... A_{k+1}||, A_j = K_j I + c_j d(out)/dx,
    vs the paper's prod_{j>k} K_j, along the full chain.
 3. Real Muninn (staleness-aware calibration, alpha = 0.05): test p_viol and bound_holds on 50k samples.
Writes runs/ddpm_checkerboard/e14/results.json and e14.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from muninnx.analysis.flow import jacobian
from muninnx.cache.bound import sensitivity_weights
from muninnx.cache.calibrate import ConformalUpperBound, collect_calibration_pairs
from muninnx.cache.controller import MuninnConfig, cache_steps, muninn_sample
from muninnx.diffusion.ddpm import DDPMSolver
from muninnx.eval.paired import trajectory_distance, wilson_ci
from muninnx.models.mlp import VelocityMLP
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver
from muninnx.utils.misc import get_device, load_checkpoint, save_json, seed_all

COLORS = {"fm_euler": "#2a78d6", "ddpm": "#eb6834", "ddim": "#1baf7a"}
LABELS = {"fm_euler": "FM + Euler", "ddpm": "VP + DDPM (paper)", "ddim": "VP + DDIM (paper)"}


def eps_coef(solver, k):
    return solver.eps_coef(k) if hasattr(solver, "eps_coef") else solver.dt(k)


@torch.no_grad()
def oracle_paper(model, solver, x0, eta, cfg):
    N = solver.n_steps
    T = set(cache_steps(N, cfg))
    L = sensitivity_weights(solver).tolist()
    B, dev = x0.shape[0], x0.device
    zeros = lambda: torch.zeros(B, device=dev)
    xf, xc, cache = x0, x0, None
    budget = torch.full((B,), float(eta), device=dev)
    lag = torch.zeros(B, dtype=torch.long, device=dev)
    b_claim, b_all, n_reuse = zeros(), zeros(), zeros()
    states = []
    for k in range(N):
        t = solver.t(k)
        states.append(xf)
        out_full, out_here = model(xf, t), model(xc, t)
        if cache is None or k not in T:
            rec = torch.ones(B, dtype=torch.bool, device=dev)
            c = zeros()
        else:
            c = L[k] * (out_full - cache).norm(dim=1)
            rec = (c > budget) | (lag >= cfg.max_lag)
        used = out_here if cache is None else torch.where(rec[:, None], out_here, cache)
        cache = used
        reuse = ~rec
        b_claim += torch.where(reuse, c, zeros())
        b_all += L[k] * (out_full - used).norm(dim=1)
        n_reuse += reuse.float()
        budget = torch.where(reuse, budget - c, budget)
        lag = torch.where(rec, torch.zeros_like(lag), lag + 1)
        xf, xc = solver.step(xf, out_full, k), solver.step(xc, used, k)
    d = (xf - xc).norm(dim=1)
    return d, b_claim, b_all, n_reuse, states


def slack_ratio(model, solver, states):
    """(N, B) ratio  ||A_{N-1}...A_{k+1}||_op / prod_{j>k} K_j  along the given chain."""
    N = solver.n_steps
    B = states[0].shape[0]
    I = torch.eye(2, device=states[0].device).expand(B, 2, 2)
    P, prodK, out = I.clone(), 1.0, [None] * N
    for k in reversed(range(N)):
        out[k] = torch.linalg.matrix_norm(P, ord=2) / prodK
        K, _ = solver.coeffs(k)
        P = P @ (K * I + eps_coef(solver, k) * jacobian(model, states[k], solver.t(k)))
        prodK *= K
    return torch.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ddpm-ckpt", default="runs/ddpm_checkerboard/model.pt")
    p.add_argument("--fm-ckpt", default="runs/toy_checkerboard/model.pt")
    p.add_argument("--n-oracle", type=int, default=20000)
    p.add_argument("--n-cal", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=50000)
    p.add_argument("--etas", type=float, nargs="+", default=[0.05, 0.1, 0.3, 0.5, 1.0])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_all(args.seed)
    dev = get_device(args.device)
    ddpm_model, dcfg = load_checkpoint(args.ddpm_ckpt, VelocityMLP, dev)
    fm_model, _ = load_checkpoint(args.fm_ckpt, VelocityMLP, dev)
    T = dcfg["T"]
    setups = {
        "fm_euler": (fm_model, lambda seed: EulerSolver(T)),
        "ddpm": (ddpm_model, lambda seed: DDPMSolver(T, "ddpm", noise_seed=seed)),
        "ddim": (ddpm_model, lambda seed: DDPMSolver(T, "ddim", noise_seed=seed)),
    }
    cfg = MuninnConfig(eta=0.0, alpha=args.alpha, forbid_prefix=5, forbid_suffix=5, max_lag=10)
    steps = cache_steps(T, cfg)
    x_or = torch.randn(args.n_oracle, 2, device=dev)
    x_cal = torch.randn(args.n_cal, 2, device=dev)
    x_test = torch.randn(args.n_test, 2, device=dev)
    res = {"args": vars(args), "oracle": [], "slack": {}, "muninn": []}

    for name, (model, make) in setups.items():
        print(f"== {LABELS[name]}")
        # 1. oracle: does the paper's per-sample certificate hold?
        states = None
        for eta in args.etas:
            d, bc, ba, nr, st = oracle_paper(model, make(1), x_or, eta, cfg)
            states = states or st
            moved = d > 1e-6
            row = {"sampler": name, "eta": eta, "reuse_steps_mean": nr.mean().item(),
                   "holds_claim": (d <= bc + 1e-6).float().mean().item(),
                   "holds_all_steps": (d <= ba + 1e-6).float().mean().item(),
                   "median_d_over_claim": (d[moved] / bc[moved].clamp_min(1e-12)).median().item() if moved.any() else 0.0}
            res["oracle"].append(row)
            print(f"  oracle eta={eta:<4} reuse {row['reuse_steps_mean']:5.1f} steps | d <= paper certificate: "
                  f"{row['holds_claim']:.3f} | d <= bound over all steps: {row['holds_all_steps']:.4f} | "
                  f"median d/certificate {row['median_d_over_claim']:.2f}")
        # 2. slack of the paper's weights along the full chain
        sr = slack_ratio(model, make(1), [s[:5000] for s in states])
        q = torch.tensor([0.5, 0.9, 0.99], device=dev)
        res["slack"][name] = {k: dict(zip(["median", "p90", "p99"], torch.quantile(sr[k], q).tolist()),
                                      max=sr[k].max().item()) for k in [5, 20, 40, 60, 80, 90]}
        print("  true flow Jacobian / paper's prod K (median/p90/p99/max): " +
              "  ".join(f"k={k}: {v['median']:.2f}/{v['p90']:.2f}/{v['p99']:.1f}/{v['max']:.0f}"
                        for k, v in res["slack"][name].items()))
        # 3. real Muninn
        pairs = collect_calibration_pairs(model, make(2), x_cal, steps, max_lag=10)
        bound = ConformalUpperBound().fit(pairs["s"], pairs["eps"], args.alpha / len(steps), seed=args.seed)
        solver = make(3)
        x_full = sample_full(model, solver, x_test)
        for eta in args.etas:
            x, info = muninn_sample(model, solver, x_test, bound, MuninnConfig(**{**cfg.__dict__, "eta": eta}))
            d = trajectory_distance(x_full, x)
            k = int((d > eta).sum())
            row = {"sampler": name, "eta": eta, "nfe": info["nfe"].float().mean().item(), "p_viol": k / args.n_test,
                   "p_viol_ci": wilson_ci(k, args.n_test),
                   "bound_holds": (d <= info["bound_used"] + 1e-6).float().mean().item()}
            res["muninn"].append(row)
            print(f"  Muninn eta={eta:<4} NFE {row['nfe']:5.1f} | p_viol {row['p_viol']:.4f} "
                  f"[{row['p_viol_ci'][0]:.4f}, {row['p_viol_ci'][1]:.4f}] | d <= sum c_hat: {row['bound_holds']:.3f}")

    out_dir = Path(args.ddpm_ckpt).parent / "e14"
    save_json(res, out_dir / "results.json")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    width = 0.27
    for i, name in enumerate(setups):
        rows = [r for r in res["oracle"] if r["sampler"] == name]
        xs = [j + (i - 1) * width for j in range(len(rows))]
        ax[0].bar(xs, [r["holds_claim"] for r in rows], width * 0.92, color=COLORS[name], label=LABELS[name])
    ax[0].set_xticks(range(len(args.etas)), [str(e) for e in args.etas])
    ax[0].axhline(1.0, color="#888888", lw=1, ls=":")
    ax[0].set_xlabel("oracle budget η"); ax[0].set_ylabel("fraction of samples with d ≤ paper certificate")
    ax[0].set_title("does the paper's per-sample certificate hold? (should be 1)", fontsize=10)
    ax[0].set_ylim(0, 1.22)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    ks = [5, 20, 40, 60, 80, 90]
    for name in setups:
        s = res["slack"][name]
        ax[1].plot(ks, [s[k]["median"] for k in ks], color=COLORS[name], marker="o", lw=2, label=f"{LABELS[name]}, median")
        ax[1].plot(ks, [s[k]["p99"] for k in ks], color=COLORS[name], marker="o", lw=2, ls="--", label=f"{LABELS[name]}, p99")
    ax[1].axhline(1.0, color="#888888", lw=1, ls=":")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("step k (0 = noise)"); ax[1].set_ylabel("true flow Jacobian / paper's ∏K")
    ax[1].set_title("> 1: paper under-counts propagation, < 1: slack", fontsize=10)
    ax[1].legend(frameon=False, fontsize=7)
    for a in ax:
        a.grid(True, axis="y", color="#e5e5e5", lw=0.8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("E14 (checkerboard): the certificate gap on the paper's own samplers")
    fig.tight_layout()
    fig.savefig(out_dir / "e14.png", dpi=150)
    print(f"wrote {out_dir / 'results.json'} and {out_dir / 'e14.png'}")


if __name__ == "__main__":
    main()

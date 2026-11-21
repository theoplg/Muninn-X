"""E7: numerical check of the corrected error recursion (THEORY_NOTES.md).

    python scripts/theory_check.py --ckpt runs/toy_checkerboard/model.pt

1. Run the full Euler chain x_k and an oracle-reuse chain x~_k (same noise, E2 decision rule),
   recording the injected errors e_k = v(x~_k) - u_k  (zero at recompute steps).
2. Network Jacobians J_k = dv/dx along the full chain -> flow Jacobians
   G_{k+1->N} = (I + h J_{N-1}) ... (I + h J_{k+1}).
3. Compare, per sample, the actual deviation d = ||x_N - x~_N|| with
     paper bound      B_paper = sum_k h ||e_k||
     corrected bound  B_corr  = sum_k h ||G_{k+1->N}||_op ||e_k||
     linear predictor Delta_lin = sum_k h G_{k+1->N} e_k     (vector)
4. Tweedie check: at a few t, compare the network Jacobian with the closed form
     J* = [ t/(1-t)^2 Cov(x1 | x_t) - I ] / (1-t),
   where the posterior over x1 is estimated by self-normalised Monte Carlo over data samples.
Writes <ckpt dir>/theory/results.json and theory.png.
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

from muninnx.cache.bound import sensitivity_weights
from muninnx.cache.controller import MuninnConfig, cache_steps
from muninnx.data.toy2d import sample_toy
from muninnx.models.mlp import VelocityMLP
from muninnx.solvers.euler import EulerSolver
from muninnx.utils.misc import get_device, load_checkpoint, save_json, seed_all

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#888888"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])


def jacobian(model, x, t):
    """(B, D, D) with J[b, i, j] = d v_i / d x_j (the MLP is per-sample, so summing over b is exact)."""
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        v = model(x, t)
        rows = [torch.autograd.grad(v[:, i].sum(), x, retain_graph=i < v.shape[1] - 1)[0]
                for i in range(v.shape[1])]
    return torch.stack(rows, dim=1)


@torch.no_grad()
def oracle_reuse_chain(model, solver, x0, cfg):
    """E2 oracle (reuse v, true error decides). Returns x~_N, injected errors e (N, B, D), NFE, states x~_k."""
    N = solver.n_steps
    T = set(cache_steps(N, cfg))
    L = sensitivity_weights(solver).tolist()
    B = x0.shape[0]
    x, v_cache = x0, None
    budget = torch.full((B,), float(cfg.eta), device=x0.device)
    lag = torch.zeros(B, dtype=torch.long, device=x0.device)
    nfe = torch.zeros(B, dtype=torch.long, device=x0.device)
    errs, states = [], []
    for k in range(N):
        states.append(x)
        v_true = model(x, solver.t(k))
        if v_cache is None or k not in T:
            rec = torch.ones(B, dtype=torch.bool, device=x.device)
            c = torch.zeros(B, device=x.device)
        else:
            c = cfg.gamma * L[k] * (v_true - v_cache).norm(dim=1)
            rec = (c > budget) | (lag >= cfg.max_lag)
        v_cache = v_true if v_cache is None else torch.where(rec[:, None], v_true, v_cache)
        errs.append(v_true - v_cache)  # e_k = v(x~_k) - u_k, zero where recomputed
        nfe += rec.long()
        lag = torch.where(rec, torch.zeros_like(lag), lag + 1)
        budget = torch.where(rec, budget, budget - c)
        x = solver.step(x, v_cache, k)
    return x, torch.stack(errs), nfe, states


def flow_op_norms(jacs, h):
    """jacs: list of N (B, D, D) Jacobians. Returns (N, B) with ||(I+hJ_{N-1})...(I+hJ_{k+1})||_op."""
    I = torch.eye(jacs[0].shape[-1], device=jacs[0].device).expand_as(jacs[0])
    P, out = I.clone(), [None] * len(jacs)
    for k in reversed(range(len(jacs))):
        out[k] = torch.linalg.matrix_norm(P, ord=2)
        P = P @ (I + h * jacs[k])
    return torch.stack(out)


def segment_jacobian(model, a, b, t, n=8):
    """Mean-value Jacobian: average of dv/dx along the segment from a to b (midpoint rule, n points)."""
    return sum(jacobian(model, a + (i + 0.5) / n * (b - a), t) for i in range(n)) / n


@torch.no_grad()
def posterior_moments(data, x, t, chunk=256):
    """Self-normalised MC estimate of E[x1 | x_t = x] and Cov[x1 | x_t = x] under x_t = t x1 + (1-t) x0."""
    means, covs, ess = [], [], []
    s2 = (1 - t) ** 2
    for xc in x.split(chunk):
        logw = -((xc[:, None, :] - t * data[None]) ** 2).sum(-1) / (2 * s2)  # (b, M)
        w = torch.softmax(logw, dim=1)
        m = w @ data
        diff = data[None] - m[:, None]
        cov = torch.einsum("bm,bmi,bmj->bij", w, diff, diff)
        means.append(m)
        covs.append(cov)
        ess.append(1.0 / (w ** 2).sum(1))
    return torch.cat(means), torch.cat(covs), torch.cat(ess)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--etas", type=float, nargs="+", default=[0.05, 0.3])
    p.add_argument("--forbid", type=int, default=5)
    p.add_argument("--max-lag", type=int, default=10)
    p.add_argument("--n-data", type=int, default=200000, help="MC data samples for the posterior")
    p.add_argument("--n-tweedie", type=int, default=2000)
    p.add_argument("--tweedie-ks", type=int, nargs="+", default=[10, 30, 50, 70, 85])
    p.add_argument("--map-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_all(args.seed)
    dev = get_device(args.device)
    model, ckpt_cfg = load_checkpoint(args.ckpt, VelocityMLP, dev)
    model.eval()
    dataset = ckpt_cfg["dataset"]
    N, S = args.n_steps, EulerSolver(args.n_steps)
    h = 1.0 / N
    out_dir = Path(args.ckpt).parent / "theory"
    results = {"args": vars(args), "dataset": dataset}

    # ---- full chain + Jacobians along it ----
    x0 = torch.randn(args.n, 2, device=dev)
    xs = [x0]
    with torch.no_grad():
        for k in range(N):
            xs.append(S.step(xs[-1], model(xs[-1], S.t(k)), k))
    I = torch.eye(2, device=dev).expand(args.n, 2, 2)
    A = [I + h * jacobian(model, xs[k], S.t(k)) for k in range(N)]
    G = [None] * N  # G[k] = A_{N-1} ... A_{k+1}
    P = I.clone()
    for k in reversed(range(N)):
        G[k] = P
        P = P @ A[k]
    G = torch.stack(G)  # (N, B, 2, 2)
    G_op = torch.linalg.matrix_norm(G, ord=2)  # (N, B)

    # ---- bounds vs actual deviation ----
    base = MuninnConfig(eta=0.0, alpha=0.05, forbid_prefix=args.forbid, forbid_suffix=args.forbid,
                        max_lag=args.max_lag)
    bound_rows, scatter = [], None
    for eta in args.etas:
        cfg = MuninnConfig(**{**base.__dict__, "eta": eta})
        x_cached, e, nfe, xt = oracle_reuse_chain(model, S, x0, cfg)
        delta = xs[-1] - x_cached
        d = delta.norm(dim=1)
        e_norm = e.norm(dim=2)                                    # (N, B)
        b_paper = (h * e_norm).sum(0)
        b_corr = (h * G_op * e_norm).sum(0)
        G_mv = flow_op_norms([segment_jacobian(model, xt[k], xs[k], S.t(k)) for k in range(N)], h)
        b_mv = (h * G_mv * e_norm).sum(0)  # exact recursion: Jacobians averaged between the two chains
        delta_lin = (h * (G @ e[..., None]).squeeze(-1)).sum(0)  # (B, 2)
        moved = d > 1e-6
        rel_lin = ((delta_lin - delta).norm(dim=1)[moved] / d[moved])
        row = {
            "eta": eta, "nfe_mean": nfe.float().mean().item(),
            "E_d": d.mean().item(),
            "holds_paper": (d <= b_paper + 1e-6).float().mean().item(),
            "holds_corr": (d <= b_corr + 1e-6).float().mean().item(),
            "holds_meanvalue": (d <= b_mv + 1e-6).float().mean().item(),
            "median_d_over_paper": (d[moved] / b_paper[moved]).median().item(),
            "median_d_over_corr": (d[moved] / b_corr[moved]).median().item(),
            "linear_pred_rel_err_median": rel_lin.median().item(),
            "linear_pred_rel_err_p90": rel_lin.quantile(0.9).item(),
        }
        bound_rows.append(row)
        print(f"eta={eta:<5} NFE {row['nfe_mean']:5.1f} | d<=B_paper {row['holds_paper']:.3f} "
              f"d<=B_corr {row['holds_corr']:.3f} d<=B_meanvalue {row['holds_meanvalue']:.4f} | median d/B_paper {row['median_d_over_paper']:.2f} "
              f"d/B_corr {row['median_d_over_corr']:.2f} | linear pred rel err median "
              f"{row['linear_pred_rel_err_median']:.3f} p90 {row['linear_pred_rel_err_p90']:.3f}")
        if scatter is None:
            idx = torch.randperm(args.n, device=dev)[:3000]
            scatter = {"eta": eta, "d": d[idx].cpu(), "paper": b_paper[idx].cpu(), "corr": b_corr[idx].cpu()}
    results["bounds"] = bound_rows

    # ---- G statistics over k ----
    q = torch.tensor([0.5, 0.9, 0.99], device=dev)
    g_rows = []
    for k in [5, 20, 40, 60, 80, 90, 94]:
        qs = torch.quantile(G_op[k], q).tolist()
        g_rows.append({"k": k, "median": qs[0], "p90": qs[1], "p99": qs[2], "max": G_op[k].max().item()})
    results["G_op"] = g_rows
    print("||G_{k+1->N}||_op  " + "  ".join(f"k={r['k']}: {r['median']:.2f}/{r['p90']:.2f}/{r['p99']:.1f}"
                                            for r in g_rows))

    # ---- Tweedie: network Jacobian vs closed form from the data posterior ----
    data = sample_toy(dataset, args.n_data, device=dev)
    tw_rows = []
    for k in args.tweedie_ks:
        t = S.t(k)
        xq = xs[k][: args.n_tweedie]
        J_net = jacobian(model, xq, t)
        m, cov, ess = posterior_moments(data, xq, t)
        J_star = (t / (1 - t) ** 2 * cov - torch.eye(2, device=dev)) / (1 - t)
        lam_net = torch.linalg.eigvalsh(0.5 * (J_net + J_net.transpose(1, 2)))[:, -1]
        lam_star = torch.linalg.eigvalsh(J_star)[:, -1]
        with torch.no_grad():
            v_net = model(xq, t)
        v_star = (m - xq) / (1 - t)
        r = {
            "k": k, "t": t,
            "lam_net_median": lam_net.median().item(), "lam_net_p90": lam_net.quantile(0.9).item(),
            "lam_star_median": lam_star.median().item(), "lam_star_p90": lam_star.quantile(0.9).item(),
            "lam_corr": torch.corrcoef(torch.stack([lam_net, lam_star]))[0, 1].item(),
            "J_rel_err_median": ((J_net - J_star).flatten(1).norm(dim=1)
                                 / J_star.flatten(1).norm(dim=1).clamp_min(1e-6)).median().item(),
            "v_rel_err_median": ((v_net - v_star).norm(dim=1) / v_star.norm(dim=1).clamp_min(1e-6)).median().item(),
            "frac_expanding_net": (lam_net > 0).float().mean().item(),
            "frac_expanding_star": (lam_star > 0).float().mean().item(),
            "ess_median": ess.median().item(),
        }
        tw_rows.append(r)
        print(f"t={t:.2f} lam_max net {r['lam_net_median']:+.2f}/{r['lam_net_p90']:+.2f}  "
              f"Tweedie {r['lam_star_median']:+.2f}/{r['lam_star_p90']:+.2f}  corr {r['lam_corr']:.2f} | "
              f"rel err J {r['J_rel_err_median']:.2f} v {r['v_rel_err_median']:.3f} | expanding "
              f"{r['frac_expanding_net']:.2f} vs {r['frac_expanding_star']:.2f} | ESS {r['ess_median']:.0f}")
    results["tweedie"] = tw_rows
    save_json(results, out_dir / "results.json")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    a = ax[0]
    a.scatter(scatter["paper"], scatter["d"], s=4, alpha=0.35, color=ORANGE, marker="o", label="paper bound  Σ h‖e_k‖")
    a.scatter(scatter["corr"], scatter["d"], s=4, alpha=0.35, color=BLUE, marker="^", label="corrected  Σ h‖G‖‖e_k‖")
    lo = max(1e-5, float(torch.cat([scatter["paper"], scatter["d"]]).clamp_min(1e-5).min()))
    hi = float(torch.cat([scatter["corr"], scatter["d"]]).max())
    a.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls=":")
    a.text(hi, hi, " d = bound", fontsize=8, color="#555555", ha="right", va="bottom")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("bound"); a.set_ylabel("actual deviation d")
    a.set_title(f"bound holds iff point is below the line (oracle, η={scatter['eta']})", fontsize=9)
    a.legend(frameon=False, fontsize=8, markerscale=3)

    a = ax[1]
    xk = xs[args.map_k].cpu()
    g = G_op[args.map_k].clamp_min(1e-3).log10().cpu()
    sc = a.scatter(xk[:, 0], xk[:, 1], c=g, s=2, cmap=SEQ, vmin=0, vmax=max(1.0, float(g.quantile(0.995))))
    fig.colorbar(sc, ax=a, label=f"log10 ‖G_{{{args.map_k + 1}→N}}‖")
    a.set_aspect("equal")
    a.set_title(f"amplification of an error made at t={S.t(args.map_k):.2f}, by position", fontsize=9)

    a = ax[2]
    ts = [r["t"] for r in tw_rows]
    a.plot(ts, [r["lam_net_median"] for r in tw_rows], color=BLUE, marker="o", lw=2, label="network, median")
    a.plot(ts, [r["lam_net_p90"] for r in tw_rows], color=BLUE, marker="o", lw=2, ls="--", label="network, p90")
    a.plot(ts, [r["lam_star_median"] for r in tw_rows], color=ORANGE, marker="s", lw=2, label="Tweedie (data), median")
    a.plot(ts, [r["lam_star_p90"] for r in tw_rows], color=ORANGE, marker="s", lw=2, ls="--", label="Tweedie (data), p90")
    a.axhline(0, color=GREY, lw=1, ls=":")
    a.set_xlabel("t"); a.set_ylabel("largest eigenvalue of ∂v/∂x")
    a.set_title(">0: errors grow, <0: errors shrink", fontsize=9)
    a.legend(frameon=False, fontsize=8)
    for a in ax:
        a.grid(True, color="#e5e5e5", lw=0.8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"{dataset}: why the Muninn certificate breaks", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "theory.png", dpi=150)
    print(f"wrote {out_dir / 'results.json'} and {out_dir / 'theory.png'}")


if __name__ == "__main__":
    main()

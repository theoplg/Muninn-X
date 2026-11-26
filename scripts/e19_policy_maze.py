"""E19: the E10 learned, G-aware extrapolation policy on a real planner (conditional Maze2D U-maze).

    python scripts/e19_policy_maze.py --ckpt runs/maze2d_umaze/maze_unet_cond/model.pt

Same pipeline as E10 (scripts/e10_predictability.py), through the task layer:
1. Training set: "ghost" chains (extrapolate-or-compute on an AB2-100 grid, random decisions with prob.
   --p-extrap, lag <= 10) on separate noise + contexts. At every step k >= 2 we record each tier's features
   and the targets
       log ||e_k||                         e_k = v(x_k) - v_ext (true extrapolation error)
       log ||G_{k+1->N} e_k|| / ||e_k||    directional amplification of that error
       log c_k = log h ||G_{k+1->N} e_k||  the G-weighted cost of the E9 oracle
   G_{k+1->N} = (I + h J_{N-1}) ... (I + h J_{k+1}) along the AB2-100 chain from the same noise. The state has
   D = 774 dims, so G is never formed: each e_k is pushed through the remaining steps by Jacobian-vector
   products (central finite differences, step 0.01, full fp32: propagated ||G e|| within 0.2% of torch.func.jvp).
2. One regressor per tier predicts log c_k (G-aware); "free_plain" predicts log h ||e_k|| (not G-aware).
3. Decision: extrapolate iff predicted cost <= tau; tau certified end-to-end exactly as in E9 / E10
   (Bonferroni LTT, exact-ODE reference, alpha = 0.05), compared with certified AB2 on the same noise.
   Cost = NFE, + probe cost per extrapolated step for the probe tiers.
The G-aware *oracle* of E9 is not run: it needs O(N^2) JVPs per decision run (one per future step).
Writes <ckpt dir>/e19/seed<s>/results.json and e19.png.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e10_predictability import ab2_states, fit_regressor, spearman  # noqa: E402

from muninnx.baselines.fewsteps import bonferroni_select, sample_ab2, sample_rk4
from muninnx.cache.extrap_policy import run_extrap_ab2
from muninnx.tasks import load_task
from muninnx.utils.misc import get_device, save_json, seed_all

TIERS = ("free", "probe_score", "probe_feat")  # "position" (the full 774-d state) is a 2D-only upper bound
STYLE = {
    "ab2":         dict(color="#2a78d6", marker="o", ls="-",  label="certified FewSteps AB2"),
    "free_plain":  dict(color="#888888", marker="x", ls=":",  label="predicted h‖e‖ (not G-aware), free signals"),
    "free":        dict(color="#eb6834", marker="s", ls="-",  label="predicted G-cost: free signals"),
    "probe_score": dict(color="#1baf7a", marker="^", ls="-",  label="predicted G-cost: + probe score"),
    "probe_feat":  dict(color="#eda100", marker="D", ls="-",  label="predicted G-cost: + probe features"),
}


@torch.no_grad()
def jvp_fd(model, x, w, t, eps=0.01, chunk=4000):
    """J(x) w per row by central differences along w / ||w|| (step eps), in chunks."""
    out = torch.empty_like(w)
    for s in range(0, x.shape[0], chunk):
        xs, ws = x[s:s + chunk], w[s:s + chunk]
        n = ws.norm(dim=1, keepdim=True).clamp_min(1e-30)
        u = ws / n
        out[s:s + chunk] = (model(xs + eps * u, t) - model(xs - eps * u, t)) / (2 * eps) * n
    return out


@torch.no_grad()
def propagate(model, xs, ts, h, E, eps=0.01):
    """E: dict k -> (B, D) error introduced just after step k. Returns dict k -> (B,) ||G_{k+1->N} e_k||."""
    B = xs[0].shape[0]
    W, owner = None, []
    for j in range(len(xs)):
        if W is not None:
            rows = torch.arange(W.shape[0], device=W.device) % B
            W = W + h * jvp_fd(model, xs[j][rows], W, ts[j], eps)
        if j in E:
            W = E[j].clone() if W is None else torch.cat([W, E[j]])
            owner.append(j)
    return {k: W[i * B:(i + 1) * B].norm(dim=1) for i, k in enumerate(owner)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-ref", type=int, default=400)
    p.add_argument("--n-fit", type=int, default=2000)
    p.add_argument("--n-cal", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=20000)
    p.add_argument("--p-extrap", type=float, default=0.7)
    p.add_argument("--etas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--max-lag", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    # full fp32 everywhere: cuDNN's default TF32 convolutions add ~1e-3 relative noise to the velocity,
    # which makes finite-difference Jacobians meaningless
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(args.seed)
    dev = get_device(args.device)
    task = load_task(args.ckpt, dev)
    model = task.model
    N, h = args.n_steps, 1.0 / args.n_steps
    ts = [k * h for k in range(N)]
    c_probe = task.probe_cost()
    x0 = {s: task.noise(n, dev) for s, n in (("fit", args.n_fit), ("cal", args.n_cal), ("test", args.n_test))}
    ref = {s: sample_rk4(model, args.n_ref, x0[s]) for s in ("cal", "test")}
    print(f"{task.name}: D = {task.D}, probe cost {c_probe:.3f}", flush=True)

    # ---- 1. training set from ghost chains ----
    feats = {tier: [] for tier in TIERS}
    E, log_e = {}, {}
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)

    def record(k, ctx):
        e = ctx["v_true"] - ctx["v_ext"]
        E[k] = e
        log_e[k] = torch.log(e.norm(dim=1) + 1e-12)
        for tier in TIERS:
            feats[tier].append(ctx["feats"][tier])

    run_extrap_ab2(model, x0["fit"], N,
                   lambda k, c: torch.rand(c["x"].shape[0], device=dev, generator=gen) < args.p_extrap,
                   args.max_lag, need_probe=True, record=record)
    Ge = propagate(model, ab2_states(model, x0["fit"], N), ts, h, E)
    ks = sorted(E)
    del E
    targ = {"e": torch.cat([log_e[k] for k in ks]),
            "c": torch.cat([torch.log(h * Ge[k] + 1e-12) for k in ks])}
    targ["gain"] = targ["c"] - torch.log(torch.tensor(h)) - targ["e"]
    targ["plain"] = torch.log(torch.tensor(h)) + targ["e"]
    feats = {k: torch.cat(v) for k, v in feats.items()}
    n_rows = targ["c"].shape[0]
    perm = torch.randperm(n_rows, device=dev)
    tr, ho = perm[: int(0.8 * n_rows)], perm[int(0.8 * n_rows):]
    g = targ["gain"].exp()
    print(f"training rows: {n_rows} | amplification ||G e||/||e||: median {g.median():.2f}, "
          f"p99 {g.quantile(0.99):.2f}, max {g.max():.1f}", flush=True)

    # ---- 2. predictors + held-out prediction quality ----
    regs, quality = {}, {}
    for tier in TIERS:
        regs[tier] = fit_regressor(feats[tier][tr], targ["c"][tr], seed=args.seed)
        with torch.no_grad():
            pred = regs[tier](feats[tier][ho])
        regG = fit_regressor(feats[tier][tr], targ["gain"][tr], seed=args.seed)
        with torch.no_grad():
            predG = regG(feats[tier][ho])
        quality[tier] = {"spearman_c": spearman(pred, targ["c"][ho]),
                         "spearman_vs_e": spearman(pred, targ["e"][ho]),
                         "spearman_vs_gain": spearman(pred, targ["gain"][ho]),
                         "spearman_gain_direct": spearman(predG, targ["gain"][ho]),
                         "rmse_log_c": (pred - targ["c"][ho]).pow(2).mean().sqrt().item()}
        q = quality[tier]
        print(f"  {tier:12s} Spearman(pred, true cost) {q['spearman_c']:.3f} | vs error {q['spearman_vs_e']:.3f} | "
              f"vs gain {q['spearman_vs_gain']:.3f} | dedicated gain predictor {q['spearman_gain_direct']:.3f} | "
              f"RMSE log-cost {q['rmse_log_c']:.2f}", flush=True)
    regs["free_plain"] = fit_regressor(feats["free"][tr], targ["plain"][tr], seed=args.seed)
    quality["_targets"] = {"spearman_c_vs_e": spearman(targ["c"][ho], targ["e"][ho]),
                           "spearman_c_vs_gain": spearman(targ["c"][ho], targ["gain"][ho])}
    print(f"  true cost vs error {quality['_targets']['spearman_c_vs_e']:.3f}, "
          f"vs gain {quality['_targets']['spearman_c_vs_gain']:.3f}", flush=True)
    del feats, targ

    # ---- 3. decision quality, certified end-to-end ----
    taus = [0.0] + torch.logspace(-7, -1, 31).tolist()
    feat_of = {"free_plain": "free"}

    def run(method, prm, split):
        if method == "ab2":
            return sample_ab2(model, prm, x0[split]), float(prm)
        reg, tier = regs[method], feat_of.get(method, method)
        dec = lambda k, c: torch.exp(reg(c["feats"][tier])) <= prm
        probe = tier.startswith("probe")
        x, nfe, n_ext = run_extrap_ab2(model, x0[split], N, dec, args.max_lag, need_probe=probe)
        cost = nfe.float() + (c_probe * n_ext.float() if probe else 0.0)
        return x, cost.mean().item()

    candidates = sorted(set(range(2, 101)) | set(range(110, 401, 10)), reverse=True)
    configs = {"ab2": candidates, "free_plain": taus, **{t: taus for t in TIERS}}
    d_cal, cost_cal = {}, {}
    for m, params in configs.items():
        d_cal[m], cost_cal[m] = [], []
        for prm in params:
            x, cost = run(m, prm, "cal")
            d_cal[m].append(task.dist(ref["cal"], x).cpu())
            cost_cal[m].append(cost)
            del x
        print(f"calibrated {m}", flush=True)

    rows, cache = [], {}
    for eta in args.etas:
        line = f"eta={eta:<6}"
        for m, params in configs.items():
            i = bonferroni_select(d_cal[m], cost_cal[m], eta, args.alpha, args.delta)
            row = {"eta": eta, "method": m, "param": None if i is None else params[i]}
            if i is not None:
                if (m, params[i]) not in cache:
                    x, cost = run(m, params[i], "test")
                    cache[(m, params[i])] = (task.dist(ref["test"], x), cost)
                    del x
                d, cost = cache[(m, params[i])]
                row.update(cost=cost, p_viol=(d > eta).float().mean().item(), E_d=d.mean().item())
            rows.append(row)
            line += f" | {m}: " + ("inf" if i is None else f"{row['cost']:5.1f} (pv {row['p_viol']:.3f})")
        print(line, flush=True)

    out_dir = Path(args.ckpt).parent / "e19" / f"seed{args.seed}"
    save_json({"args": vars(args), "probe_cost": c_probe, "n_rows": n_rows, "quality": quality, "results": rows,
               "amplification": {"median": g.median().item(), "p99": g.quantile(0.99).item(), "max": g.max().item()}},
              out_dir / "results.json")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    for m, st in STYLE.items():
        pts = sorted((r for r in rows if r["method"] == m and r["param"] is not None), key=lambda r: r["eta"])
        ax[0].plot([r["eta"] for r in pts], [r["cost"] for r in pts], lw=2, ms=6, **st)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(f"tolerance η  (certified: P(d > η) ≤ {args.alpha}, LTT + Bonferroni)")
    ax[0].set_ylabel("mean cost (NFE + probe), lower is better")
    ax[0].set_title("compute for the same certified guarantee", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    xs = range(len(TIERS))
    w = 0.38
    ax[1].bar([i - w / 2 for i in xs], [quality[n]["spearman_c"] for n in TIERS], w * 0.92, color="#2a78d6",
              label="cost predictor vs true cost (what decides)")
    ax[1].bar([i + w / 2 for i in xs], [quality[n]["spearman_gain_direct"] for n in TIERS], w * 0.92,
              color="#eb6834", label="dedicated predictor of the amplification")
    ax[1].set_xticks(list(xs), ["free", "+probe\nscore", "+probe\nfeatures"])
    ax[1].set_ylim(0, 1.32)
    ax[1].set_ylabel("held-out Spearman rank correlation")
    ax[1].set_title("held-out prediction quality", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8, loc="upper center")
    for a in ax:
        a.grid(True, which="both", axis="y", color="#e5e5e5", lw=0.8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"{task.name}: learned G-aware extrapolation policy vs certified AB2")
    fig.tight_layout()
    fig.savefig(out_dir / "e19.png", dpi=150)
    print(f"wrote {out_dir / 'results.json'} and {out_dir / 'e19.png'}")


if __name__ == "__main__":
    main()

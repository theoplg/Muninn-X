"""E10: can a real sampler predict the cost that made the E9 oracle win?

    python scripts/e10_predictability.py --ckpt runs/toy_checkerboard/model.pt

1. Training set: "ghost" chains (extrapolate-or-compute on an AB2-100 grid, random decisions: extrapolate
   with prob. --p-extrap, lag <= 10) on separate noise. At every step k >= 2 we record the features of each
   tier (muninnx.cache.extrap_policy) and the oracle targets:
       log ||e_k||            (true extrapolation error, e_k = v(x_k) - v_ext)
       log ||G_{k+1->N}||     (amplification, flow Jacobian along the AB2-100 reference chain)
       log c_k = log h ||G_{k+1->N} e_k||   (the cost the E9 oracle used)
2. One MLP regressor per tier predicts log c_k; held-out Spearman for the three targets.
3. Decision quality: extrapolate iff predicted cost <= tau, certified end-to-end exactly as in E9
   (Bonferroni LTT, exact-ODE reference, alpha = 0.05) and compared with certified AB2 and the oracle.
   Cost = NFE, plus c_probe per extrapolated step for the probe tiers (the probe is the first layer, so
   it is free on computed steps).
Writes <ckpt dir>/e10/results.json and e10.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from muninnx.analysis.flow import flow_jacobians
from muninnx.baselines.fewsteps import bonferroni_select, sample_ab2, sample_rk4
from muninnx.cache.extrap_policy import TIERS, run_extrap_ab2
from muninnx.eval.paired import trajectory_distance
from muninnx.models.mlp import VelocityMLP
from muninnx.utils.misc import get_device, load_checkpoint, save_json, seed_all

STYLE = {
    "ab2":         dict(color="#2a78d6", marker="o", ls="-",  label="certified FewSteps AB2"),
    "oracle":      dict(color="#888888", marker="x", ls=":",  label="oracle (true cost, E9)"),
    "free":        dict(color="#eb6834", marker="s", ls="-",  label="predicted: free signals"),
    "probe_score": dict(color="#1baf7a", marker="^", ls="-",  label="predicted: + probe score"),
    "probe_feat":  dict(color="#eda100", marker="D", ls="-",  label="predicted: + probe features"),
    "position":    dict(color="#e87ba4", marker="v", ls="--", label="predicted: + position (2D only)"),
}


@torch.no_grad()
def ab2_states(model, x0, n):
    h, x, v_prev, xs = 1.0 / n, x0, None, []
    for k in range(n):
        xs.append(x)
        v = model(x, k * h)
        x = x + h * (v if v_prev is None else 1.5 * v - 0.5 * v_prev)
        v_prev = v
    return xs


def probe_cost(model):
    macs = lambda mod: sum(m.in_features * m.out_features for m in mod.modules() if isinstance(m, nn.Linear))
    return macs(model.stem) / macs(model)


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


class Regressor(nn.Module):
    def __init__(self, d_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))

    def forward(self, f):
        return self.net((f - self.mu) / self.sd).squeeze(-1)


def fit_regressor(X, y, iters=4000, batch=16384, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    reg = Regressor(X.shape[1]).to(X.device)
    reg.mu.copy_(X.mean(0))
    reg.sd.copy_(X.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(reg.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        idx = torch.randint(0, X.shape[0], (batch,), device=X.device)
        loss = nn.functional.mse_loss(reg(X[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return reg.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-ref", type=int, default=400)
    p.add_argument("--n-fit", type=int, default=4000)
    p.add_argument("--n-cal", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=50000)
    p.add_argument("--p-extrap", type=float, default=0.7)
    p.add_argument("--etas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--max-lag", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_all(args.seed)
    dev = get_device(args.device)
    model, _ = load_checkpoint(args.ckpt, VelocityMLP, dev)
    N, h = args.n_steps, 1.0 / args.n_steps
    ts = [k * h for k in range(N)]
    c_probe = probe_cost(model)
    x0 = {s: torch.randn(n, 2, device=dev) for s, n in
          (("fit", args.n_fit), ("cal", args.n_cal), ("test", args.n_test))}
    G = {s: flow_jacobians(model, ab2_states(model, x0[s], N), ts, h) for s in ("fit", "cal", "test")}
    ref = {s: sample_rk4(model, args.n_ref, x0[s]) for s in ("cal", "test")}

    # ---- 1. training set from ghost chains ----
    feats = {tier: [] for tier in TIERS}
    targ = {"e": [], "G": [], "c": []}
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)

    def record(k, ctx):
        e = ctx["v_true"] - ctx["v_ext"]
        Ge = (G["fit"][k] @ e[..., None]).squeeze(-1)
        targ["e"].append(torch.log(e.norm(dim=1) + 1e-12))
        targ["G"].append(torch.log(torch.linalg.matrix_norm(G["fit"][k], ord=2)))
        targ["c"].append(torch.log(h * Ge.norm(dim=1) + 1e-12))
        for tier in TIERS:
            feats[tier].append(ctx["feats"][tier])

    run_extrap_ab2(model, x0["fit"], N,
                   lambda k, c: torch.rand(c["x"].shape[0], device=dev, generator=gen) < args.p_extrap,
                   args.max_lag, need_probe=True, record=record)
    targ = {k: torch.cat(v) for k, v in targ.items()}
    feats = {k: torch.cat(v) for k, v in feats.items()}
    n_rows = targ["c"].shape[0]
    perm = torch.randperm(n_rows, device=dev)
    tr, ho = perm[: int(0.8 * n_rows)], perm[int(0.8 * n_rows):]
    print(f"training rows: {n_rows}  (features: " + ", ".join(f"{t}={feats[t].shape[1]}" for t in TIERS) + ")")

    # ---- 2. predictors + held-out prediction quality ----
    regs, quality = {}, {}
    for tier in TIERS:
        regs[tier] = fit_regressor(feats[tier][tr], targ["c"][tr], seed=args.seed)
        with torch.no_grad():
            pred = regs[tier](feats[tier][ho])
        quality[tier] = {"spearman_c": spearman(pred, targ["c"][ho]),
                         "spearman_vs_e": spearman(pred, targ["e"][ho]),
                         "spearman_vs_G": spearman(pred, targ["G"][ho]),
                         "rmse_log_c": (pred - targ["c"][ho]).pow(2).mean().sqrt().item()}
        print(f"  {tier:12s} held-out Spearman(pred, true cost) {quality[tier]['spearman_c']:.3f} | "
              f"vs error {quality[tier]['spearman_vs_e']:.3f} | vs amplification {quality[tier]['spearman_vs_G']:.3f} | "
              f"RMSE log-cost {quality[tier]['rmse_log_c']:.2f}")
        regG = fit_regressor(feats[tier][tr], targ["G"][tr], seed=args.seed)
        with torch.no_grad():
            quality[tier]["spearman_G_direct"] = spearman(regG(feats[tier][ho]), targ["G"][ho])
        print(f"  {'':12s} predictor trained on amplification alone: held-out Spearman {quality[tier]['spearman_G_direct']:.3f}")
    quality["_targets"] = {"spearman_c_vs_e": spearman(targ["c"][ho], targ["e"][ho]),
                           "spearman_c_vs_G": spearman(targ["c"][ho], targ["G"][ho])}

    # ---- 3. decision quality, certified end-to-end ----
    taus = [0.0] + torch.logspace(-7, -1, 31).tolist()

    def run(method, prm, split):
        if method == "ab2":
            return sample_ab2(model, prm, x0[split]), float(prm)
        if method == "oracle":
            dec = lambda k, c: (h * (G[split][k] @ (c["v_true"] - c["v_ext"])[..., None]).squeeze(-1)
                                .norm(dim=1)) <= prm
            x, nfe, _ = run_extrap_ab2(model, x0[split], N, dec, args.max_lag, need_true=True)
            return x, nfe.float().mean().item()
        reg = regs[method]
        dec = lambda k, c: torch.exp(reg(c["feats"][method])) <= prm
        x, nfe, n_ext = run_extrap_ab2(model, x0[split], N, dec, args.max_lag,
                                       need_probe=method.startswith("probe"))
        cost = nfe.float() + (c_probe * n_ext.float() if method.startswith("probe") else 0.0)
        return x, cost.mean().item()

    candidates = sorted(set(range(2, 101)) | set(range(110, 401, 10)), reverse=True)
    configs = {"ab2": candidates, "oracle": taus, **{t: taus for t in TIERS}}
    d_cal, cost_cal = {}, {}
    for m, params in configs.items():
        d_cal[m], cost_cal[m] = [], []
        for prm in params:
            x, cost = run(m, prm, "cal")
            d_cal[m].append(trajectory_distance(ref["cal"], x).cpu())
            cost_cal[m].append(cost)

    rows, cache = [], {}
    for eta in args.etas:
        line = f"eta={eta:<6}"
        for m, params in configs.items():
            i = bonferroni_select(d_cal[m], cost_cal[m], eta, args.alpha, args.delta)
            row = {"eta": eta, "method": m, "param": None if i is None else params[i]}
            if i is not None:
                if (m, params[i]) not in cache:
                    cache[(m, params[i])] = run(m, params[i], "test")
                x, cost = cache[(m, params[i])]
                d = trajectory_distance(ref["test"], x)
                row.update(cost=cost, p_viol=(d > eta).float().mean().item())
            rows.append(row)
            line += f" | {m}: " + ("inf" if i is None else f"{row['cost']:5.1f} (pv {row['p_viol']:.3f})")
        print(line)

    out_dir = Path(args.ckpt).parent / "e10" / f"seed{args.seed}"
    save_json({"args": vars(args), "probe_cost": c_probe, "n_rows": n_rows, "quality": quality, "results": rows},
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
    names = list(TIERS)
    xs = range(len(names))
    w = 0.38
    ax[1].bar([i - w / 2 for i in xs], [quality[n]["spearman_c"] for n in names], w * 0.92, color="#2a78d6",
              label="cost predictor vs true cost (what decides)")
    ax[1].bar([i + w / 2 for i in xs], [quality[n]["spearman_G_direct"] for n in names], w * 0.92, color="#eb6834",
              label="dedicated predictor of the amplification ‖G‖")
    ax[1].set_xticks(list(xs), ["free", "+probe\nscore", "+probe\nfeatures", "+position"])
    ax[1].set_ylim(0, 1.32)
    ax[1].set_ylabel("held-out Spearman rank correlation")
    ax[1].set_title("held-out prediction quality", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8, loc="upper center")
    for a in ax:
        a.grid(True, which="both", axis="y", color="#e5e5e5", lw=0.8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"{Path(args.ckpt).parent.name}: can a real sampler reach the oracle's gain?")
    fig.tight_layout()
    fig.savefig(out_dir / "e10.png", dpi=150)
    print(f"wrote {out_dir / 'results.json'} and {out_dir / 'e10.png'}")


if __name__ == "__main__":
    main()

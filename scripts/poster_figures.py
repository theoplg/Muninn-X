"""Vector figures for the A1 portrait poster (poster/figures/*.pdf), sized for their printed width.

    python scripts/poster_figures.py            # all figures (the maze and map panels use the GPU)

Each figure is drawn at (about) its printed size, so its fonts print at their nominal size (Times-like serif,
matching the poster body). Method colours are the same in every figure:
    AB2 = blue, ours = orange, Muninn = aqua, Euler / ablations = greys.

  fig_pareto.pdf        E[d] vs NFE: FewSteps vs Muninn vs oracle, checkerboard (E2)
  fig_gnorm.pdf         amplification ||G_{k+1->N}|| along the chain, 2D toys (E7, runs/toy_*/theory)
  fig_tweedie.pdf       largest eigenvalue of dv/dx: network vs Tweedie formula (E7)
  fig_amplification.pdf map of ||G|| for an error made at t = 0.2, checkerboard (E7)
  fig_maze.pdf          U-maze plans of the conditional planner
  fig_e18.pdf           certified NFE vs tolerance: Muninn vs FewSteps Euler / AB2 (E18)
  fig_e19.pdf           certified NFE vs tolerance: our policy vs AB2 (E19, 3 seeds)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm

OUT = Path("poster/figures")
UMAZE = Path("runs/maze2d_umaze/maze_unet_cond")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, DGRAY, NAVY = "#9a9994", "#4d4c48", "#1f3a5f"
INK, INK2, GRID = "#111111", "#4a4a46", "#e6e6e3"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#e9f1fb", "#9cc0ee", BLUE, "#123f78"])
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 15, "axes.titlesize": 15.5, "axes.labelsize": 15.5,
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "legend.fontsize": 13,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "axes.linewidth": 0.9, "lines.linewidth": 1.8, "lines.markersize": 6,
    "pdf.fonttype": 42, "savefig.transparent": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "legend.frameon": False, "legend.handlelength": 2.0,
})


def style(ax, grid="both"):
    ax.grid(True, which="major", axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def log_eta_axis(ax, etas):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(etas, [f"{e:g}" for e in etas])
    ax.minorticks_off()


# ------------------------------------------------------------------------------------------- section 2
def fig_gnorm():
    fig, ax = plt.subplots(figsize=(13.0 * CM, 6.7 * CM))
    for ds, c, lab, xy in (("toy_checkerboard", NAVY, "checkerboard", (6, 9.5)),
                           ("toy_8gaussians", GRAY, "8 Gaussians", (6, 58))):
        G = json.load(open(f"runs/{ds}/theory/results.json"))["G_op"]
        k = [g["k"] for g in G]
        ax.plot(k, [g["median"] for g in G], color=c, marker="o")
        ax.plot(k, [g["p99"] for g in G], color=c, marker="o", ls="--", mfc="white")
        ax.text(*xy, lab, color=c, fontsize=13, ha="left", va="center")
    ax.plot([], [], color=INK2, marker="o", label="median")
    ax.plot([], [], color=INK2, marker="o", ls="--", mfc="white", label="99th percentile")
    ax.axhline(1, color=ORANGE, lw=1.6, ls=":")
    ax.text(2, 0.93, "value assumed by Eq. (1)", color=ORANGE, ha="left", va="top", fontsize=13)
    ax.set_yscale("log"); ax.set_ylim(0.62, 110)
    ax.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"]); ax.minorticks_off()
    ax.set_xlim(0, 100)
    ax.set_xlabel(r"step $k$ at which the error is made")
    ax.set_ylabel(r"$\Vert G_{k+1\to N}\Vert$")
    ax.legend(loc="upper right", fontsize=12)
    style(ax)
    fig.savefig(OUT / "fig_gnorm.pdf")


def fig_tweedie():
    fig, ax = plt.subplots(figsize=(13.0 * CM, 6.7 * CM))
    for ds, c, lab, xy in (("toy_checkerboard", NAVY, "checkerboard", (0.6, 1.95)),
                           ("toy_8gaussians", GRAY, "8 Gaussians", (0.31, 2.28))):
        T = json.load(open(f"runs/{ds}/theory/results.json"))["tweedie"]
        t = [r["t"] for r in T]
        ax.plot(t, [r["lam_net_median"] for r in T], color=c, marker="o")
        ax.plot(t, [r["lam_star_median"] for r in T], color=c, marker="s", ls="none", mfc="none", ms=11, mew=1.6)
        ax.text(*xy, lab, color=c, fontsize=13.5, ha="left", va="bottom")
    ax.plot([], [], color=INK2, marker="o", label="trained network")
    ax.plot([], [], color=INK2, marker="s", ls="none", mfc="none", ms=11, mew=1.6, label="Tweedie formula, Eq. (3)")
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xlim(0, 0.92); ax.set_ylim(-1.75, 2.75)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"median $\lambda_{\max}(\partial v/\partial x)$")
    ax.legend(loc="lower left", fontsize=12)
    style(ax)
    fig.savefig(OUT / "fig_tweedie.pdf")


def fig_amplification(dev):
    from muninnx.analysis.flow import flow_jacobians
    from muninnx.tasks import load_task
    model = load_task("runs/toy_checkerboard/model.pt", dev).model
    N, h, k = 100, 0.01, 20
    x = torch.randn(40000, 2, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    xs = []
    with torch.no_grad():
        for j in range(N):
            xs.append(x)
            x = x + h * model(x, j * h)
    G = flow_jacobians(model, xs, [j * h for j in range(N)], h)[k]
    amp = torch.linalg.matrix_norm(G, ord=2).cpu().numpy()
    pos, fin = xs[k].cpu().numpy(), x.cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(13.4 * CM, 6.9 * CM), gridspec_kw={"wspace": 0.05})
    norm = LogNorm(vmin=0.5, vmax=30)
    o = np.argsort(amp)  # amplified points on top
    for a, p, title in ((ax[0], pos, r"at $t = 0.2$"), (ax[1], fin, r"final sample")):
        sc = a.scatter(p[o, 0], p[o, 1], c=amp[o], cmap=SEQ, norm=norm, s=0.9, lw=0, rasterized=True)
        a.set_aspect("equal"); a.set_title(title, fontsize=13.5, pad=2)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    ax[0].set_xlim(-3.2, 3.2); ax[0].set_ylim(-3.2, 3.2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, shrink=0.9)
    cb.set_ticks([0.5, 1, 3, 10, 30], labels=["0.5", "1", "3", "10", "30"])
    cb.ax.tick_params(labelsize=12)
    cb.set_label(r"$\Vert G_{k+1\to N}\Vert$", fontsize=13.5)
    cb.outline.set_visible(False)
    fig.savefig(OUT / "fig_amplification.pdf", dpi=220)
    print(f"amplification at t=0.2: median {np.median(amp):.2f}, p99 {np.quantile(amp, 0.99):.1f}")


def fig_pareto():
    """E2: E[d] to the full Euler-100 chain vs mean NFE, checkerboard (Muninn's own target)."""
    r = json.load(open("runs/toy_checkerboard/oracle/results.json"))["results"]
    fig, ax = plt.subplots(figsize=(11.6 * CM, 7.4 * CM))
    few = sorted((x["nfe_mean"], x["E_d"]) for x in r if x["method"] == "fewsteps" and x["E_d"] > 0)
    ax.plot(*zip(*few), color=DGRAY, lw=2, label="FewSteps (uniform Euler)")
    for m, mk, fc, lab in (("muninn", "^", AQUA, "Muninn"), ("oracle_lag", "D", "white", "Muninn, exact errors")):
        pts = sorted((x["nfe_mean"], x["E_d"]) for x in r if x["method"] == m and x["E_d"] > 0)
        ax.plot(*zip(*pts), ls="none", marker=mk, ms=8, mfc=fc, mec=AQUA, mew=1.8, label=lab)
    ax.set_yscale("log"); ax.set_xlim(0, 102)
    ax.set_ylim(5e-4, 1.2)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1], ["0.001", "0.01", "0.1", "1"]); ax.minorticks_off()
    ax.set_xlabel("mean NFE per sample")
    ax.set_ylabel(r"$\mathbb{E}[d]$ to the full chain")
    ax.legend(loc="lower left", fontsize=12)
    style(ax)
    fig.savefig(OUT / "fig_pareto.pdf")


# ------------------------------------------------------------------------------------------- section 4
@torch.no_grad()
def fig_maze(dev):
    from muninnx.baselines.fewsteps import sample_rk4
    from muninnx.data.maze2d import Maze2DWindows, window_context
    from muninnx.data.maze_maps import MAPS
    from muninnx.tasks import load_task
    task = load_task(str(UMAZE / "model.pt"), dev)
    H = task.H
    ds = Maze2DWindows("umaze", H=H, device=dev)
    g = torch.Generator(device=dev).manual_seed(11)
    n = 6
    ctx = window_context(ds.sample(n, generator=g))
    z = torch.randn(n, task.Dx, device=dev, generator=g)
    x = sample_rk4(task.model, 100, torch.cat([z, ctx], 1))
    plan = ds.unnormalize(x[:, :task.Dx].reshape(n, H, 6)).cpu()
    fig, ax = plt.subplots(figsize=(7.6 * CM, 7.8 * CM))
    grid = MAPS["umaze"]
    Hc, Wc = len(grid), len(grid[0])
    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch == "#":
                ax.add_patch(plt.Rectangle((j - Wc / 2, Hc / 2 - i - 1), 1, 1, color="#cfcec9", lw=0))
    for tr in plan:
        ax.plot(tr[:, 0], tr[:, 1], color=NAVY, lw=1.5, alpha=0.85)
        ax.scatter(tr[0, 0], tr[0, 1], s=55, color=AQUA, edgecolor="white", lw=1, zorder=3)
        ax.scatter(tr[-1, 0], tr[-1, 1], s=140, marker="*", color=ORANGE, edgecolor="white", lw=0.8, zorder=3)
    ax.scatter([], [], s=55, color=AQUA, label=r"start $s_0$")
    ax.scatter([], [], s=140, marker="*", color=ORANGE, label=r"goal $g$")
    ax.set_aspect("equal"); ax.set_xlim(-Wc / 2, Wc / 2); ax.set_ylim(-Hc / 2, Hc / 2)
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2, fontsize=13, handletextpad=0.2)
    fig.savefig(OUT / "fig_maze.pdf")


def fig_e18():
    rows = json.load(open(UMAZE / "certified_compare/results.json"))["results"]
    sel = [r for r in rows if r["ref"] == "exact" and r["alpha"] == 0.05 and r["test"] == "ltt"]
    etas = sorted({r["eta"] for r in sel})
    fig, ax = plt.subplots(figsize=(13.4 * CM, 6.8 * CM))
    for m, c, mk, ls, lab, xy in (("muninn", AQUA, "^", "-", "Muninn [1]", (0.052, 125)),
                                  ("euler", DGRAY, "o", "--", "FewSteps, Euler", (0.024, 300)),
                                  ("ab2", BLUE, "s", "-", "FewSteps, AB2", (0.0056, 17))):
        pts = [(r["eta"], r["nfe"]) for r in sorted(sel, key=lambda r: r["eta"]) if r["method"] == m and r["param"] is not None]
        x, y = zip(*pts)
        ax.plot(x, y, color=c, marker=mk, ls=ls)
        ax.text(*xy, lab, color=c, fontsize=13.5, ha="left", va="bottom")
        miss = [e for e in etas if e not in x]
        if miss:
            ax.scatter(miss, [420] * len(miss), marker="x", s=55, color=c, lw=2, zorder=3)
    ax.text(0.0047, 560, r"$\times$: not certifiable", color=INK2, fontsize=12.5, va="bottom")
    log_eta_axis(ax, etas)
    ax.set_ylim(7, 900)
    ax.set_yticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"]); ax.minorticks_off()
    ax.set_xlabel(r"tolerance $\eta$")
    ax.set_ylabel("NFE per plan")
    style(ax)
    fig.savefig(OUT / "fig_e18.pdf")


def e19_costs():
    R = [json.load(open(UMAZE / f"e19/seed{s}/results.json"))["results"] for s in (0, 1, 2)]
    etas = sorted({r["eta"] for r in R[0]})
    get = lambda m, key: np.array([[next(r[key] for r in Rs if r["eta"] == e and r["method"] == m) for e in etas]
                                   for Rs in R])
    return etas, get


def fig_e19():
    etas, get = e19_costs()
    fig, ax = plt.subplots(figsize=(13.4 * CM, 6.8 * CM))
    for m, c, mk, ls, lab in (("ab2", BLUE, "s", "-", "FewSteps, AB2"),
                              ("free_plain", GRAY, "x", "--", r"ours without $G$ (cost $h\Vert e_k\Vert$)"),
                              ("free", ORANGE, "o", "-", "ours (Alg. 2)")):
        y = get(m, "cost")
        ax.plot(etas, y.mean(0), color=c, marker=mk, ls=ls, label=lab)
        ax.fill_between(etas, y.min(0), y.max(0), color=c, alpha=0.16, lw=0)
    sp = get("ab2", "cost").mean(0) / get("free", "cost").mean(0)
    for e, y, r in zip(etas[:3], get("free", "cost").mean(0)[:3], sp[:3]):
        ax.annotate(rf"${r:.2f}\times$", (e, y), textcoords="offset points", xytext=(7, -21), ha="left",
                    color=INK, fontsize=13)
    log_eta_axis(ax, etas)
    ax.set_ylim(8, 130)
    ax.set_yticks([10, 20, 50, 100], ["10", "20", "50", "100"]); ax.minorticks_off()
    ax.set_xlabel(r"tolerance $\eta$")
    ax.set_ylabel("NFE per plan")
    ax.legend(loc="upper right")
    style(ax)
    fig.savefig(OUT / "fig_e19.pdf")


def table_e19():
    """Numbers for Table 2 of the poster (printed, copied into poster.tex)."""
    etas, get = e19_costs()
    ab2, ours = get("ab2", "cost").mean(0), get("free", "cost").mean(0)
    pv = get("free", "p_viol").max(0)
    for e, a, o, p in zip(etas, ab2, ours, pv):
        print(f"eta {e:<6} AB2 {a:5.1f}  ours {o:5.1f}  speed-up {a / o:.2f}  max p_viol(ours) {p:.3f}")


def main():
    from muninnx.utils.misc import get_device
    OUT.mkdir(parents=True, exist_ok=True)
    dev = get_device("auto")
    fig_pareto()
    fig_gnorm()
    fig_tweedie()
    fig_e18()
    fig_e19()
    table_e19()
    fig_maze(dev)
    fig_amplification(dev)
    print(f"wrote {sorted(p.name for p in OUT.glob('*.pdf'))}")


if __name__ == "__main__":
    main()

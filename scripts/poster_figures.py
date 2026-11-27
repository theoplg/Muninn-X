"""Vector figures for the A1 poster (poster/figures/*.pdf), with poster-sized fonts.

    python scripts/poster_figures.py

  fig_amplification.pdf : 2D toy (checkerboard), amplification ||G|| of an error made at t = 0.2 (E7)
  fig_maze.pdf          : U-maze plans of the conditional planner for random start / goal pairs
  fig_e18.pdf           : certified cost vs tolerance, Muninn vs FewSteps Euler / AB2 on the U-maze (E18)
  fig_e19.pdf           : certified cost of the learned G-aware policy vs certified AB2 (E19, 3 seeds)
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
# reference categorical palette (validated; first three slots are safe all-pairs) + neutral gray
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e5e3"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#e8f0fb", "#9cc0ee", BLUE, "#123f78"])

plt.rcParams.update({
    "font.family": "Lato", "font.size": 22, "axes.titlesize": 24, "axes.labelsize": 22,
    "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 19,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "axes.linewidth": 1.2, "pdf.fonttype": 42,
    "savefig.transparent": True,
})


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_e18():
    rows = json.load(open(UMAZE / "certified_compare/results.json"))["results"]
    sel = [r for r in rows if r["ref"] == "exact" and r["alpha"] == 0.05 and r["test"] == "ltt"]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    etas = sorted({r["eta"] for r in sel})
    for m, c, mk, lab in (("muninn", AQUA, "^", "Muninn (paper), certified"),
                          ("euler", ORANGE, "o", "FewSteps Euler, certified"),
                          ("ab2", BLUE, "s", "FewSteps AB2, certified")):
        pts = [(r["eta"], r["nfe"]) for r in sorted(sel, key=lambda r: r["eta"]) if r["method"] == m and r["param"] is not None]
        x, y = zip(*pts)
        ax.plot(x, y, color=c, marker=mk, ms=11, lw=2.5, label=lab)
        missing = [e for e in etas if e not in x]
        if missing:  # not certifiable at any budget: mark at the top
            ax.scatter(missing, [330] * len(missing), marker="x", s=140, color=c, lw=3, zorder=3)
    ax.text(etas[0], 400, "×  = not certifiable", color=INK2, fontsize=18, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(7, 520)
    ax.set_xticks(etas, [f"{e:g}" for e in etas])
    ax.set_yticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("tolerance η   (guarantee: P(d > η) ≤ 5%)")
    ax.set_ylabel("network calls per plan")
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e18.pdf")


def fig_e19():
    R = [json.load(open(UMAZE / f"e19/seed{s}/results.json"))["results"] for s in (0, 1, 2)]
    etas = sorted({r["eta"] for r in R[0]})

    def cost(m):
        return np.array([[next(r["cost"] for r in Rs if r["eta"] == e and r["method"] == m) for e in etas] for Rs in R])

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for m, c, mk, ls, lab in (("ab2", BLUE, "s", "-", "FewSteps AB2, certified"),
                              ("free_plain", GRAY, "x", "--", "adaptive, not G-aware"),
                              ("free", ORANGE, "o", "-", "adaptive, G-aware (ours)")):
        y = cost(m)
        ax.plot(etas, y.mean(0), color=c, marker=mk, ms=11, lw=2.5, ls=ls, label=lab)
        ax.fill_between(etas, y.min(0), y.max(0), color=c, alpha=0.15, lw=0)
    sp = cost("ab2").mean(0) / cost("free").mean(0)
    for e, y, r in zip(etas[:3], cost("free").mean(0)[:3], sp[:3]):
        ax.annotate(f"{r:.2f}×", (e, y), textcoords="offset points", xytext=(10, -30), ha="left",
                    color=INK, fontsize=19, fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_yticks([10, 20, 50, 100], ["10", "20", "50", "100"])
    ax.set_xticks(etas, [f"{e:g}" for e in etas]); ax.minorticks_off()
    ax.set_xlabel("tolerance η   (guarantee: P(d > η) ≤ 5%)"); ax.set_ylabel("network calls per plan")
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_e19.pdf")


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
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    grid = MAPS["umaze"]
    Hc, Wc = len(grid), len(grid[0])
    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch == "#":
                ax.add_patch(plt.Rectangle((j - Wc / 2, Hc / 2 - i - 1), 1, 1, color="#c9c8c3", lw=0))
    for tr in plan:
        ax.plot(tr[:, 0], tr[:, 1], color=BLUE, lw=2.2, alpha=0.9)
        ax.scatter(tr[0, 0], tr[0, 1], s=110, color=AQUA, edgecolor="white", lw=1.5, zorder=3)
        ax.scatter(tr[-1, 0], tr[-1, 1], s=260, marker="*", color=ORANGE, edgecolor="white", lw=1.2, zorder=3)
    ax.scatter([], [], s=110, color=AQUA, label="start")
    ax.scatter([], [], s=260, marker="*", color=ORANGE, label="goal")
    ax.set_aspect("equal"); ax.set_xlim(-Wc / 2, Wc / 2); ax.set_ylim(-Hc / 2, Hc / 2)
    ax.axis("off")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_maze.pdf")


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
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0))
    norm = LogNorm(vmin=0.5, vmax=30)
    o = np.argsort(amp)  # draw the amplified points last
    for a, p, title in ((ax[0], pos, "state at t = 0.2"), (ax[1], fin, "final sample (t = 1)")):
        sc = a.scatter(p[o, 0], p[o, 1], c=amp[o], cmap=SEQ, norm=norm, s=3, lw=0, rasterized=True)
        a.set_aspect("equal"); a.set_title(title, loc="left")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_ticks([0.5, 1, 3, 10, 30], labels=["0.5", "1", "3", "10", "30"])
    cb.set_label("amplification of an error made at t = 0.2")
    cb.outline.set_visible(False)
    fig.savefig(OUT / "fig_amplification.pdf", dpi=200, bbox_inches="tight")
    print(f"amplification at t=0.2: median {np.median(amp):.2f}, p99 {np.quantile(amp, 0.99):.1f}")


def main():
    from muninnx.utils.misc import get_device
    OUT.mkdir(parents=True, exist_ok=True)
    dev = get_device("auto")
    fig_e18()
    fig_e19()
    fig_maze(dev)
    fig_amplification(dev)
    print(f"wrote {sorted(p.name for p in OUT.glob('*.pdf'))}")


if __name__ == "__main__":
    main()

"""Poster figure: the same error e_k, injected at step k (t = 0.2), on two checkerboard samples.

    python scripts/poster_fig_ridge.py   -> poster/figures/fig_ridge.pdf (full column), fig_ridge_half.pdf (Fig. 3b)

(a) a sample "in a valley": ||G_{k+1->N}|| ~ 1, its destination square is already decided;
(b) a sample "on a ridge": large ||G_{k+1->N}||, it is still between two squares.
Each sample is integrated with Euler-100, once without error and once per direction with a state error
h e_k of fixed size DELTA added right after step K (Delta_{k+1} = h e_k, notation of Proposition 1).
The directions are the most amplified one (top right-singular vector of G_{k+1->N}) rotated by multiples of
360 / N_DIR degrees, so the largest final deviation approaches ||G_{k+1->N}|| * DELTA.
DELTA is larger than a typical reuse error so that the paths are visible; for small errors the amplification
factor does not depend on the size.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from muninnx.analysis.flow import flow_jacobians
from muninnx.data.toy2d import checkerboard
from muninnx.tasks import load_task
from muninnx.utils.misc import get_device

NAVY, ORANGE, INK, INK2 = "#1f3a5f", "#eb6834", "#111111", "#4a4a46"
SQUARE, SQUARE_EDGE = "#ecebe6", "#d6d5cf"
CM = 1 / 2.54
import os  # noqa: E402
N, H, K, N_DIR = 100, 0.01, 20, 6
DELTA = float(os.environ.get("DELTA", 0.1))
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 14, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
                     "savefig.transparent": True})


@torch.no_grad()
def euler_from(model, x, k0):
    path = [x]
    for j in range(k0, N):
        x = x + H * model(x, j * H)
        path.append(x)
    return torch.stack(path, 1)  # (B, steps, 2)


def cell(p):
    return (torch.floor(p[..., 0] / 2) * 10 + torch.floor(p[..., 1] / 2)).long()


def depth(e):
    """Distance to the nearest edge of the 2x2 cell, in cell units (0 = on an edge, 0.5 = centre)."""
    return 0.5 - (e / 2 - torch.floor(e / 2) - 0.5).abs().max().item()


def main():
    dev = get_device("auto")
    model = load_task("runs/toy_checkerboard/model.pt", dev).model
    x0 = torch.randn(6000, 2, device=dev, generator=torch.Generator(device=dev).manual_seed(4))
    full = euler_from(model, x0, 0)                                   # states x_0 .. x_N
    G = flow_jacobians(model, [full[:, j] for j in range(N)], [j * H for j in range(N)], H)[K]
    _, S, Vh = torch.linalg.svd(G)
    amp, v1 = S[:, 0], Vh[:, 0]                                       # ||G_{k+1->N}|| and its input direction

    def directions(i):
        a0 = torch.atan2(v1[i, 1], v1[i, 0])
        ang = a0 + torch.arange(N_DIR, device=dev) * 2 * torch.pi / N_DIR
        return torch.stack([ang.cos(), ang.sin()], 1)

    def with_error(i):                                                # (N_DIR, steps, 2)
        return euler_from(model, full[i, K + 1] + DELTA * directions(i), K + 1)

    data = checkerboard(20000, generator=torch.Generator().manual_seed(0))
    filled = sorted(set(cell(data).tolist()))
    squares = sorted(set(map(tuple, torch.floor(data / 2).long().tolist())))   # (column, row) of each square
    end, start = full[:, -1].cpu(), full[:, 0].cpu()
    length = (end - start).norm(dim=1)
    in_filled = torch.tensor([c in filled for c in cell(end).tolist()])

    # (a) valley: ||G|| ~ 1, ends well inside a square, long path
    cand = [i for i in torch.argsort((amp - 1).abs()).tolist()[:400]
            if in_filled[i] and depth(end[i]) > 0.18]
    valley = max(cand[:60], key=lambda i: length[i].item())
    # (b) ridge: ||G|| in the top percent but not extreme; +-v1 errors end in two different squares
    best = None
    for i in torch.where((amp > 8) & (amp < 60))[0].tolist():
        if not in_filled[i]:
            continue
        ends = with_error(i)[[0, N_DIR // 2], -1].cpu()                # errors along +v1 and -v1
        c = cell(ends).tolist()
        if c[0] != c[1] and all(x in filled for x in c):
            score = min(depth(ends[0]), depth(ends[1])) + 0.05 * length[i].item()
            if best is None or score > best[0]:
                best = (score, i)
    ridge = best[1]

    paths = {i: (full[i].cpu(), with_error(i).cpu()) for i in (valley, ridge)}
    ratio = {i: ((q[:, -1] - p[-1]).norm(dim=1).max() / DELTA).item() for i, (p, q) in paths.items()}
    for name, i in (("valley", valley), ("ridge", ridge)):
        print(f"{name}: ||G_k+1->N|| = {amp[i]:.2f}, max ||Delta_N|| / ||h e_k|| = {ratio[i]:.2f}")

    # same zoom scale in both panels (different centres)
    half = 0.0
    centres = {}
    for i, (p, q) in paths.items():
        pts = torch.cat([p[K + 1:], q.reshape(-1, 2)])
        lo, hi = pts.min(0).values, pts.max(0).values
        centres[i], half = (lo + hi) / 2, max(half, (hi - lo).max().item() / 2 + 0.45)

    for layout in ("full", "half"):
        draw(layout, paths, squares, amp, ratio, valley, ridge, centres, half)


# sizes per layout: full = full poster column (two panels side by side), half = half column (Figure 3b)
LAYOUTS = {
    "full": dict(size=(26.5, 12.2), title=15.5, label=14, box=14, legend=13, ncol=3, lw=(3.0, 1.6), ms=(95, 75, 42),
                 out="poster/figures/fig_ridge.pdf"),
    "half": dict(size=(13.4, 6.9), title=12.5, label=12, box=11.5, legend=11.5, ncol=2, lw=(2.4, 1.2), ms=(55, 45, 22),
                 out="poster/figures/fig_ridge_half.pdf"),
}


def draw(layout, paths, squares, amp, ratio, valley, ridge, centres, half):
    L = LAYOUTS[layout]
    full_ = layout == "full"
    fig, axes = plt.subplots(1, 2, figsize=(L["size"][0] * CM, L["size"][1] * CM), gridspec_kw={"wspace": 0.06})
    for ax, i, tag, name in ((axes[0], valley, "(a)", "in a valley"), (axes[1], ridge, "(b)", "on a ridge")):
        p, q = paths[i]
        for col, row in squares:
            ax.add_patch(Rectangle((2 * col, 2 * row), 2, 2, facecolor=SQUARE, edgecolor=SQUARE_EDGE, lw=1))
        for path in q:
            ax.plot(path[:, 0], path[:, 1], color=ORANGE, lw=L["lw"][1], alpha=0.95)
        ax.plot(p[K + 1:, 0], p[K + 1:, 1], color=NAVY, lw=L["lw"][0], solid_capstyle="round")
        ax.scatter(q[:, -1, 0], q[:, -1, 1], s=L["ms"][2], color=ORANGE, edgecolor="white", lw=0.6, zorder=4)
        ax.scatter(*p[K + 1], s=L["ms"][1], color=NAVY, marker="s", edgecolor="white", lw=0.6, zorder=5)
        ax.scatter(*p[-1], s=L["ms"][0], color=NAVY, edgecolor="white", lw=0.6, zorder=5)
        xk_off, xn_off = ((10, -2), (10, -2)) if tag == "(a)" else ((0, 10), (9, -5))
        xk_text = r"$x_k$  ($t_k=0.2$)" if full_ else r"$x_k$"
        ax.annotate(xk_text, p[K + 1], textcoords="offset points", xytext=xk_off,
                    ha="left" if tag == "(a)" else "center", va="center" if tag == "(a)" else "bottom",
                    fontsize=L["label"], color=NAVY)
        ax.annotate(r"$x_N$", p[-1], textcoords="offset points", xytext=xn_off, ha="left", va="center",
                    fontsize=L["label"], color=NAVY)
        title = (rf"{tag} {name}:  " if full_ else f"{name.split()[-1]}:  ") + rf"$\Vert G_{{k+1\to N}}\Vert = {amp[i]:.1f}$"
        ax.set_title(title, loc="left", fontsize=L["title"], linespacing=1.15)
        box = (rf"$\max\,\Vert\Delta_N\Vert\,/\,\Vert h\,e_k\Vert = {ratio[i]:.1f}$" if full_
               else rf"$\Vert\Delta_N\Vert \leq {ratio[i]:.1f}\,\Vert h\,e_k\Vert$")
        ax.text(0.03, 0.97 if tag == "(a)" else 0.03, box, transform=ax.transAxes, fontsize=L["box"], color=INK,
                va="top" if tag == "(a)" else "bottom",
                bbox=dict(facecolor="white", edgecolor=SQUARE_EDGE, boxstyle="round,pad=0.25"))
        c = centres[i]
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(SQUARE_EDGE)
    err = (rf"error $h\,e_k$ added at step $k$ ($\Vert h\,e_k\Vert = {DELTA}$, {N_DIR} directions)" if full_
           else rf"with an error $h\,e_k$ at step $k$")
    handles = [plt.Line2D([], [], color=NAVY, lw=L["lw"][0], marker="o", ms=7, label="without error"),
               plt.Line2D([], [], color=ORANGE, lw=L["lw"][1], marker="o", ms=5, label=err)]
    if full_:
        handles.append(Rectangle((0, 0), 1, 1, facecolor=SQUARE, edgecolor=SQUARE_EDGE,
                                 label="data (checkerboard squares)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03 if full_ else -0.04),
               ncol=L["ncol"], fontsize=L["legend"], frameon=False, handlelength=2.2, columnspacing=1.6)
    fig.savefig(L["out"], dpi=220)
    print(f"wrote {L['out']}")


if __name__ == "__main__":
    main()

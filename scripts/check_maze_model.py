"""Quality check of Maze2D flow-matching models before running Muninn experiments on them.

    python scripts/check_maze_model.py runs/maze2d_umaze/maze_unet/model.pt runs/maze2d_umaze/maze_unet_v2/model.pt

Per model (unconditional samples, same noise for all models):
  - wall violations: fraction of plans with a point inside a wall (0.1 margin), vs the data
    (U-maze only for now: interior wall |y| < 0.5, x < 0.5; outer walls at |x|, |y| = 1.5)
  - E[d(Euler-100, exact)] with exact = RK4-200 (paper metric)
  - time smoothness of the velocity field (streamed, constant memory): ~1 = smooth (finding 13)
  - AB2 p95 error vs number of steps, and count of erratic (non-monotone) jumps
Writes <model dir>/check.json and a side-by-side plot next to the first model.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from muninnx.baselines.fewsteps import sample_ab2, sample_rk4
from muninnx.data.maze2d import Maze2DWindows
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver
from muninnx.tasks import load_task
from muninnx.utils.misc import get_device, save_json, seed_all


def wall_rate(x):
    p = x[..., :2]
    inside = (p[..., 1].abs() < 0.4) & (p[..., 0] < 0.4)
    outer = (p.abs() > 1.45).any(-1)
    return (inside | outer).any(1).float().mean().item()


@torch.no_grad()
def smoothness(model, x, T=9001, lag=100):
    ts = torch.linspace(0.05, 0.95, T)
    win, fs, cs, nf, nc = [], 0.0, 0.0, 0, 0
    for t in ts:
        v = model(x, float(t))
        if win:
            fs += ((v - win[-1]).norm(dim=1) / 1e-4).sum().item(); nf += x.shape[0]
        if len(win) == lag:
            cs += ((v - win[0]).norm(dim=1) / 1e-2).sum().item(); nc += x.shape[0]; win.pop(0)
        win.append(v)
    return (fs / nf) / (cs / nc)


def main(paths):
    seed_all(0)
    dev = get_device()
    ds = Maze2DWindows("umaze", device=dev)
    data = ds.unnormalize(ds.sample(4000))
    print(f"data            | wall violations {wall_rate(data):.4f}")
    Ns = [10, 15, 20, 30, 40, 60, 80, 100, 150]
    gens = {}
    for path in paths:
        task = load_task(path, dev)
        m = task.model
        g = torch.Generator(device=dev).manual_seed(1)
        x0 = task.noise(4000, dev, generator=g)
        with torch.no_grad():
            ref = sample_rk4(m, 200, x0)
            eul = sample_full(m, EulerSolver(100), x0)
            p95 = [task.dist(sample_ab2(m, n, x0[:2000]), ref[:2000]).quantile(0.95).item() for n in Ns]
        gen = ds.unnormalize(m.unflat(eul))
        gens[path] = gen
        res = {"wall_violations": wall_rate(gen), "wall_violations_exact": wall_rate(ds.unnormalize(m.unflat(ref))),
               "euler100_vs_exact": task.dist(eul, ref).mean().item(), "t_smoothness": smoothness(m, x0[:500]),
               "ab2_p95": dict(zip(Ns, p95)), "ab2_erratic_jumps": sum(p95[i + 1] > 1.15 * p95[i] for i in range(len(Ns) - 1))}
        save_json(res, Path(path).parent / "check.json")
        print(f"{Path(path).parent.name:15s} | wall violations {res['wall_violations']:.4f} | E[d(Euler-100, exact)] "
              f"{res['euler100_vs_exact']:.4f} | t-smoothness {res['t_smoothness']:.2f} | AB2 p95 " +
              " ".join(f"{n}:{v:.3f}" for n, v in zip(Ns, p95)) + f" | erratic jumps {res['ab2_erratic_jumps']}")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    panels = [("data", data)] + [(Path(p).parent.name, g) for p, g in gens.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for a, (name, w) in zip(axes, panels):
        a.add_patch(plt.Rectangle((-1.5, -0.5), 2.0, 1.0, color="#888888", alpha=0.5))
        a.add_patch(plt.Rectangle((-1.5, -1.5), 3, 3, fill=False, ec="#888888", lw=2))
        for tr in w[:60].cpu():
            a.plot(tr[:, 0], tr[:, 1], lw=0.8, alpha=0.7, color="#2a78d6")
        a.set_aspect("equal"); a.set_xlim(-1.7, 1.7); a.set_ylim(-1.7, 1.7); a.set_title(name, fontsize=10)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Maze2D U-maze: unconditional 128-step plans (Euler-100); grey = walls", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(paths[0]).parent.parent / "samples_compare.png", dpi=110)


if __name__ == "__main__":
    main(sys.argv[1:])

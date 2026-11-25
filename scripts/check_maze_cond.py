"""Quality check of start/goal-conditioned Maze2D planners (U-maze).

    python scripts/check_maze_cond.py runs/maze2d_umaze/maze_unet_cond/model_it10000.pt ... [--inpaint-baseline v2.pt]

Same contexts (start state + goal position from real data windows) and same noise for every model.
Per model, on Euler-100 plans (unnormalised maze units):
  start / goal error : distance between the plan's first / last position and the requested start / goal
  jumps              : fraction of plans with a step longer than 0.15 (data max ~0.06): "teleports"
  walls              : fraction of plans entering a wall (same rule as check_maze_model.py)
  E[d(Euler-100, exact)], AB2 p95 at 20 / 40 / 80 steps (exact = RK4-200)
The baseline (--inpaint-baseline) is the unconditional model with inpainting (Diffuser-style).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from muninnx.baselines.fewsteps import sample_ab2, sample_rk4
from muninnx.conditioning import Inpaint, InpaintSolver, maze_condition_indices
from muninnx.data.maze2d import Maze2DWindows, window_context
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver
from muninnx.tasks import load_task
from muninnx.utils.misc import get_device, save_json, seed_all


def wall_rate(p):
    inside = (p[..., 1].abs() < 0.4) & (p[..., 0] < 0.4)
    return (inside | (p.abs() > 1.45).any(-1)).any(1).float().mean().item()


@torch.no_grad()
def evaluate(task, ds, ctx, n, inpaint=False):
    m = task.model
    g = torch.Generator(device=ctx.device).manual_seed(7)
    Dx = m.H * m.dim
    z = torch.randn(n, Dx, device=ctx.device, generator=g)
    if inpaint:  # unconditional model + inpainting of start / goal
        idx = maze_condition_indices(m.H)
        target = torch.cat([ctx[:, :4], ctx[:, 4:]], dim=1)
        proj = Inpaint(idx, target, z)
        run = lambda solver: sample_full(m, InpaintSolver(solver, proj), z)
        ref = sample_rk4(m, 200, z, project=proj)
        ab2 = lambda k: sample_ab2(m, k, z, project=proj)
    else:
        x0 = torch.cat([z, ctx], dim=1)
        run = lambda solver: sample_full(m, solver, x0)
        ref = sample_rk4(m, 200, x0)
        ab2 = lambda k: sample_ab2(m, k, x0)
    eul = run(EulerSolver(100))
    plan = ds.unnormalize(eul[:, :Dx].reshape(n, m.H, m.dim))
    want = ds.unnormalize(torch.cat([ctx[:, :4], torch.zeros(n, 2, device=ctx.device)], 1)[:, None].expand(n, 1, 6))
    goal = ds.unnormalize(torch.cat([ctx[:, 4:6], torch.zeros(n, 4, device=ctx.device)], 1)[:, None])[:, 0, :2]
    step = (plan[:, 1:, :2] - plan[:, :-1, :2]).norm(dim=-1)
    return plan, {
        "start_err": (plan[:, 0, :2] - want[:, 0, :2]).norm(dim=-1).mean().item(),
        "goal_err": (plan[:, -1, :2] - goal).norm(dim=-1).mean().item(),
        "jump_frac": (step > 0.15).any(1).float().mean().item(),
        "walls": wall_rate(plan[..., :2]),
        "euler100_vs_exact": task.dist(eul, ref).mean().item(),
        "ab2_p95": {k: task.dist(ab2(k), ref).quantile(0.95).item() for k in (20, 40, 80)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--inpaint-baseline", default=None)
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()
    seed_all(0)
    dev = get_device()
    ds = Maze2DWindows("umaze", device=dev)
    ctx = window_context(ds.sample(args.n, generator=torch.Generator(device=dev).manual_seed(3)))
    rows, plans = {}, {}
    todo = ([("v2 + inpainting", args.inpaint_baseline, True)] if args.inpaint_baseline else []) + \
           [(Path(c).stem, c, False) for c in args.ckpts]
    for name, ckpt, inp in todo:
        plan, r = evaluate(load_task(ckpt, dev), ds, ctx, args.n, inpaint=inp)
        rows[name], plans[name] = r, plan
        print(f"{name:16s} | start err {r['start_err']:.3f} | goal err {r['goal_err']:.3f} | plans with a jump "
              f"{r['jump_frac']:.3f} | walls {r['walls']:.4f} | E[d(Euler-100, exact)] {r['euler100_vs_exact']:.4f} | "
              f"AB2 p95 20/40/80: " + "/".join(f"{v:.3f}" for v in r["ab2_p95"].values()))
    out = Path(args.ckpts[-1]).parent
    save_json(rows, out / "check_cond.json")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    show = [k for k in rows if k.startswith("v2")] + [list(rows)[-1]]
    fig, axes = plt.subplots(1, len(show), figsize=(5.2 * len(show), 5.2))
    axes = axes if len(show) > 1 else [axes]
    for a, name in zip(axes, show):
        pl = plans[name][:20].cpu()
        a.add_patch(plt.Rectangle((-1.5, -0.5), 2.0, 1.0, color="#888888", alpha=0.5))
        a.add_patch(plt.Rectangle((-1.5, -1.5), 3, 3, fill=False, ec="#888888", lw=2))
        for tr in pl:
            a.plot(tr[:, 0], tr[:, 1], lw=1, alpha=0.8, color="#2a78d6")
        s = ds.unnormalize(torch.cat([ctx[:20, :4], torch.zeros(20, 2, device=dev)], 1)).cpu()
        gl = ds.unnormalize(torch.cat([ctx[:20, 4:6], torch.zeros(20, 4, device=dev)], 1)).cpu()
        a.scatter(s[:, 0], s[:, 1], s=16, color="#1baf7a", zorder=3, label="requested start")
        a.scatter(gl[:, 0], gl[:, 1], s=30, marker="*", color="#eb6834", zorder=3, label="requested goal")
        a.set_aspect("equal"); a.set_xlim(-1.7, 1.7); a.set_ylim(-1.7, 1.7); a.set_title(name, fontsize=10)
        a.legend(frameon=False, fontsize=7, loc="center")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("U-maze: plans for a requested start and goal (Euler-100)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "conditioned_compare.png", dpi=110)


if __name__ == "__main__":
    main()

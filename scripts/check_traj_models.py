"""M0.5 phase 2 check: are the trajectory backbones good enough to run the Muninn experiments on?

    python scripts/check_traj_models.py

For each trained backbone in runs/traj_toy/<arch>/model.pt:
  - sample quality (Euler-100 and near-exact RK4-200, 20k samples): collision rate, fraction of paths
    above the obstacle (data: 0.5), mean distance of samples to the nearest of 20k data trajectories
    (vs data-to-data), roughness (mean squared second difference along the path, vs data);
  - smoothness of the velocity field in t (finding 13): mean |dv/dt| at fine scale vs over one step of
    a 100-step grid (ratio close to 1 = smooth);
  - compute split (FLOPs) of encode / core / decode, i.e. probe cost and feature-caching cost.
Writes runs/traj_toy/check.json and samples.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.flop_counter import FlopCounterMode

from muninnx.baselines.fewsteps import sample_rk4
from muninnx.data.traj2d import OBSTACLE_RADIUS, collision_rate, path_side, sample_traj
from muninnx.models.traj_backbones import load_traj_model
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver
from muninnx.utils.misc import cap_gpu_memory, save_json, seed_all

ARCHS = ["mlp", "unet", "transformer"]


def roughness(x):
    return (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).pow(2).sum(-1).mean().item()


@torch.no_grad()
def nn_dist(x, data, chunk=1000):
    """Mean over x of the RMS-over-time distance to the nearest data trajectory."""
    out, D = [], data.flatten(1)
    for xc in x.flatten(1).split(chunk):
        out.append(torch.cdist(xc, D).min(dim=1).values / x.shape[1] ** 0.5)
    return torch.cat(out).mean().item()


@torch.no_grad()
def time_smoothness(model, n=2000, H=32, dev="cuda", T=9001, lag=100):
    """Streaming version (constant memory): mean |dv/dt| at the fine scale (dt = 1e-4) divided by the mean
    over one step of a 100-step grid (dt = 1e-2). Keeps only the last `lag` outputs instead of all T
    (the full stack is n * T * H * 2 floats = 4.6 GB for trajectories and must never be materialised)."""
    x = torch.randn(n, H, 2, device=dev)
    ts = torch.linspace(0.05, 0.95, T)
    window, fine_sum, coarse_sum, n_fine, n_coarse = [], 0.0, 0.0, 0, 0
    for t in ts:
        v = model(x, float(t))
        if window:
            fine_sum += ((v - window[-1]).flatten(1).norm(dim=1) / 1e-4).sum().item()
            n_fine += n
        if len(window) == lag:
            coarse_sum += ((v - window[0]).flatten(1).norm(dim=1) / 1e-2).sum().item()
            n_coarse += n
            window.pop(0)
        window.append(v)
    return (fine_sum / n_fine) / (coarse_sum / n_coarse)


def flop_split(model, dev="cuda"):
    x, t = torch.randn(256, 32, 2, device=dev), torch.rand(256, device=dev)
    with torch.no_grad():
        h, aux = model.encode(x, t)
        hc = model.core(h, aux)
        res = {}
        for part, fn in [("encode", lambda: model.encode(x, t)), ("core", lambda: model.core(h, aux)),
                         ("decode", lambda: model.decode(hc, h, aux)), ("total", lambda: model(x, t))]:
            with FlopCounterMode(display=False) as fc:
                fn()
            res[part] = fc.get_total_flops()
    return {k: v / res["total"] for k, v in res.items() if k != "total"}


def main():
    seed_all(0)
    dev = torch.device("cuda")
    cap_gpu_memory()
    data = sample_traj(20000, device=dev)
    data_ref = sample_traj(20000, device=dev)
    out = {"data": {"collision": collision_rate(data), "frac_above": (path_side(data) > 0).float().mean().item(),
                    "nn_dist": nn_dist(data_ref[:5000], data), "roughness": roughness(data)}}
    print(f"data        | collisions {out['data']['collision']:.4f} | above {out['data']['frac_above']:.3f} | "
          f"NN dist (data->data) {out['data']['nn_dist']:.4f} | roughness {out['data']['roughness']:.5f}")
    x0 = torch.randn(20000, 32, 2, device=dev)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    samples = {"data": data[:250].cpu()}
    for arch in ARCHS:
        path = Path(f"runs/traj_toy/{arch}/model.pt")
        if not path.exists():
            print(f"{arch}: not trained yet")
            continue
        model, cfg = load_traj_model(path, dev)
        row = {"n_params": cfg.get("n_params")}
        for solver_name, fn in [("euler100", lambda: sample_full(model, EulerSolver(100), x0)),
                                ("rk4_200", lambda: sample_rk4(model, 200, x0))]:
            x = fn()
            row[solver_name] = {"collision": collision_rate(x), "frac_above": (path_side(x) > 0).float().mean().item(),
                                "nn_dist": nn_dist(x[:5000], data), "roughness": roughness(x)}
            if solver_name == "euler100":
                samples[arch] = x[:250].cpu()
        row["euler100_vs_exact"] = (sample_full(model, EulerSolver(100), x0[:5000])
                                    - sample_rk4(model, 200, x0[:5000])).flatten(1).norm(dim=1).div(32 ** 0.5).mean().item()
        row["t_smoothness_ratio"] = time_smoothness(model)
        row["flops"] = flop_split(model)
        out[arch] = row
        e = row["euler100"]
        print(f"{arch:11s} | collisions {e['collision']:.4f} | above {e['frac_above']:.3f} | NN dist {e['nn_dist']:.4f} | "
              f"roughness {e['roughness']:.5f} | E[d(Euler-100, exact)] {row['euler100_vs_exact']:.4f} | "
              f"t-smoothness {row['t_smoothness_ratio']:.2f} | probe {row['flops']['encode']:.3f}, "
              f"encode+decode {row['flops']['encode'] + row['flops']['decode']:.3f} of a forward")
    for ax, (name, x) in zip(axes, samples.items()):
        side = path_side(x)
        for tr, s in zip(x, side):
            ax.plot(tr[:, 0], tr[:, 1], color="#2a78d6" if s > 0 else "#eb6834", lw=0.6, alpha=0.5)
        ax.add_patch(plt.Circle((0, 0), OBSTACLE_RADIUS, color="#888888", alpha=0.6))
        ax.set_aspect("equal"); ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.2, 3.2)
        ax.set_title(name if name == "data" else f"{name} (FM, Euler-100)", fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig("runs/traj_toy/samples.png", dpi=120)
    save_json(out, "runs/traj_toy/check.json")
    print("wrote runs/traj_toy/check.json and runs/traj_toy/samples.png")


if __name__ == "__main__":
    main()

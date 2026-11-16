"""Plotting helpers (matplotlib, saved to files)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def scatter_compare(panels, path, lim=6.0, s=1.0):
    """panels: dict title -> (N, 2) tensor/array. One subplot per entry."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, (title, pts) in zip(axes[0], panels.items()):
        pts = pts.detach().cpu().numpy() if hasattr(pts, "detach") else pts
        ax.scatter(pts[:, 0], pts[:, 1], s=s, alpha=0.5)
        ax.set_title(title)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_paths(traj, path, n_show=200, lim=6.0):
    """traj: (n_steps+1, N, 2) states along the ODE; draws the first n_show paths."""
    traj = traj.detach().cpu().numpy() if hasattr(traj, "detach") else traj
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(min(n_show, traj.shape[1])):
        ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.4, alpha=0.5)
    ax.scatter(traj[0, :n_show, 0], traj[0, :n_show, 1], s=3, c="k", label="x0 (noise)")
    ax.scatter(traj[-1, :n_show, 0], traj[-1, :n_show, 1], s=3, c="r", label="x1 (sample)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_lines(xs, ys_dict, path, xlabel="", ylabel="", title=""):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, ys in ys_dict.items():
        ax.plot(xs, ys, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)

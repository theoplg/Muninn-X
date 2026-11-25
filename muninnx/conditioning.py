"""Inpainting conditioning for flow-matching samplers (Diffuser-style start / goal conditioning).

The model is trained unconditionally. At sampling time, some coordinates of the (flat) state are pinned:
after every solver step, coordinate set `idx` is overwritten with its value on the straight FM path,
    x_t[idx] = (1 - t) * x0[idx] + t * target,
so it equals `target` exactly at t = 1 and stays consistent with the noise x0 the sample started from.

Usage: proj = Inpaint(idx, target, x0); pass `project=proj` to the samplers in this repo, or wrap a solver
with InpaintSolver(base_solver, proj) for code that only calls solver.step (e.g. the Muninn controller).
Both paired chains of an experiment must use the same Inpaint object (same x0, same targets).
"""
import torch


class Inpaint:
    def __init__(self, idx, target, x0):
        self.idx = torch.as_tensor(idx, dtype=torch.long, device=x0.device)
        self.target = target            # (B, len(idx))
        self.x0 = x0[:, self.idx]       # (B, len(idx))

    def __call__(self, x, t):
        x = x.clone()
        x[:, self.idx] = (1 - t) * self.x0 + t * self.target
        return x

    def subset(self, mask):
        """Same conditioning restricted to a subset of the batch rows."""
        out = Inpaint.__new__(Inpaint)
        out.idx, out.target, out.x0 = self.idx, self.target[mask], self.x0[mask]
        return out


class InpaintSolver:
    """Wraps a solver so that every step is followed by the inpainting projection."""

    def __init__(self, base, project):
        self.base, self.project = base, project
        self.n_steps, self.ts = base.n_steps, base.ts

    def t(self, k):
        return self.base.t(k)

    def dt(self, k):
        return self.base.dt(k)

    def coeffs(self, k):
        return self.base.coeffs(k)

    def step(self, x, v, k):
        return self.project(self.base.step(x, v, k), float(self.base.ts[k + 1]))


def maze_condition_indices(H, dim=6, start_dims=(0, 1, 2, 3), goal_dims=(0, 1)):
    """Flat indices of the start state (first time step) and the goal position (last time step) for a
    (H, dim) trajectory flattened row-major."""
    return [d for d in start_dims] + [(H - 1) * dim + d for d in goal_dims]

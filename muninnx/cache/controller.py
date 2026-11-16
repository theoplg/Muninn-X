"""Muninn controller = paper Algorithm 1, for flow matching (staleness-aware).

>>> YOU IMPLEMENT `muninn_sample` (GUIDE.md, step 6) <<<
"""
from dataclasses import dataclass

import torch


@dataclass
class MuninnConfig:
    eta: float                 # trajectory tolerance eta_traj (in units of the metric d)
    alpha: float               # global risk level
    omega: float = 1e-3        # score denominator constant
    forbid_prefix: int = 5     # first steps (near noise) always recomputed
    forbid_suffix: int = 5     # last steps (near data) always recomputed
    max_lag: int = 10          # a cached velocity may be reused at most this many steps in a row
    gamma: float = 1.0         # metric constant: d(a, b) <= gamma * ||a - b||  (1/sqrt(H) for trajectories)


def cache_steps(n_steps, cfg):
    """T_cache: the step indices where reuse is allowed. Step 0 is never cacheable (empty cache)."""
    lo = max(1, cfg.forbid_prefix)
    hi = n_steps - cfg.forbid_suffix
    return list(range(lo, hi))


@torch.no_grad()
def muninn_sample(model, solver, x0, bound, cfg):
    """Sample with certified velocity reuse.

    Per-sample state (batch of B independent chains):
        budget (B,) = cfg.eta,  v_cache (B, ...),  f_cache (B, D) = probe features at the
        step where v_cache was computed,  lag (B,) = steps since the last recompute.

    For each step k = 0..N-1:
      1. f_k = probe(x_k, t_k)
      2. if k in T_cache:  s = score(f_k, f_cache),  c_hat = cfg.gamma * L_k * bound(s)
         reuse = (c_hat <= budget) & (lag < cfg.max_lag)            [else: reuse = False]
      3. recompute rows: v_cache = model(x, t_k), f_cache = f_k, lag = 0
         reused rows:    budget -= c_hat, lag += 1
      4. x_{k+1} = solver.step(x_k, v_cache, k)
    (L_k from muninnx.cache.bound.sensitivity_weights.)

    Batch tip: with a boolean mask `rec` of shape (B,), call model(x[rec], t) only on
    those rows and write the result into v_cache[rec].

    Args:
        bound: a fitted ConformalUpperBound (fit with alpha_step = cfg.alpha / |T_cache|,
               on pairs collected with the same max_lag)
    Returns:
        x_N: (B, ...) samples
        info: dict with
            "nfe":         (B,) long   number of full network evaluations per sample
            "reused":      (N, B) bool which steps were reused
            "bound_used":  (B,) float  sum of c_hat over reused steps  (= eta - budget)
            "budget_left": (B,) float  budget at the end (must be >= 0)
    """
    # TODO(you)
    raise NotImplementedError

"""Paired evaluation: full vs cached chain from the SAME noise x0.

>>> YOU IMPLEMENT `trajectory_distance` and `summarize` (GUIDE.md, step 4) <<<
"""
import math

import torch


def trajectory_distance(a, b):
    """d(a, b) = ||a - b||_F / sqrt(H)   per sample (paper, Assumption 2).

    Shapes: (B, D) for toys (treat H = 1) or (B, H, D) for trajectories.
    Returns (B,).
    """
    # TODO(you)
    raise NotImplementedError


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(d, eta, nfe, n_steps):
    """Aggregate a paired run.

    Args:
        d: (B,) distances to the full-compute samples
        eta: tolerance
        nfe: (B,) number of network evaluations used
        n_steps: N of the full chain
    Returns dict with at least:
        "E_d", "p_viol" (= fraction d > eta), "p_viol_ci" (wilson_ci), "nfe_mean",
        "speedup_nfe" (= n_steps / nfe_mean)
    """
    # TODO(you)
    raise NotImplementedError

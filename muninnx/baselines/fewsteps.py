"""Certified FewSteps: choose the smallest uniform Euler step count N' with a calibrated
guarantee P(d > eta) <= alpha, where d = distance to the full N-step chain from the same noise.

Selection is fixed-sequence testing (Learn-Then-Test style): test N' = N, N-1, ... in order
and stop at the first N' that fails. Order matters for validity: no correction for multiple
testing is needed because the sequence is fixed in advance and we stop at the first failure.

Two tests for "P(d > eta) <= alpha" from k violations among n calibration samples:
  - "conformal": (k + 1) / (n + 1) <= alpha
        marginal guarantee (over calibration and test draw), same type as Muninn's.
  - "ltt": binomial p-value  P(Binom(n, alpha) <= k) <= delta
        PAC guarantee: with prob >= 1 - delta over calibration, P(d > eta) <= alpha.
"""
import torch
from scipy.stats import binom

from muninnx.eval.paired import trajectory_distance
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver


@torch.no_grad()
def fewsteps_distances(model, x0, n_full, candidates):
    """d(N') for each N' in candidates, against the full n_full-step chain. Returns {N': (B,) cpu}."""
    x_full = sample_full(model, EulerSolver(n_full), x0)
    return {n: trajectory_distance(x_full, sample_full(model, EulerSolver(n), x0)).cpu()
            for n in candidates}


def certifies(k, n, alpha, method="conformal", delta=0.05):
    if method == "conformal":
        return (k + 1) / (n + 1) <= alpha
    if method == "ltt":
        return binom.cdf(k, n, alpha) <= delta
    raise ValueError(method)


def fixed_sequence_select(d_list, eta, alpha, method="conformal", delta=0.05):
    """Generic fixed-sequence testing. d_list: calibration distances, one (n,) tensor per configuration,
    ordered from the SAFEST to the most aggressive. Returns the index of the last configuration that
    certifies P(d > eta) <= alpha before the first failure, or None if even the first one fails."""
    chosen = None
    for i, d in enumerate(d_list):
        if not certifies(int((d > eta).sum()), d.numel(), alpha, method, delta):
            break
        chosen = i
    return chosen


def bonferroni_select(d_list, costs, eta, alpha, delta=0.05):
    """Order-free LTT: test EVERY configuration at level delta / |configs| (Bonferroni) and return the
    index of the cheapest one that certifies P(d > eta) <= alpha, or None. Valid without any
    monotonicity of the risk in the knob (unlike fixed-sequence testing), at the price of a
    stricter per-configuration test."""
    level = delta / len(d_list)
    ok = [i for i, d in enumerate(d_list)
          if certifies(int((d > eta).sum()), d.numel(), alpha, "ltt", level)]
    return min(ok, key=lambda i: costs[i]) if ok else None


def select_n_steps(d_cal, eta, alpha, method="conformal", delta=0.05):
    """d_cal: {N': (n,) distances}. Returns the smallest certified N' via fixed-sequence testing
    from the largest N' downward (the full chain N has d = 0 and always certifies)."""
    chosen = None
    for n_steps in sorted(d_cal, reverse=True):
        d = d_cal[n_steps]
        k = int((d > eta).sum())
        if not certifies(k, d.numel(), alpha, method, delta):
            break
        chosen = n_steps
    return chosen


# ---------------------------------------------------------------------------------------------
# Higher-order uniform-grid samplers (FewSteps baselines for E5)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_ab2(model, n_steps, x0, project=None):
    """Adams-Bashforth 2 on a uniform grid, 1 NFE per step (first step is Euler). NFE = n_steps.
    project(x, t): optional conditioning applied after every step (muninnx.conditioning.Inpaint)."""
    h = 1.0 / n_steps
    x, v_prev = x0, None
    for k in range(n_steps):
        v = model(x, k * h)
        x = x + h * (v if v_prev is None else 1.5 * v - 0.5 * v_prev)
        if project is not None:
            x = project(x, (k + 1) * h)
        v_prev = v
    return x


@torch.no_grad()
def sample_heun(model, n_steps, x0):
    """Heun (explicit trapezoid) on a uniform grid, 2 NFE per step. NFE = 2 * n_steps."""
    h = 1.0 / n_steps
    x = x0
    for k in range(n_steps):
        t = k * h
        v1 = model(x, t)
        v2 = model(x + h * v1, t + h)
        x = x + 0.5 * h * (v1 + v2)
    return x


@torch.no_grad()
def sample_rk4(model, n_steps, x0, project=None):
    """Classical RK4, used as a near-exact reference for the ODE solution (project: as in sample_ab2)."""
    h = 1.0 / n_steps
    x = x0
    for k in range(n_steps):
        t = k * h
        k1 = model(x, t)
        k2 = model(x + 0.5 * h * k1, t + 0.5 * h)
        k3 = model(x + 0.5 * h * k2, t + 0.5 * h)
        k4 = model(x + h * k3, t + h)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if project is not None:
            x = project(x, t + h)
    return x

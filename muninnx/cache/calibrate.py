"""Calibration: collect (score, reuse-error) pairs and build a conformal upper bound U(s).

>>> YOU IMPLEMENT THIS (GUIDE.md, step 5) <<<

Paper Sec. IV-C, adapted (staleness-aware; see GUIDE.md "Finding 1" for why):
  1. Run N_cal full chains from fresh noise, storing probe features F_k and velocities v_k
     at every step. For each cacheable step k and each lag l = 1..max_lag (with k - l >= 0)
     record one pair:
        s   = score(F_k, F_{k-l})           (features now vs. features when v was cached)
        eps = || v_k - v_{k-l} ||           (error of reusing a velocity that is l steps stale)
  2. Fit a regressor m(s) ~ eps on a train split (start with a 1-D linear fit).
  3. Split conformal on the held-out split: residuals r_i = eps_i - m(s_i),
     q = the ceil((n+1)(1-alpha_step))-th smallest residual (q = +inf if that rank > n).
  4. U(s) = max(m(s) + q, 0).  Then P(eps <= U(s)) >= 1 - alpha_step under exchangeability.
"""
import torch


@torch.no_grad()
def collect_calibration_pairs(model, solver, x0, steps, max_lag=10, omega=1e-3):
    """Run the calibration chains.

    Args:
        model: velocity net (eval mode)
        solver: Solver (EulerSolver for v0)
        x0: (N_cal, ...) noise, one row per calibration chain
        steps: iterable of step indices k where reuse is allowed (T_cache); k >= 1
        max_lag: largest staleness l considered (must match the controller's cfg.max_lag)
        omega: score denominator constant
    Returns:
        dict with flat 1-D CPU tensors of equal length:
            "s":   probe scores score(F_k, F_{k-l})
            "eps": per-sample reuse errors ||v_k - v_{k-l}||   (norm over all non-batch dims)
            "k":   step index of each pair
            "lag": staleness l of each pair
    Hints: muninnx.cache.probe.probe_features, muninnx.cache.score.relative_l1_score.
    Storing all F_k and v_k for the whole chain is fine at this scale.
    """
    # TODO(you)
    raise NotImplementedError


class ConformalUpperBound:
    """U(s) = max(m(s) + q, 0) with m linear, q a split-conformal residual quantile."""

    def __init__(self):
        self.coef = None   # (slope, intercept) of m
        self.q = None      # conformal quantile

    def fit(self, s, eps, alpha_step, train_frac=0.5, seed=0):
        """Random train/held-out split, least-squares fit of m on train, quantile q on held-out.

        Returns self.
        """
        # TODO(you)
        raise NotImplementedError

    def __call__(self, s):
        """Upper bound U(s) for a tensor of scores s (any shape)."""
        # TODO(you)
        raise NotImplementedError

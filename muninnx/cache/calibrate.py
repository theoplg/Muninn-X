"""Calibration: collect (score, reuse-error) pairs and build a conformal upper bound U(s).


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
import math

import torch
from muninnx.cache.probe import probe_features
from muninnx.cache.score import relative_l1_score



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
    # 1. full-compute chains: store probe features and velocities at every step
    F, V = [], []
    x = x0
    for k in range(solver.n_steps):
        t = solver.t(k)
        F.append(probe_features(model, x, t))
        V.append(model(x, t))
        x = solver.step(x, V[k], k)

    # 2. one (s, eps) pair per chain, cacheable step k and lag l = 1..max_lag (k - l >= 0)
    S, E, Ks, Ls = [], [], [], []
    n = x0.shape[0]
    for k in steps:
        for l in range(1, min(max_lag, k) + 1):
            S.append(relative_l1_score(F[k], F[k - l], omega))
            E.append((V[k] - V[k - l]).flatten(1).norm(dim=1))
            Ks.append(torch.full((n,), k, dtype=torch.long))
            Ls.append(torch.full((n,), l, dtype=torch.long))

    # 3. flatten into 1-D CPU tensors
    return {
        "s": torch.cat(S).cpu(),
        "eps": torch.cat(E).cpu(),
        "k": torch.cat(Ks),
        "lag": torch.cat(Ls),
    }



class ConformalUpperBound:
    """U(s) = max(m(s) + q, 0) with m linear, q a split-conformal residual quantile."""

    def __init__(self):
        self.coef = None   # (slope, intercept) of m
        self.q = None      # conformal quantile

    def fit(self, s, eps, alpha_step, train_frac=0.5, seed=0):
        """Random train/held-out split, least-squares fit of m on train, quantile q on held-out.

        Returns self.
        """
        s = s.detach().flatten().double().cpu()
        eps = eps.detach().flatten().double().cpu()
        n = s.numel()

        # 1. random split: train (fit m) / held-out (conformal quantile), disjoint
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        n_train = int(train_frac * n)
        tr, cal = perm[:n_train], perm[n_train:]

        # 2. OLS fit eps ~ a*s + b on the train split
        X = torch.stack([s[tr], torch.ones_like(s[tr])], dim=1)       # (n_train, 2)
        sol = torch.linalg.lstsq(X, eps[tr].unsqueeze(1)).solution    # (2, 1)
        a, b = sol[0, 0].item(), sol[1, 0].item()

        # 3. split-conformal quantile of the signed residuals on the held-out split
        r = eps[cal] - (a * s[cal] + b)
        n_cal = r.numel()
        rank = math.ceil((n_cal + 1) * (1 - alpha_step))
        q = math.inf if rank > n_cal else r.sort().values[rank - 1].item()

        self.coef = (a, b)
        self.q = q
        return self

    def __call__(self, s):
        """Upper bound U(s) for a tensor of scores s (any shape)."""
        a, b = self.coef
        return (a * s + b + self.q).clamp_min(0.0)

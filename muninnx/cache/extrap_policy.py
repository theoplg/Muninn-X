"""Extrapolate-or-compute sampler on an AB2 base grid, with pluggable decision rules (E9 / E10).

At each step k >= 2, per sample, the velocity is either computed (1 NFE) or linearly extrapolated from
the last two COMPUTED velocities (at most max_lag consecutive extrapolations); the state update is AB2 on
the velocities actually used (Euler at k = 0). The decision rule is a callback, so the same loop serves:
  - the E9 oracle (true G-weighted cost, free network evaluations),
  - predicted-cost policies (E10): a regressor on features a real sampler can observe,
  - "ghost" chains with random decisions, used to build the E10 training set.

Features (per sample, step k) are built only from what is known before deciding at step k:
  free     : time, lag, history of the last three computes (velocity change rate, endpoint drift rate,
             curvature estimate, first/second-order extrapolation-size proxies), ||v_b||
  probe    : Muninn's relative L1 change score between the probe at x_k and at the last compute
             (+ optionally the full probe vector F_k)
  position : the current point x_k (2D toys only: an optimistic upper bound, not a scalable signal)
"""
import torch

from muninnx.cache.probe import probe_features
from muninnx.cache.score import relative_l1_score

TIERS = ("free", "probe_score", "probe_feat", "position")
_EPS = 1e-8


def _log(x):
    return torch.log(x + _EPS)


def free_features(t, lag, h, H):
    """H: history dict (per-sample tensors) of the last three computes a (older) / b (newest) / c (oldest)."""
    dt_ba = H["t_b"] - H["t_a"]
    slope_ba = (H["v_b"] - H["v_a"]) / dt_ba[:, None]
    rate = slope_ba.norm(dim=1)
    ep_rate = (H["xh_b"] - H["xh_a"]).norm(dim=1) / dt_ba
    dt_ac = (H["t_a"] - H["t_c"]).clamp_min(1e-6)
    slope_ac = (H["v_a"] - H["v_c"]) / dt_ac[:, None]
    curv = torch.where(H["has_c"], (slope_ba - slope_ac).norm(dim=1) / ((H["t_b"] - H["t_c"]) / 2).clamp_min(1e-6),
                       torch.zeros_like(rate))
    since = t - H["t_b"]
    return torch.stack([
        torch.full_like(rate, t), lag.float(), since / h, _log(H["v_b"].norm(dim=1)), _log(rate), _log(ep_rate),
        _log(curv), H["has_c"].float(), _log(since * rate), _log(since ** 2 * curv),
    ], dim=1)


@torch.no_grad()
def run_extrap_ab2(model, x0, n_steps, decide, max_lag=10, need_probe=False, need_true=False, record=None):
    """decide(k, ctx) -> bool tensor (B,): True = extrapolate (only allowed rows are kept).

    ctx keys: t, x, lag, v_ext, feats (dict tier -> (B, F)), and if need_true: v_true.
    record(k, ctx): optional callback (e.g. to log features and targets).
    Returns x_N, nfe (B,), n_extrap (B,).
    """
    N, h = n_steps, 1.0 / n_steps
    B, D, dev = x0.shape[0], x0.shape[1], x0.device
    x = x0
    lag = torch.zeros(B, dtype=torch.long, device=dev)
    nfe = torch.zeros(B, dtype=torch.long, device=dev)
    n_ext = torch.zeros(B, dtype=torch.long, device=dev)
    z, zv = torch.zeros(B, device=dev), torch.zeros(B, D, device=dev)
    H = {"t_a": z.clone(), "t_b": z.clone(), "t_c": z.clone(), "v_a": zv.clone(), "v_b": zv.clone(),
         "v_c": zv.clone(), "xh_a": zv.clone(), "xh_b": zv.clone(), "has_c": torch.zeros(B, dtype=torch.bool, device=dev)}
    F_b, u_prev = None, None

    for k in range(N):
        t = k * h
        F_k = probe_features(model, x, t) if need_probe else None
        if k < 2:
            ext = torch.zeros(B, dtype=torch.bool, device=dev)
            v_true = model(x, t)
            ctx = None
        else:
            v_ext = H["v_b"] + ((t - H["t_b"]) / (H["t_b"] - H["t_a"]))[:, None] * (H["v_b"] - H["v_a"])
            fr = free_features(t, lag, h, H)
            feats = {"free": fr, "position": torch.cat([fr, x], dim=1)}
            if need_probe:
                s = _log(relative_l1_score(F_k, F_b))[:, None]
                feats["probe_score"] = torch.cat([fr, s], dim=1)
                feats["probe_feat"] = torch.cat([fr, s, F_k], dim=1)
            v_true = model(x, t)  # computed for all rows for simplicity; only computed rows count as NFE
            ctx = {"t": t, "k": k, "x": x, "lag": lag, "v_ext": v_ext, "feats": feats,
                   "v_true": v_true if need_true else None}
            if record is not None:
                record(k, {**ctx, "v_true": v_true})
            ext = decide(k, ctx) & (lag < max_lag)

        rec = ~ext
        u = v_true if k < 2 else torch.where(ext[:, None], v_ext, v_true)
        # shift history on computed rows: c <- a <- b <- new
        tk = torch.full((B,), t, device=dev)
        xh_new = x + (1 - t) * v_true
        if k == 0:
            H["t_b"], H["v_b"], H["xh_b"] = tk, v_true, xh_new
        else:
            r1 = rec[:, None]
            H["has_c"] = torch.where(rec, k >= 2, H["has_c"])
            H["t_c"], H["v_c"] = torch.where(rec, H["t_a"], H["t_c"]), torch.where(r1, H["v_a"], H["v_c"])
            H["t_a"], H["v_a"], H["xh_a"] = (torch.where(rec, H["t_b"], H["t_a"]), torch.where(r1, H["v_b"], H["v_a"]),
                                             torch.where(r1, H["xh_b"], H["xh_a"]))
            H["t_b"], H["v_b"], H["xh_b"] = (torch.where(rec, tk, H["t_b"]), torch.where(r1, v_true, H["v_b"]),
                                             torch.where(r1, xh_new, H["xh_b"]))
        if need_probe:
            F_b = F_k if F_b is None else torch.where(rec[:, None], F_k, F_b)
        nfe += rec.long()
        n_ext += ext.long()
        lag = torch.where(rec, torch.zeros_like(lag), lag + 1)
        x = x + h * (u if u_prev is None else 1.5 * u - 0.5 * u_prev)
        u_prev = u
    return x, nfe, n_ext

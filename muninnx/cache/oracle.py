"""Oracle controller: Muninn's decision rule with the TRUE reuse error instead of U(s).

Diagnostic only (not a sampler): it evaluates the network at every step to know the true
velocity, so its NFE counts only the evaluations the real controller would have paid for.

At a cacheable step the oracle reuses iff  gamma * L_k * ||v(x_k) - v_cache|| <= budget
(and lag < max_lag). Since the error is measured on the chain actually being run, the
budget accounting is an exact triangle-inequality bound: d <= eta holds deterministically.

Comparing it with Muninn at equal NFE isolates the cost of the score + conformal bound;
comparing it with FewSteps tells whether velocity reuse can win at all.
"""
import torch

from muninnx.cache.bound import sensitivity_weights
from muninnx.cache.controller import cache_steps


@torch.no_grad()
def oracle_sample(model, solver, x0, cfg):
    N = solver.n_steps
    T_cache = set(cache_steps(N, cfg))
    L = sensitivity_weights(solver).tolist()
    B, dev = x0.shape[0], x0.device

    x = x0
    budget = torch.full((B,), float(cfg.eta), device=dev)
    lag = torch.zeros(B, dtype=torch.long, device=dev)
    nfe = torch.zeros(B, dtype=torch.long, device=dev)
    reused = torch.zeros(N, B, dtype=torch.bool, device=dev)
    v_cache = None

    for k in range(N):
        v_true = model(x, solver.t(k))  # free for the oracle
        if v_cache is None or k not in T_cache:
            rec = torch.ones(B, dtype=torch.bool, device=dev)
            c = torch.zeros(B, device=dev)
        else:
            eps = (v_true - v_cache).flatten(1).norm(dim=1)
            c = cfg.gamma * L[k] * eps
            rec = ~((c <= budget) & (lag < cfg.max_lag))

        m = rec.view(-1, *[1] * (x.ndim - 1))
        v_cache = v_true if v_cache is None else torch.where(m, v_true, v_cache)
        nfe += rec.long()
        lag = torch.where(rec, torch.zeros_like(lag), lag + 1)
        budget = torch.where(rec, budget, budget - c)
        reused[k] = ~rec
        x = solver.step(x, v_cache, k)

    return x, {"nfe": nfe, "reused": reused, "bound_used": cfg.eta - budget, "budget_left": budget}


@torch.no_grad()
def oracle_extrap_sample(model, solver, x0, cfg):
    """Oracle with linear EXTRAPOLATION instead of plain reuse.

    Keeps the last two computed velocities (t_a, v_a), (t_b, v_b) per sample. At a cacheable step
    the candidate is v_ext = v_b + (t_k - t_b) (v_b - v_a) / (t_b - t_a). It is used iff
    gamma * L_k * ||v(x_k) - v_ext|| <= budget and fewer than max_lag consecutive non-recomputes.
    The step is always the Euler step of the reference chain: x_{k+1} = x_k + dt_k * v_used.
    """
    N = solver.n_steps
    T_cache = set(cache_steps(N, cfg))
    L = sensitivity_weights(solver).tolist()
    B, dev = x0.shape[0], x0.device
    col = lambda m: m.view(-1, *[1] * (x0.ndim - 1))

    x = x0
    budget = torch.full((B,), float(cfg.eta), device=dev)
    lag = torch.zeros(B, dtype=torch.long, device=dev)
    nfe = torch.zeros(B, dtype=torch.long, device=dev)
    reused = torch.zeros(N, B, dtype=torch.bool, device=dev)
    t_a = t_b = v_a = v_b = None

    for k in range(N):
        t = solver.t(k)
        v_true = model(x, t)  # free for the oracle
        if v_a is None or k not in T_cache:  # need two computed velocities to extrapolate
            rec = torch.ones(B, dtype=torch.bool, device=dev)
            c = torch.zeros(B, device=dev)
            v_ext = v_true
        else:
            v_ext = v_b + col((t - t_b) / (t_b - t_a)) * (v_b - v_a)
            c = cfg.gamma * L[k] * (v_true - v_ext).flatten(1).norm(dim=1)
            rec = (c > budget) | (lag >= cfg.max_lag)

        v_used = torch.where(col(rec), v_true, v_ext)
        # shift history on recompute rows
        tk = torch.full((B,), t, device=dev)
        if v_b is None:
            t_b, v_b = tk.clone(), v_true.clone()
        else:
            if v_a is None:
                t_a, v_a = t_b.clone(), v_b.clone()
            t_a = torch.where(rec, t_b, t_a)
            v_a = torch.where(col(rec), v_b, v_a)
            t_b = torch.where(rec, tk, t_b)
            v_b = torch.where(col(rec), v_true, v_b)

        nfe += rec.long()
        lag = torch.where(rec, torch.zeros_like(lag), lag + 1)
        budget = torch.where(rec, budget, budget - c)
        reused[k] = ~rec
        x = solver.step(x, v_used, k)

    return x, {"nfe": nfe, "reused": reused, "bound_used": cfg.eta - budget, "budget_left": budget}

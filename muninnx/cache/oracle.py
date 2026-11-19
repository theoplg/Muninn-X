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

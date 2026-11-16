import torch

from muninnx.cache.controller import MuninnConfig, cache_steps, muninn_sample
from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver

N = 20


def run(model, eta, c, constant_bound, max_lag=1000, batch=8):
    torch.manual_seed(1)
    x0 = torch.randn(batch, 2)
    cfg = MuninnConfig(eta=eta, alpha=0.05, forbid_prefix=3, forbid_suffix=3, max_lag=max_lag)
    x, info = muninn_sample(model, EulerSolver(N), x0, constant_bound(c), cfg)
    return x0, cfg, x, info


def test_zero_budget_reproduces_full_chain(small_model, constant_bound):
    x0, _, x, info = run(small_model, eta=0.0, c=1.0, constant_bound=constant_bound)
    assert torch.equal(x, sample_full(small_model, EulerSolver(N), x0))
    assert (info["nfe"] == N).all()
    assert not info["reused"].any()


def test_infinite_budget_reuses_all_cacheable_steps(small_model, constant_bound):
    _, cfg, _, info = run(small_model, eta=1e9, c=1.0, constant_bound=constant_bound)
    steps = cache_steps(N, cfg)
    assert (info["nfe"] == N - len(steps)).all()
    assert info["reused"].shape == (N, 8)
    assert info["reused"][steps].all()
    others = [k for k in range(N) if k not in steps]
    assert not info["reused"][others].any()


def test_max_lag_caps_consecutive_reuse(small_model, constant_bound):
    # cacheable steps 3..16 (14 steps); with max_lag=3: reuse 3, recompute 1, ... -> 11 reuses
    _, _, _, info = run(small_model, eta=1e9, c=1.0, constant_bound=constant_bound, max_lag=3)
    assert (info["reused"].sum(0) == 11).all()
    r = info["reused"][:, 0].int().tolist()
    longest = max(len(run_) for run_ in "".join(map(str, r)).split("0"))
    assert longest == 3


def test_budget_accounting(small_model, constant_bound):
    # c_hat = gamma * L_k * U = 1 * 0.05 * 1 = 0.05 per reuse -> eta = 0.12 allows exactly 2 reuses
    _, _, _, info = run(small_model, eta=0.12, c=1.0, constant_bound=constant_bound)
    assert (info["budget_left"] >= 0).all()
    assert torch.allclose(info["bound_used"] + info["budget_left"], torch.full((8,), 0.12))
    assert (info["reused"].sum(0) == 2).all()
    assert (info["nfe"] == N - 2).all()

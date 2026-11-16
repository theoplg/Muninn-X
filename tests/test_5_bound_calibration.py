import torch

from muninnx.cache.bound import sensitivity_weights
from muninnx.cache.calibrate import ConformalUpperBound, collect_calibration_pairs
from muninnx.solvers.base import Solver
from muninnx.solvers.euler import EulerSolver


class HalvingSolver(Solver):
    """Fake solver with K_k = 0.5, Lp_k = 1 to test the product structure."""

    def coeffs(self, k):
        return 0.5, 1.0


def test_sensitivity_euler_uniform():
    L = sensitivity_weights(EulerSolver(10))
    assert L.shape == (10,)
    assert torch.allclose(L, torch.full((10,), 0.1))


def test_sensitivity_product():
    N = 6
    L = sensitivity_weights(HalvingSolver(N))
    expected = torch.tensor([0.5 ** (N - 1 - k) for k in range(N)])
    assert torch.allclose(L, expected)


def test_conformal_coverage_synthetic():
    g = torch.Generator().manual_seed(0)
    def draw(n):
        s = torch.rand(n, generator=g)
        eps = 2 * s + 0.1 * torch.randn(n, generator=g).abs()
        return s, eps
    s, eps = draw(4000)
    bound = ConformalUpperBound().fit(s, eps, alpha_step=0.1, seed=0)
    s_t, eps_t = draw(20000)
    U = bound(s_t)
    assert (U >= 0).all()
    coverage = (eps_t <= U).float().mean().item()
    assert coverage >= 0.88, coverage
    assert coverage <= 0.95, f"bound is too loose: {coverage}"


def test_collect_pairs_shapes_and_values(small_model):
    solver = EulerSolver(10)
    steps = [1, 3, 4, 5]
    x0 = torch.randn(6, 2)
    pairs = collect_calibration_pairs(small_model, solver, x0, steps, max_lag=2)
    # k=1 has only lag 1 (k - l >= 0); k=3,4,5 have lags 1 and 2  -> 7 pairs per chain
    for key in ("s", "eps", "k", "lag"):
        assert pairs[key].shape == (6 * 7,)
    assert set(pairs["k"].tolist()) == set(steps)
    assert set(pairs["lag"].tolist()) == {1, 2}
    assert (pairs["s"] >= 0).all() and (pairs["eps"] >= 0).all()

    # recompute eps at step k=3 by hand along the full chain
    x, vs = x0, []
    for k in range(4):
        v = small_model(x, solver.t(k))
        vs.append(v)
        x = solver.step(x, v, k)
    for lag in (1, 2):
        manual = (vs[3] - vs[3 - lag]).norm(dim=1)
        got = pairs["eps"][(pairs["k"] == 3) & (pairs["lag"] == lag)]
        assert torch.allclose(got.sort().values, manual.sort().values, atol=1e-5)

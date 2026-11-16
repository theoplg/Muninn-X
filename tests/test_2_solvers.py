import math

import torch

from muninnx.sampling import sample_full
from muninnx.solvers.euler import EulerSolver


def test_constant_field_is_exact():
    c = torch.tensor([[1.0, -2.0]])
    x0 = torch.randn(3, 2)
    x1 = sample_full(lambda x, t: c.expand_as(x), EulerSolver(10), x0)
    assert torch.allclose(x1, x0 + c, atol=1e-5)


def test_linear_field_converges():
    # dx/dt = x  ->  x(1) = e * x(0); Euler error is O(dt)
    x0 = torch.randn(4, 2)
    x1 = sample_full(lambda x, t: x, EulerSolver(2000), x0)
    assert torch.allclose(x1, math.e * x0, rtol=2e-3)


def test_coeffs_uniform_and_nonuniform():
    s = EulerSolver(10)
    for k in range(10):
        K, Lp = s.coeffs(k)
        assert abs(K - 1.0) < 1e-6 and abs(Lp - 0.1) < 1e-6
    s = EulerSolver(3, ts=[0.0, 0.5, 0.9, 1.0])
    assert abs(s.coeffs(1)[1] - 0.4) < 1e-6

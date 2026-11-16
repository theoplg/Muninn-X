import math

import torch

from muninnx.eval.paired import summarize, trajectory_distance


def test_distance_points_and_trajectories():
    a = torch.randn(5, 2)
    assert torch.allclose(trajectory_distance(a, a), torch.zeros(5))
    b = a + torch.tensor([3.0, 4.0])
    assert torch.allclose(trajectory_distance(a, b), torch.full((5,), 5.0))
    # trajectories: (B, H, D), ||ones(4,2)||_F / sqrt(4) = sqrt(8)/2 = sqrt(2)
    A = torch.zeros(3, 4, 2)
    assert torch.allclose(trajectory_distance(A, A + 1), torch.full((3,), math.sqrt(2)))


def test_summarize():
    d = torch.tensor([0.0, 0.2, 0.05, 0.3])
    nfe = torch.tensor([50, 50, 25, 25])
    r = summarize(d, eta=0.1, nfe=nfe, n_steps=100)
    assert abs(r["E_d"] - 0.1375) < 1e-6
    assert abs(r["p_viol"] - 0.5) < 1e-6
    assert abs(r["nfe_mean"] - 37.5) < 1e-6
    assert abs(r["speedup_nfe"] - 100 / 37.5) < 1e-6
    lo, hi = r["p_viol_ci"]
    assert lo <= 0.5 <= hi

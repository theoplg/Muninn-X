import torch

from muninnx.fm.loss import fm_loss, interpolate, target_velocity


def test_interpolate_endpoints_and_midpoint():
    x0, x1 = torch.randn(5, 2), torch.randn(5, 2)
    assert torch.allclose(interpolate(x0, x1, torch.zeros(5)), x0)
    assert torch.allclose(interpolate(x0, x1, torch.ones(5)), x1)
    assert torch.allclose(interpolate(x0, x1, torch.full((5,), 0.5)), 0.5 * (x0 + x1))


def test_interpolate_broadcasts_over_trajectory_dims():
    x0, x1 = torch.randn(4, 8, 3), torch.randn(4, 8, 3)
    t = torch.rand(4)
    xt = interpolate(x0, x1, t)
    assert xt.shape == x0.shape
    assert torch.allclose(xt[2], (1 - t[2]) * x0[2] + t[2] * x1[2])


def test_target_velocity():
    x0, x1 = torch.randn(5, 2), torch.randn(5, 2)
    assert torch.allclose(target_velocity(x0, x1), x1 - x0)


def test_fm_loss_scalar_differentiable_reproducible(small_model):
    x1 = torch.randn(64, 2)
    l1 = fm_loss(small_model, x1, generator=torch.Generator().manual_seed(3))
    l2 = fm_loss(small_model, x1, generator=torch.Generator().manual_seed(3))
    assert l1.ndim == 0 and l1.item() >= 0
    assert torch.allclose(l1, l2)
    l1.backward()
    assert small_model.head.weight.grad is not None

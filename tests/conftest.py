import pytest
import torch

from muninnx.models.mlp import VelocityMLP


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    return VelocityMLP(dim=2, hidden=32, depth=2).eval()


class ConstantBound:
    """Stand-in for a fitted ConformalUpperBound: U(s) = c for every s."""

    def __init__(self, c):
        self.c = c

    def __call__(self, s):
        return torch.full_like(torch.as_tensor(s, dtype=torch.float32), self.c)


@pytest.fixture
def constant_bound():
    return ConstantBound

import torch

from muninnx.cache.probe import probe_features
from muninnx.cache.score import relative_l1_score


def test_probe_shape_and_matches_stem(small_model):
    x = torch.randn(7, 2)
    f = probe_features(small_model, x, 0.3)
    assert f.shape == (7, 32)
    ref = small_model.stem(small_model.embed_input(x, 0.3))
    assert torch.allclose(f, ref)


def test_score_values():
    f_prev = torch.ones(2, 4)
    assert torch.allclose(relative_l1_score(f_prev.clone(), f_prev), torch.zeros(2))
    s = relative_l1_score(2 * f_prev, f_prev, omega=1.0)  # |diff|_1 = 4, |prev|_1 = 4
    assert s.shape == (2,)
    assert torch.allclose(s, torch.full((2,), 4.0 / 5.0))

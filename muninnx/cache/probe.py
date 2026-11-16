"""Cheap probe features F_k = Psi(x_k, t_k).   >>> YOU IMPLEMENT THIS (GUIDE.md, step 3) <<<

The paper runs only a short prefix of the network ("Stem-1") and pools it.
For VelocityMLP the prefix is `model.stem(model.embed_input(x, t))`.
For bigger models later (U-Net / transformer) you would use a forward hook or
an early-exit, and mean-pool over tokens / time.
"""
import torch


@torch.no_grad()
def probe_features(model, x, t):
    """Return probe features of shape (B, D_probe).

    Must be MUCH cheaper than model(x, t): only run the stem.
    """
    # TODO(you)
    raise NotImplementedError

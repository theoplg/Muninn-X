"""Probe change score (paper Eq. 6).   >>> YOU IMPLEMENT THIS (GUIDE.md, step 3) <<<

    s_k = || F_k - F_{k-1} ||_1 / ( || F_{k-1} ||_1 + omega )

(The paper indexes time backwards, so its F_{t+1} is our F_{k-1}: the previous step.)
"""
import torch


def relative_l1_score(f_cur, f_prev, omega=1e-3):
    """f_cur, f_prev: (B, D) probe features. Returns (B,) scores, >= 0."""
    # TODO(you)
    raise NotImplementedError

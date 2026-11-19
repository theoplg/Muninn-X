"""Probe change score (paper Eq. 6).  

    s_k = || F_k - F_{k-1} ||_1 / ( || F_{k-1} ||_1 + omega )

(The paper indexes time backwards, so its F_{t+1} is our F_{k-1}: the previous step.)
"""
import torch


def relative_l1_score(f_cur, f_prev, omega=1e-3):
    """f_cur, f_prev: (B, D) probe features. Returns (B,) scores, >= 0."""
    s_k = (f_cur - f_prev).abs().sum(dim=1) / (f_prev.abs().sum(dim=1) + omega) # sum over features
    return s_k 

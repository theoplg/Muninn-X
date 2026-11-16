"""Solver interface for flow-matching ODEs  dx/dt = v_theta(x, t),  t: 0 (noise) -> 1 (data).

Index convention used everywhere in this repo
---------------------------------------------
- Time grid  ts[0] = 0 < ts[1] < ... < ts[N] = 1   (N = n_steps)
- Step k  (k = 0..N-1) maps  x_k -> x_{k+1}  using the network evaluated at (x_k, ts[k]).
- Muninn view of one step (paper's Phi_t):   x_{k+1} = Phi_k(x_k, v_k)
  and the per-step Lipschitz constants of Assumption 1:
      ||Phi_k(x, v) - Phi_k(x', v')|| <= K_k ||x - x'|| + Lp_k ||v - v'||
"""
import torch


class Solver:
    def __init__(self, n_steps, ts=None):
        self.n_steps = n_steps
        self.ts = torch.linspace(0.0, 1.0, n_steps + 1) if ts is None else torch.as_tensor(ts, dtype=torch.float32)
        assert len(self.ts) == n_steps + 1

    def dt(self, k):
        return float(self.ts[k + 1] - self.ts[k])

    def t(self, k):
        return float(self.ts[k])

    def step(self, x, v, k):
        """Return x_{k+1} = Phi_k(x_k, v)."""
        raise NotImplementedError

    def coeffs(self, k):
        """Return (K_k, Lp_k) for step k (see module docstring)."""
        raise NotImplementedError

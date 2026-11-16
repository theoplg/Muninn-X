"""Explicit Euler solver for flow matching."""
from .base import Solver


class EulerSolver(Solver):
    """x_{k+1} = x_k + dt_k * v,   dt_k = ts[k+1] - ts[k]."""

    def step(self, x, v, k):
        """One Euler step.

        Args:
            x: (B, ...) current state x_k
            v: (B, ...) velocity to use (freshly computed OR reused from the cache)
            k: step index (use self.dt(k))
        Returns:
            x_{k+1}, same shape as x.
        """
        return x + self.dt(k)*v

    def coeffs(self, k):
        """Lipschitz constants (K_k, Lp_k) of Phi_k(x, v) = x + dt_k * v.
        Returns:
            (K_k, Lp_k) as python floats.
        """
        K_k = 1 
        Lp_k = abs(self.dt(k))
        return (float(K_k), float(Lp_k))

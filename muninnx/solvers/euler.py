"""Explicit Euler solver for flow matching.   >>> YOU IMPLEMENT THIS (GUIDE.md, step 2) <<<"""
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
        # TODO(you)
        raise NotImplementedError

    def coeffs(self, k):
        """Lipschitz constants (K_k, Lp_k) of Phi_k(x, v) = x + dt_k * v.

        Hint: differentiate Phi_k w.r.t. x and w.r.t. v. Both are constants here.
        Returns:
            (K_k, Lp_k) as python floats.
        """
        # TODO(you)
        raise NotImplementedError

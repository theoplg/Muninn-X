"""Error propagation bound (paper Eqs. 1-3).   >>> YOU IMPLEMENT THIS (GUIDE.md, step 5) <<<

Recursion (our forward-time indexing, Delta_k = x_k^full - x_k^cached, Delta_0 = 0):
    ||Delta_{k+1}|| <= K_k ||Delta_k|| + Lp_k ||e_k||
where e_k is the velocity error injected at step k (0 if v was recomputed).

Unrolled to the final state x_N:
    ||Delta_N|| <= sum_k  L_k ||e_k||,     L_k = Lp_k * prod_{j=k+1}^{N-1} K_j
"""
import torch


def sensitivity_weights(solver):
    """Return a (N,) float tensor with L_k for k = 0..N-1 (uses solver.coeffs(k)).

    Sanity check: for Euler on a uniform grid every L_k equals dt.
    """
    N = solver.n_steps
    Ls = [0.0] * N
    prod = 1.0  # prod_{j=k+1}^{N-1} K_j, empty product (=1) at k = N-1
    for k in reversed(range(N)):
        K_k, Lp_k = solver.coeffs(k)
        Ls[k] = Lp_k * prod
        prod *= K_k  # now covers j >= k, i.e. the product needed for step k-1
    return torch.tensor(Ls, dtype=torch.float32)

    

        
        


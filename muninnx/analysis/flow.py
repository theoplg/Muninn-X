"""Jacobian tools for analysing error propagation along a sampling chain (see THEORY_NOTES.md)."""
import torch


def jacobian(model, x, t):
    """(B, D, D) with J[b, i, j] = d v_i / d x_j. Exact for per-sample models (batch rows independent)."""
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        v = model(x, t)
        rows = [torch.autograd.grad(v[:, i].sum(), x, retain_graph=i < v.shape[1] - 1)[0]
                for i in range(v.shape[1])]
    return torch.stack(rows, dim=1).detach()


def flow_jacobians(model, xs, ts, h):
    """Flow Jacobians G_{k+1->N} = (I + h J_{N-1}) ... (I + h J_{k+1}) along a chain.

    xs: list of N states x_0..x_{N-1} (each (B, D)), ts: list of N times. Returns (N, B, D, D),
    where entry k maps a perturbation made just after step k to its effect on the final state.
    """
    N = len(xs)
    B, D = xs[0].shape
    I = torch.eye(D, device=xs[0].device).expand(B, D, D)
    out = [None] * N
    P = I.clone()
    for k in reversed(range(N)):
        out[k] = P
        P = P @ (I + h * jacobian(model, xs[k], ts[k]))
    return torch.stack(out)

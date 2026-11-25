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


def flow_gain(model, xs, ts, h, vecs=None, chunk=500):
    """Scalable version of flow_jacobians for high-dimensional states (never stores the (N, B, D, D) tensor).

    For each step k and sample b along the chain xs (N states of shape (B, D)), returns
        op[k, b] = ||G_{k+1->N}||_op                  (largest amplification)
        gv[k, b] = ||G_{k+1->N} vecs[k, b]||          (if vecs (N, B, D) is given: effect of a specific error)
    with G_{k+1->N} = (I + h J_{N-1}) ... (I + h J_{k+1}). Works chunk by chunk over the batch; memory per
    chunk is N * chunk * D * D floats (100 * 500 * 64 * 64 * 4 B = 0.8 GB for D = 64).
    """
    N = len(xs)
    B, D = xs[0].shape
    op = torch.empty(N, B, device=xs[0].device)
    gv = torch.empty(N, B, device=xs[0].device) if vecs is not None else None
    for s in range(0, B, chunk):
        sl = slice(s, min(s + chunk, B))
        J = [jacobian(model, xs[k][sl], ts[k]) for k in range(N)]
        I = torch.eye(D, device=xs[0].device).expand(J[0].shape)
        P = I.clone()
        for k in reversed(range(N)):
            op[k, sl] = torch.linalg.matrix_norm(P, ord=2)
            if vecs is not None:
                gv[k, sl] = (P @ vecs[k][sl][..., None]).squeeze(-1).norm(dim=1)
            P = P @ (I + h * J[k])
        del J
    return op, gv

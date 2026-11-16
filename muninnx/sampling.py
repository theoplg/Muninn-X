"""Full-compute (reference) sampling: one network evaluation per step."""
import torch


@torch.no_grad()
def sample_full(model, solver, x0, return_traj=False):
    """Integrate dx/dt = model(x, t) from x0 (noise) with `solver`.

    Returns x_N (and the (N+1, B, ...) trajectory if return_traj).
    """
    x = x0
    traj = [x]
    for k in range(solver.n_steps):
        v = model(x, solver.t(k))
        x = solver.step(x, v, k)
        if return_traj:
            traj.append(x)
    if return_traj:
        return x, torch.stack(traj)
    return x

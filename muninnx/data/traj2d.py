"""Synthetic multimodal 2D trajectories (milestone M0.5): paths around an obstacle.

Each sample is a trajectory of H points in 2D, shape (H, 2), going from a start area on the left to a goal
area on the right, passing either ABOVE or BELOW a disk obstacle at the origin. The two homotopy classes
are the "modes": a sample near the decision between them is exactly where error amplification is
expected (THEORY_NOTES §4). Paths are cubic Bezier curves with random start / goal / control points;
any path that enters the obstacle (plus a margin) is rejected and resampled.
"""
import torch

OBSTACLE_RADIUS = 1.0
MARGIN = 0.15


def _bezier(p0, p1, p2, p3, s):
    """Cubic Bezier at parameters s (H,), control points (n, 2) -> (n, H, 2)."""
    s = s[None, :, None]
    return ((1 - s) ** 3 * p0[:, None] + 3 * (1 - s) ** 2 * s * p1[:, None]
            + 3 * (1 - s) * s ** 2 * p2[:, None] + s ** 3 * p3[:, None])


def _candidates(n, H, device, generator):
    u = lambda *shape: torch.rand(*shape, device=device, generator=generator)
    side = torch.where(u(n) < 0.5, -1.0, 1.0)
    start = torch.stack([-3.0 + 0.4 * (u(n) - 0.5), 1.2 * (u(n) - 0.5)], dim=1)
    goal = torch.stack([3.0 + 0.4 * (u(n) - 0.5), 1.2 * (u(n) - 0.5)], dim=1)
    c1 = torch.stack([-1.6 + 0.6 * (u(n) - 0.5), side * (2.0 + 0.8 * u(n))], dim=1)
    c2 = torch.stack([1.6 + 0.6 * (u(n) - 0.5), side * (2.0 + 0.8 * u(n))], dim=1)
    s = torch.linspace(0, 1, H, device=device)
    return _bezier(start, c1, c2, goal, s), side


def sample_traj(n, H=32, device=None, generator=None, return_side=False):
    """n trajectories of shape (H, 2) that avoid the obstacle. Optionally the side (+1 above, -1 below)."""
    device = device or "cpu"
    out, sides, have = [], [], 0
    while have < n:
        x, side = _candidates(2 * (n - have) + 16, H, device, generator)
        ok = (x.norm(dim=2).min(dim=1).values > OBSTACLE_RADIUS + MARGIN)
        out.append(x[ok])
        sides.append(side[ok])
        have += int(ok.sum())
    x, side = torch.cat(out)[:n], torch.cat(sides)[:n]
    return (x, side) if return_side else x


def path_side(x):
    """Mode label of trajectories (B, H, 2): +1 if the path passes above the obstacle, -1 below.
    Uses the signed y-coordinate where the path crosses x = 0 (nearest point in x)."""
    i = x[..., 0].abs().argmin(dim=1)
    return torch.sign(x[torch.arange(x.shape[0], device=x.device), i, 1])


def collision_rate(x):
    """Fraction of trajectories with at least one point inside the obstacle (no margin)."""
    return (x.norm(dim=2).min(dim=1).values < OBSTACLE_RADIUS).float().mean().item()

"""Maze geometry for Maze2D (gymnasium-robotics PointMaze maps, cell size 1, maze centred at the origin).

Cell (i, j) of the map (row i from the top, column j) has centre
    x = (j + 0.5) - W / 2,   y = H / 2 - (i + 0.5)            (W, H: map width / height in cells),
the convention of gymnasium_robotics.envs.maze.Maze.make_maze with maze_size_scaling = 1.
"""
import torch

MAPS = {
    "umaze": ["#####", "#...#", "###.#", "#...#", "#####"],
    "medium": ["########", "#..##..#", "#..#...#", "##...###", "#..#...#", "#.#..#.#", "#...#..#", "########"],
    "large": ["############", "#....#.....#", "#.##.#.#.#.#", "#......#...#", "#.####.###.#",
              "#..#.#.....#", "##.#.#.#.###", "#..#...#...#", "############"],
}


def wall_grid(name, device="cpu"):
    return torch.tensor([[c == "#" for c in row] for row in MAPS[name]], device=device)


def in_wall(p, grid):
    """p (..., 2) positions -> bool: inside a wall cell, or outside the map."""
    Hc, Wc = grid.shape
    j = torch.floor(p[..., 0] + Wc / 2).long()
    i = torch.floor(Hc / 2 - p[..., 1]).long()
    out = (i < 0) | (i >= Hc) | (j < 0) | (j >= Wc)
    return out | grid[i.clamp(0, Hc - 1), j.clamp(0, Wc - 1)]


def wall_violation(p, grid, margin=0.1):
    """Inside a wall by more than `margin` (walls eroded by margin: the point and its four offsets by
    +-margin along x and y are all in walls). p: (..., 2) -> bool (...)."""
    v = in_wall(p, grid)
    for d in ((margin, 0.0), (-margin, 0.0), (0.0, margin), (0.0, -margin)):
        v = v & in_wall(p + torch.tensor(d, device=p.device), grid)
    return v


def wall_rate(plans, grid, margin=0.1):
    """Fraction of plans (B, H, >=2) with at least one point inside a wall."""
    return wall_violation(plans[..., :2], grid, margin).any(dim=1).float().mean().item()

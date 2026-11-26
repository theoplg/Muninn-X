"""Maze2D (Minari D4RL/pointmaze/*-v2) as fixed-length trajectory windows, Diffuser-style.

Minari splits one long run of the point robot into short episodes, one per goal; consecutive episodes
are exactly continuous (last state of episode k == first state of episode k+1, checked on U-maze), so we
stitch them back into one stream and cut windows of H steps. Each step is
    [x, y, vx, vy, ax, ay]   (position, velocity, action)  -> transition dim 6,
normalised per dimension to [-1, 1] with the data min / max (Diffuser's LimitsNormalizer).
Set MINARI_DATASETS_PATH to the folder holding the downloaded datasets (data/minari in this repo).
"""
import os
from pathlib import Path

import numpy as np
import torch

DATASETS = {"umaze": "D4RL/pointmaze/umaze-v2", "medium": "D4RL/pointmaze/medium-v2", "large": "D4RL/pointmaze/large-v2"}
HORIZONS = {"umaze": 128, "medium": 256, "large": 384}  # Diffuser's maze2d horizons
RESET_JUMP = 0.15  # a step longer than this is a reset of the robot, not motion (data max motion step ~0.06)
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "minari"


STREAM_CACHE = Path(__file__).resolve().parents[2] / "data" / "maze2d_streams"


def load_stream(name):
    """Stitched stream (T, 6) float32 numpy: x, y, vx, vy, ax, ay. Action at step i is the one taken from
    state i; the final state of the whole run (no action) is dropped. Cached as data/maze2d_streams/<name>.npy
    (24 MB, loads instantly; enough to train and evaluate without the Minari dataset)."""
    cache = STREAM_CACHE / f"{name}.npy"
    if cache.exists():
        return np.load(cache)
    stream = _build_stream(name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, stream)
    return stream


def _build_stream(name):
    os.environ.setdefault("MINARI_DATASETS_PATH", str(DEFAULT_PATH))
    import minari
    d = minari.load_dataset(DATASETS[name])
    obs, act = [], []
    for e in d.iterate_episodes():
        o = e.observations["observation"]  # (L + 1, 4)
        obs.append(o[:-1])                 # drop the last state: it is the next episode's first state
        act.append(e.actions)              # (L, 2)
    return np.concatenate([np.concatenate(obs), np.concatenate(act)], axis=1).astype(np.float32)


class Maze2DWindows:
    """Random windows of H consecutive steps from the stitched stream, normalised to [-1, 1], on device."""

    def __init__(self, name="umaze", H=None, device="cpu"):
        self.name, self.H = name, H or HORIZONS[name]
        stream = torch.from_numpy(load_stream(name))
        self.lo, self.hi = stream.min(0).values, stream.max(0).values
        self.data = self.normalize(stream).to(device)
        self.lo, self.hi = self.lo.to(device), self.hi.to(device)
        self.T, self.dim = self.data.shape
        # The stitched stream contains rare resets (position jumps, ~0.1% of windows): only use windows that do
        # not straddle one. A window starting at s uses transitions s .. s+H-2.
        step = (stream[1:, :2] - stream[:-1, :2]).norm(dim=1)
        bad = torch.cat([torch.zeros(1), (step > RESET_JUMP).float().cumsum(0)])  # bad[i] = resets before index i
        starts = torch.arange(self.T - self.H + 1)
        ok = bad[starts + self.H - 1] - bad[starts] == 0
        self.starts = starts[ok].to(device)
        self.n_resets = int((step > RESET_JUMP).sum())

    def normalize(self, x):
        return 2 * (x - self.lo) / (self.hi - self.lo) - 1

    def unnormalize(self, x):
        return (x + 1) / 2 * (self.hi - self.lo) + self.lo

    def sample(self, n, generator=None):
        start = self.starts[torch.randint(0, len(self.starts), (n,), device=self.data.device, generator=generator)]
        idx = start[:, None] + torch.arange(self.H, device=self.data.device)[None]
        return self.data[idx]  # (n, H, 6)

    def stats(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist(), "T": self.T, "H": self.H}


CONTEXT_DIM = 6


def window_context(w):
    """Planning condition of normalised windows (B, H, 6): start state (x, y, vx, vy) at the first step and
    goal position (x, y) at the last step -> (B, 6)."""
    return torch.cat([w[:, 0, :4], w[:, -1, :2]], dim=1)

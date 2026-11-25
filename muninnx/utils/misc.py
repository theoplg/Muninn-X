"""Seeding, device selection, JSON results, checkpoints."""
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if dev.type == "cuda":
        cap_gpu_memory()  # never spill into system RAM (see cap_gpu_memory)
    return dev


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def save_checkpoint(model, cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, path)


def load_checkpoint(path, model_cls, device):
    ckpt = torch.load(path, map_location=device)
    m = ckpt["cfg"]["model"]
    model = model_cls(**m).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["cfg"]


def cap_gpu_memory(fraction=0.9):
    """Hard cap on this process's GPU memory. On Windows/WSL the NVIDIA driver otherwise spills
    over-allocations into shared system RAM instead of raising an out-of-memory error, which can freeze
    and crash the whole machine. With the cap, an oversized allocation fails with a normal CUDA OOM."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)

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
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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

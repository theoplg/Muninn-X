"""Train an epsilon-prediction VP diffusion model (cosine schedule) on a 2D toy, for E14.

    python scripts/train_ddpm_toy.py --dataset checkerboard
"""
import argparse
import time

import torch

from muninnx.data.toy2d import sample_toy
from muninnx.diffusion.ddpm import cosine_alphabar, ddpm_loss
from muninnx.models.mlp import VelocityMLP  # same architecture, used as an epsilon-network
from muninnx.utils.misc import get_device, save_checkpoint, seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="checkerboard")
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    model_cfg = dict(dim=2, hidden=args.hidden, depth=args.depth)
    model = VelocityMLP(**model_cfg).to(device)
    abar = cosine_alphabar(args.T)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        loss = ddpm_loss(model, sample_toy(args.dataset, args.batch, device=device), abar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 5000 == 0 or it == 1:
            print(f"it {it:6d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
    out = args.out or f"runs/ddpm_{args.dataset}/model.pt"
    save_checkpoint(model, {"model": model_cfg, "dataset": args.dataset, "T": args.T, "schedule": "cosine",
                            "args": vars(args)}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

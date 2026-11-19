"""Train a flow-matching velocity MLP on a 2D toy distribution.

    python scripts/train_toy.py --dataset 8gaussians
"""
import argparse
import time

import torch

from muninnx.data.toy2d import sample_toy
from muninnx.fm.loss import fm_loss
from muninnx.models.mlp import VelocityMLP
from muninnx.utils.misc import get_device, save_checkpoint, seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="8gaussians")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls for training throughput
    model_cfg = dict(dim=2, hidden=args.hidden, depth=args.depth)
    model = VelocityMLP(**model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        x1 = sample_toy(args.dataset, args.batch, device=device)  # generated on the GPU
        loss = fm_loss(model, x1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == 1:
            print(f"it {it:6d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")

    out = args.out or f"runs/toy_{args.dataset}/model.pt"
    save_checkpoint(model, {"model": model_cfg, "dataset": args.dataset, "args": vars(args)}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

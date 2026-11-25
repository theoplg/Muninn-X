"""Train a flow-matching velocity backbone on trajectories: the trajectory toy (M0.5) or Maze2D (M1).

    python scripts/train_traj.py --arch unet
    python scripts/train_traj.py --arch maze_unet --dataset maze-umaze --batch 1024 --iters 60000
"""
import argparse
import copy
import time
from pathlib import Path

import torch

from muninnx.data.maze2d import CONTEXT_DIM, Maze2DWindows, window_context
from muninnx.data.traj2d import sample_traj
from muninnx.fm.loss import fm_loss
from muninnx.models.traj_backbones import ARCHS
from muninnx.utils.misc import get_device, save_checkpoint, seed_all

DEFAULTS = {"mlp": dict(lr=1e-3), "unet": dict(lr=3e-4), "transformer": dict(lr=3e-4), "maze_unet": dict(lr=2e-4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=list(ARCHS), required=True)
    p.add_argument("--dataset", default="traj2d", choices=["traj2d", "maze-umaze", "maze-medium", "maze-large"])
    p.add_argument("--H", type=int, default=None, help="horizon (default: 32 for traj2d, Diffuser's for Maze2D)")
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--cond", action="store_true", help="Maze2D: condition on start state + goal position")
    p.add_argument("--init", default=None, help="warm-start from this checkpoint (missing layers keep their init)")
    p.add_argument("--save-every", type=int, default=0, help="also save model_it<step>.pt every N steps")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the saved weights (0 = no EMA; Diffuser uses ~0.995-0.999)")
    p.add_argument("--t-scale", type=float, default=None, help="time-embedding frequency scale (maze_unet; default 1000)")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    lr = args.lr or DEFAULTS[args.arch]["lr"]
    data_stats = None
    if args.dataset == "traj2d":
        H = args.H or 32
        draw = lambda n: sample_traj(n, H=H, device=device)
        model_cfg = {"H": H}
    else:
        ds = Maze2DWindows(args.dataset.split("-")[1], H=args.H, device=device)
        H, draw, data_stats = ds.H, ds.sample, ds.stats()
        model_cfg = {"H": H, "dim": ds.dim}
        if args.t_scale is not None:
            model_cfg["t_scale"] = args.t_scale
        if args.cond:
            model_cfg["cond_dim"] = CONTEXT_DIM
    model = ARCHS[args.arch](**model_cfg).to(device)
    if args.init:
        res = model.load_state_dict(torch.load(args.init, map_location=device)["state_dict"], strict=False)
        print(f"warm-start from {args.init}: new layers {res.missing_keys} | ignored {res.unexpected_keys}")
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / args.warmup) * 0.5 * (1 + torch.cos(torch.tensor(i / args.iters * 3.14159)).item()))
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema > 0 else None
    net = model if args.no_compile else torch.compile(model, dynamic=False)  # fuses the many small ops of these small models
    print(f"arch={args.arch} params={n_params / 1e6:.2f}M lr={lr} batch={args.batch} iters={args.iters} "
          f"compile={not args.no_compile} bf16={not args.no_bf16}")
    default_dir = "runs/traj_toy" if args.dataset == "traj2d" else f"runs/{args.dataset.replace('-', '2d_')}"
    out = args.out or f"{default_dir}/{args.arch}/model.pt"
    ckpt_cfg = {"model": model_cfg, "arch": args.arch, "dataset": args.dataset, "data_stats": data_stats,
                "args": vars(args), "n_params": n_params, "ema": args.ema}
    t0 = time.time()
    for it in range(1, args.iters + 1):
        x1 = draw(args.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_bf16):  # training only; eval is fp32
            if args.cond:
                c = window_context(x1)
                loss = fm_loss(lambda x, t: net(x, t, c), x1)
            else:
                loss = fm_loss(net, x1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:  # EMA with warm-up: decay grows from 0.9 to args.ema
            d = min(args.ema, (1 + it) / (10 + it))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
        if it % 5000 == 0 or it == 1:
            print(f"it {it:6d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
        if args.save_every and it % args.save_every == 0 and it < args.iters:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ema if ema is not None else model, {**ckpt_cfg, "iters_done": it},
                            str(Path(out).with_name(f"model_it{it}.pt")))
    save_checkpoint(ema if ema is not None else model, {**ckpt_cfg, "iters_done": args.iters}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

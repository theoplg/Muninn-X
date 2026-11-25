"""Velocity backbones for trajectories x of shape (B, H, D) (milestone M0.5).

All three share one interface, used by the probe and by feature caching:
    h, aux = encode(x, t)        cheap shallow part (input layers + first block)
    hc     = core(h, aux)        expensive middle part
    v      = decode(hc, h, aux)  cheap last part (may also use the shallow features h)
    forward(x, t) = decode(core(encode(x, t)), ...)
    probe(x, t)   = encode output pooled to (B, C): the Muninn probe ("Stem-1")
Feature caching (later): keep hc from the last full evaluation and only recompute encode + decode.

TrajMLP          : flattened trajectory, time-conditioned MLP (continuity with the 2D toys)
TrajUNet         : temporal 1D U-Net with FiLM time conditioning (Diffuser-style)
TrajTransformer  : one token per time step, DiT-style adaLN-zero blocks
"""
import torch
from torch import nn
from torch.nn import functional as F

from muninnx.models.mlp import SinusoidalEmbedding, as_batch_time


class TimeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(SinusoidalEmbedding(64), nn.Linear(64, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        return self.net(t)


class TrajBackbone(nn.Module):
    def forward(self, x, t):
        h, aux = self.encode(x, t)
        return self.decode(self.core(h, aux), h, aux)

    def probe(self, x, t):
        h, _ = self.encode(x, t)
        return self.pool(h)


# --------------------------------------------------------------------------------------------- MLP
class TrajMLP(TrajBackbone):
    def __init__(self, H=32, dim=2, hidden=512, depth=4, t_dim=64):
        super().__init__()
        self.H, self.dim = H, dim
        self.t_embed = SinusoidalEmbedding(t_dim)
        self.stem = nn.Sequential(nn.Linear(H * dim + t_dim, hidden), nn.SiLU())
        self.body = nn.Sequential(*[m for _ in range(depth - 1) for m in (nn.Linear(hidden, hidden), nn.SiLU())])
        self.head = nn.Linear(hidden, H * dim)

    def encode(self, x, t):
        t = as_batch_time(t, x.shape[0], x.device)
        return self.stem(torch.cat([x.flatten(1), self.t_embed(t)], dim=1)), None

    def core(self, h, aux):
        return self.body(h)

    def decode(self, hc, h, aux):
        return self.head(hc).view(-1, self.H, self.dim)

    def pool(self, h):
        return h


# ------------------------------------------------------------------------------------------ U-Net
class FiLMResBlock1d(nn.Module):
    def __init__(self, cin, cout, t_dim, k=5):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv1d(cin, cout, k, padding=k // 2)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv1d(cout, cout, k, padding=k // 2)
        self.film = nn.Linear(t_dim, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, h, temb):
        z = self.c1(F.silu(self.n1(h)))
        scale, shift = self.film(temb)[..., None].chunk(2, dim=1)
        z = self.c2(F.silu(self.n2(z) * (1 + scale) + shift))
        return z + self.skip(h)


class TrajUNet(TrajBackbone):
    """32 -> 16 -> 8 -> 16 -> 32 over time, channels c, 2c, 4c; skip connections."""

    def __init__(self, H=32, dim=2, c=64, t_dim=128):
        super().__init__()
        self.H, self.dim = H, dim
        self.time = TimeMLP(t_dim)
        self.inp = nn.Conv1d(dim, c, 5, padding=2)
        self.enc = FiLMResBlock1d(c, c, t_dim)
        self.down1, self.b1 = nn.Conv1d(c, 2 * c, 3, stride=2, padding=1), FiLMResBlock1d(2 * c, 2 * c, t_dim)
        self.down2, self.b2 = nn.Conv1d(2 * c, 4 * c, 3, stride=2, padding=1), FiLMResBlock1d(4 * c, 4 * c, t_dim)
        self.mid = FiLMResBlock1d(4 * c, 4 * c, t_dim)
        self.up2, self.b3 = nn.ConvTranspose1d(4 * c, 2 * c, 4, stride=2, padding=1), FiLMResBlock1d(4 * c, 2 * c, t_dim)
        self.up1 = nn.ConvTranspose1d(2 * c, c, 4, stride=2, padding=1)
        self.dec = FiLMResBlock1d(2 * c, c, t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, dim, 5, padding=2))

    def encode(self, x, t):
        temb = self.time(as_batch_time(t, x.shape[0], x.device))
        return self.enc(self.inp(x.transpose(1, 2)), temb), temb

    def core(self, h, temb):
        s1 = self.b1(self.down1(h), temb)
        z = self.mid(self.b2(self.down2(s1), temb), temb)
        z = self.b3(torch.cat([self.up2(z), s1], dim=1), temb)
        return self.up1(z)

    def decode(self, hc, h, temb):
        return self.out(self.dec(torch.cat([hc, h], dim=1), temb)).transpose(1, 2)

    def pool(self, h):
        return h.mean(dim=-1)


# ------------------------------------------------------------------------------------ Transformer
class AdaLNBlock(nn.Module):
    def __init__(self, d, heads, t_dim):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Linear(t_dim, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h, temb):
        s1, b1, g1, s2, b2, g2 = self.ada(temb)[:, None].chunk(6, dim=-1)
        z = self.n1(h) * (1 + s1) + b1
        B, L, D = z.shape
        q, k, v = self.qkv(z).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, D)
        h = h + g1 * self.proj(a)
        z = self.n2(h) * (1 + s2) + b2
        return h + g2 * self.mlp(z)


class TrajTransformer(TrajBackbone):
    def __init__(self, H=32, dim=2, d=128, depth=6, heads=4):
        super().__init__()
        self.H, self.dim = H, dim
        self.time = TimeMLP(d)
        self.inp = nn.Linear(dim, d)
        self.pos = nn.Parameter(0.02 * torch.randn(1, H, d))
        self.blocks = nn.ModuleList([AdaLNBlock(d, heads, d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_ada = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, dim)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)

    def encode(self, x, t):
        temb = self.time(as_batch_time(t, x.shape[0], x.device))
        return self.blocks[0](self.inp(x) + self.pos, temb), temb

    def core(self, h, temb):
        for blk in self.blocks[1:-1]:
            h = blk(h, temb)
        return h

    def decode(self, hc, h, temb):
        z = self.blocks[-1](hc, temb)
        scale, shift = self.final_ada(temb)[:, None].chunk(2, dim=-1)
        return self.out(self.norm(z) * (1 + scale) + shift)

    def pool(self, h):
        return h.mean(dim=1)


class MazeUNet(TrajBackbone):
    """Configurable-depth temporal U-Net (Diffuser's maze2d setting: c = 32, mults = (1, 2, 4, 8)).
    Same encode / core / decode split as TrajUNet: encode = input conv + first block at full resolution,
    core = all down / mid / up levels, decode = last block (with the full-resolution skip) + output conv.
    Like Diffuser it has two mid blocks at the deepest level (TrajUNet has one), so it is not TrajUNet's topology."""

    def __init__(self, H=128, dim=6, c=32, mults=(1, 2, 4, 8), t_dim=128):
        super().__init__()
        assert H % 2 ** (len(mults) - 1) == 0, "H must be divisible by 2^(levels-1)"
        self.H, self.dim = H, dim
        ch = [c * m for m in mults]
        self.time = TimeMLP(t_dim)
        self.inp = nn.Conv1d(dim, ch[0], 5, padding=2)
        self.enc = FiLMResBlock1d(ch[0], ch[0], t_dim)
        self.downs = nn.ModuleList([nn.Conv1d(ch[i - 1], ch[i], 3, stride=2, padding=1) for i in range(1, len(ch))])
        self.down_blocks = nn.ModuleList([FiLMResBlock1d(ch[i], ch[i], t_dim) for i in range(1, len(ch))])
        self.mid = nn.ModuleList([FiLMResBlock1d(ch[-1], ch[-1], t_dim) for _ in range(2)])
        self.ups = nn.ModuleList([nn.ConvTranspose1d(ch[i], ch[i - 1], 4, stride=2, padding=1) for i in range(len(ch) - 1, 0, -1)])
        self.up_blocks = nn.ModuleList([FiLMResBlock1d(2 * ch[i - 1], ch[i - 1], t_dim) for i in range(len(ch) - 1, 1, -1)])
        self.dec = FiLMResBlock1d(2 * ch[0], ch[0], t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv1d(ch[0], dim, 5, padding=2))

    def encode(self, x, t):
        temb = self.time(as_batch_time(t, x.shape[0], x.device))
        return self.enc(self.inp(x.transpose(1, 2)), temb), temb

    def core(self, h, temb):
        z, skips = h, []
        for dn, blk in zip(self.downs, self.down_blocks):
            z = blk(dn(z), temb)
            skips.append(z)
        skips.pop()  # the deepest level feeds the mid blocks, not a skip
        for blk in self.mid:
            z = blk(z, temb)
        for i, up in enumerate(self.ups):
            z = up(z)
            if i < len(self.up_blocks):
                z = self.up_blocks[i](torch.cat([z, skips.pop()], dim=1), temb)
        return z

    def decode(self, hc, h, temb):
        return self.out(self.dec(torch.cat([hc, h], dim=1), temb)).transpose(1, 2)

    def pool(self, h):
        return h.mean(dim=-1)


ARCHS = {"mlp": TrajMLP, "unet": TrajUNet, "transformer": TrajTransformer, "maze_unet": MazeUNet}


def load_traj_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = ARCHS[ckpt["cfg"]["arch"]](**ckpt["cfg"]["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["cfg"]

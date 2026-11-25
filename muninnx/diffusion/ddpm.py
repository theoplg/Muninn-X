"""VP diffusion (DDPM / DDIM) for the E14 check: is the Muninn certificate gap generic?

Index convention follows muninnx.solvers.base: forward step k = 0..T-1 maps tau_k -> tau_{k+1} and
corresponds to the paper's diffusion step t = T - k (t: T -> 1, noise -> data). The network is an
epsilon-predictor eps_theta(tau, s) with time input s = t / T in (0, 1].

The step maps are the paper's (Appendix H, Propositions 2-3):
  DDPM: Phi_t(tau, eps, xi) = (tau - beta_t / sqrt(1 - abar_t) * eps) / sqrt(alpha_t) + sigma_t xi
        K_t = 1 / sqrt(alpha_t),   L'_t = beta_t / (sqrt(alpha_t) sqrt(1 - abar_t))
  DDIM: Phi_t(tau, eps) = sqrt(abar_{t-1}) x0_hat + sqrt(1 - abar_{t-1}) eps,
        x0_hat = (tau - sqrt(1 - abar_t) eps) / sqrt(abar_t)
        K_t = sqrt(abar_{t-1} / abar_t),   L'_t = |sqrt(1 - abar_{t-1}) - sqrt(abar_{t-1}) sqrt(1 - abar_t) / sqrt(abar_t)|
Paired chains must share the sampler noise xi: it is a deterministic function of (noise_seed, k).
"""
import math

import torch

from muninnx.solvers.base import Solver


def cosine_alphabar(T, s=0.008, max_beta=0.999):
    """abar_0..abar_T (abar_0 = 1) for the cosine schedule (Nichol & Dhariwal), betas clipped."""
    f = lambda u: math.cos((u / T + s) / (1 + s) * math.pi / 2) ** 2
    abar = [1.0]
    for t in range(1, T + 1):
        beta = min(1 - f(t) / f(t - 1), max_beta)
        abar.append(abar[-1] * (1 - beta))
    return torch.tensor(abar, dtype=torch.float64)


def ddpm_loss(model, x0, abar, generator=None):
    """Epsilon-prediction loss with t ~ U{1..T}; network time input t / T."""
    T = len(abar) - 1
    B = x0.shape[0]
    t = torch.randint(1, T + 1, (B,), device=x0.device, generator=generator)
    eps = torch.randn(x0.shape, device=x0.device, generator=generator)
    ab = abar.to(x0.device, torch.float32)[t][:, None]
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    return torch.nn.functional.mse_loss(model(x_t, t.float() / T), eps)


class DDPMSolver(Solver):
    def __init__(self, T, mode="ddpm", noise_seed=0, abar=None):
        super().__init__(T)
        assert mode in ("ddpm", "ddim")
        self.T, self.mode, self.noise_seed = T, mode, noise_seed
        self.abar = cosine_alphabar(T) if abar is None else abar
        self.alpha = torch.ones(T + 1, dtype=torch.float64)
        self.alpha[1:] = self.abar[1:] / self.abar[:-1]
        self.beta = 1 - self.alpha

    def _t(self, k):
        return self.T - k

    def t(self, k):
        return self._t(k) / self.T

    def dt(self, k):
        return 1.0 / self.T

    def _coefs(self, k):
        t = self._t(k)
        return (float(self.alpha[t]), float(self.beta[t]), float(self.abar[t]), float(self.abar[t - 1]))

    def noise(self, x, k):
        g = torch.Generator(device=x.device)
        g.manual_seed(self.noise_seed * 1_000_003 + k)
        return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)

    def step(self, x, eps, k):
        a, b, ab, ab_prev = self._coefs(k)
        if self.mode == "ddpm":
            mean = (x - b / math.sqrt(1 - ab) * eps) / math.sqrt(a)
            sigma = math.sqrt(b * (1 - ab_prev) / (1 - ab))
            return mean + sigma * self.noise(x, k) if sigma > 0 else mean
        x0_hat = (x - math.sqrt(1 - ab) * eps) / math.sqrt(ab)
        return math.sqrt(ab_prev) * x0_hat + math.sqrt(1 - ab_prev) * eps

    def coeffs(self, k):
        a, b, ab, ab_prev = self._coefs(k)
        if self.mode == "ddpm":
            return 1 / math.sqrt(a), b / (math.sqrt(a) * math.sqrt(1 - ab))
        return math.sqrt(ab_prev / ab), abs(math.sqrt(1 - ab_prev) - math.sqrt(ab_prev) * math.sqrt(1 - ab) / math.sqrt(ab))

    def eps_coef(self, k):
        """Signed coefficient c of eps in the (affine) update: Phi = K tau + c eps (+ noise)."""
        a, b, ab, ab_prev = self._coefs(k)
        if self.mode == "ddpm":
            return -b / (math.sqrt(a) * math.sqrt(1 - ab))
        return math.sqrt(1 - ab_prev) - math.sqrt(ab_prev) * math.sqrt(1 - ab) / math.sqrt(ab)

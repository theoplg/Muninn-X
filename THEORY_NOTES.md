# Muninn-X — theory notes

Why the Muninn certificate fails for flow matching, what the correct error propagation is,
and what it predicts. Every claim below is checked numerically in **E7**
([REPORT_NOTES.md](REPORT_NOTES.md), `scripts/theory_check.py`).

Notation follows [GUIDE.md](GUIDE.md): time runs noise → data ($t = 0 \to 1$), uniform Euler grid
$t_k = k h$ with $h = 1/N$, velocity field $v(x, t) = v_\theta(x, t)$.

---

## 1. Setup: two chains from the same noise

- **Full chain** (reference): $x_{k+1} = x_k + h\, v(x_k, t_k)$.
- **Cached chain** (Muninn, oracle, any caching scheme): $\tilde x_{k+1} = \tilde x_k + h\, u_k$, where
  $u_k$ is either a fresh evaluation $v(\tilde x_k, t_k)$ or a cached / extrapolated velocity.

Both start from $x_0 = \tilde x_0$. Define:
- the **deviation** $\Delta_k = x_k - \tilde x_k$ (what the certificate must control: $d = \lVert \Delta_N \rVert$),
- the **injected error** $e_k = v(\tilde x_k, t_k) - u_k$: the error made *at* step $k$ by not
  evaluating the network at the current state. It is $0$ at recompute steps. This is the quantity
  Muninn calibrates (its $\varepsilon$).

## 2. The exact recursion

Subtracting the two updates:

$$
\Delta_{k+1} = \Delta_k + h\,\big[v(x_k) - u_k\big]
= \Delta_k + h\,\underbrace{\big[v(x_k) - v(\tilde x_k)\big]}_{\text{drift propagation}} + h\, e_k .
$$

The drift term exists because the cached chain has already drifted, so even a *fresh* evaluation
at $\tilde x_k$ differs from the full chain's evaluation at $x_k$. By the mean-value theorem,

$$
v(x_k) - v(\tilde x_k) = \bar J_k\, \Delta_k,
\qquad
\bar J_k = \int_0^1 \frac{\partial v}{\partial x}\big(\tilde x_k + s\,\Delta_k,\ t_k\big)\, ds ,
$$

so the recursion is exactly linear in $\Delta$:

$$
\boxed{\ \Delta_{k+1} = \big(I + h \bar J_k\big)\,\Delta_k + h\, e_k\ }
$$

Unrolling from $\Delta_0 = 0$:

$$
\boxed{\ \Delta_N = \sum_{k=0}^{N-1} h\, M_{k+1\to N}\, e_k,
\qquad
M_{k+1\to N} = \big(I + h\bar J_{N-1}\big)\cdots\big(I + h\bar J_{k+1}\big)\ }
$$

(with $M_{N\to N} = I$). This is an identity, not a bound. Taking norms gives the
**corrected bound**

$$
d = \lVert \Delta_N \rVert \;\le\; \sum_{k=0}^{N-1} h\, \lVert M_{k+1\to N} \rVert\, \lVert e_k \rVert ,
$$

i.e. the correct sensitivity weight is $L_k = h\,\lVert M_{k+1\to N}\rVert$. For small deviations
$M_{k+1\to N} \approx G_{k+1\to N} = \partial x_N / \partial x_{k+1}$, the **Jacobian of the
remaining flow** along the full chain (§2.1).

### 2.1 Small-deviation approximation: the flow Jacobian $G$

The **exact** one-step matrix uses the Jacobian *averaged along the segment* between the two chains:

$$
\bar J_k = \int_0^1 \frac{\partial v}{\partial x}\big(\tilde x_k + s\,\Delta_k\big)\,ds .
$$

That segment runs from $\tilde x_k$ (cached chain) to $x_k$ (full chain). **If the deviation
$\Delta_k$ is small, the segment is short**, and the Jacobian barely changes along it. So

$$
\bar J_k \approx J(x_k) := \frac{\partial v}{\partial x}(x_k, t_k),
\qquad
A_k = I + h\bar J_k \approx I + h\,J(x_k).
$$

Now look at one Euler step **of the full chain alone**, $x_{k+1} = x_k + h\,v(x_k)$. Its derivative
with respect to its input is

$$
\frac{\partial x_{k+1}}{\partial x_k} = I + h\,J(x_k),
$$

which is exactly the approximated $A_k$. Chaining steps with the chain rule:

$$
M_{k+1\to N} \approx \big(I + hJ(x_{N-1})\big)\cdots\big(I + hJ(x_{k+1})\big)
= \frac{\partial x_{N}}{\partial x_{N-1}}\cdots\frac{\partial x_{k+2}}{\partial x_{k+1}}
= \frac{\partial x_N}{\partial x_{k+1}} =: G_{k+1\to N}.
$$

**Meaning:** $G_{k+1\to N}$ says how much the **final full-chain sample moves if you nudge the
state just after step $k$**. It depends on the full chain only, not on the cached chain or on the
deviation. That is what makes it usable.

**When it breaks:** the gap is $\bar J_k - J(x_k) = O\big(\lVert\Delta_k\rVert \cdot \partial^2 v/\partial x^2\big)$.
It is small when the deviation is small or $v$ is smooth. It fails when the deviation is large
**and** the Jacobian varies quickly, which is precisely near mode boundaries, where $J$ jumps.
That is why the $G$-based bound held for 100% of samples on 8gaussians but only 86% on
checkerboard, where the deviations are larger and the bound is tight (median $d/B = 0.73$).

## 3. What the paper does, and where it breaks

The paper's Assumption 1 bounds the step map $\Phi_k(x, v)$ separately in its two arguments:

$$
\lVert \Phi_k(x, v) - \Phi_k(x', v') \rVert \le K_k \lVert x - x' \rVert + L'_k \lVert v - v' \rVert .
$$

For Euler, $\Phi_k(x, v) = x + h v$ gives $K_k = 1$ and $L'_k = h$, hence $L_k = h \prod_{j>k} K_j = h$ and

$$
d \;\le\; \sum_{k} h\, \lVert e_k \rVert \qquad \text{(paper)} .
$$

The flaw: **$v$ is treated as a free input, but at every step it is the network evaluated at the
current state**, $v = v(\tilde x_k)$. Treating $\Phi$ as a function of $x$ alone, the true one-step
Lipschitz constant is $\lVert I + h\, \partial v/\partial x \rVert$, not $1$. Equivalently, the paper's
recursion drops the drift term $h \bar J_k \Delta_k$, which is present at *every* step (recompute
steps included). The paper bound is exactly the corrected bound with $M$ replaced by $I$.

- If the flow **expands** ($\lVert M \rVert > 1$), the paper bound is too small → the certificate is invalid.
- If the flow **contracts** ($\lVert M \rVert < 1$), the paper bound is valid but loose.

A worst-case Lipschitz fix, $K_k = 1 + h\,\mathrm{Lip}(v)$, is useless in practice: $\mathrm{Lip}(v)$
blows up like $1/(1-t)^3$ near the data (see §4), and the product $\prod_k \big(1 + h\,\mathrm{Lip}\big)$
is astronomically large.

## 4. Why flow matching expands: the Tweedie Jacobian

For the straight (CondOT) path $x_t = t\, x_1 + (1-t)\, x_0$ with $x_0 \sim \mathcal N(0, I)$:

- $x_1 - x_0 = \dfrac{x_1 - x_t}{1-t}$, so the optimal velocity is

$$
v(x, t) = \frac{m(x, t) - x}{1 - t}, \qquad m(x, t) = \mathbb E[x_1 \mid x_t = x] .
$$

- The likelihood $x_t \mid x_1 \sim \mathcal N\big(t\, x_1,\ (1-t)^2 I\big)$ is Gaussian, so
  (Tweedie / exponential-family identity)

$$
\frac{\partial m}{\partial x} = \frac{t}{(1-t)^2}\, \Sigma(x, t), \qquad \Sigma(x, t) = \mathrm{Cov}[x_1 \mid x_t = x] .
$$

Therefore

$$
\boxed{\ \frac{\partial v}{\partial x} = \frac{1}{1-t}\left[\frac{t}{(1-t)^2}\, \Sigma(x, t) - I\right]\ }
$$

The Jacobian is symmetric, and its eigenvalues are

$$
\lambda_i = \frac{1}{1-t}\left(\frac{t\, \sigma_i^2}{(1-t)^2} - 1\right),
$$

with $\sigma_i^2$ the posterior variances. A perturbation along direction $i$ **grows iff
$\sigma_i > (1-t)/\sqrt t$**, i.e. iff the endpoint is still uncertain in that direction compared
with the current noise level.

Three regimes:

| situation | posterior $\Sigma$ | $\partial v/\partial x$ | effect on errors |
|---|---|---|---|
| sample between modes ("undecided") | large, along the direction separating the modes | large positive eigenvalue | **strong amplification** |
| sample inside a Gaussian mode of std $s$ | $\sigma^2 = \left(1/s^2 + t^2/(1-t)^2\right)^{-1}$ | $\lambda < 0$ **iff** $t < 1/(1+s^2)$ | **contraction** (errors forgotten) until $t = 1/(1+s^2)$, mild growth after |
| locally flat / uniform data (far from edges) | $\sigma^2 \approx (1-t)^2/t^2$ | $\approx I/t$ | mild growth, $G \approx 1/t_{\text{start}}$ |

Consequences:
1. **The amplification is concentrated on the boundaries between basins of attraction**:
   the set of noise points where a tiny push changes which mode the sample ends in. That is
   where the max ~900× of E3 comes from.
2. **It is per-sample**, not per-time: at the same $t$, most samples have $\lVert G \rVert \approx 1$–$2$
   while boundary samples have $\lVert G \rVert \gg 1$. A per-step weight $L_k$ cannot capture it.
3. **The data geometry decides the sign.** Narrow modes (8gaussians, $s = 0.5$) give contraction
   late in sampling; flat data (checkerboard squares) give mild growth everywhere. That is why the
   paper bound failed far more on checkerboard (77% of samples) than on 8gaussians (22%).

## 5. What this implies for a valid certificate

The correct weights $L_k = h\,\lVert M_{k+1\to N}\rVert$ cannot be used directly by a controller:
- they depend on the **future** of the trajectory ($M$ runs from $k+1$ to $N$),
- they depend on the **deviation itself** (mean-value Jacobian between the two chains),
- they are **heavy-tailed across samples**.

Options:
- **(a) Probabilistic weights.** Replace $L_k$ by a conformal upper bound on $h\,\lVert G_{k+1\to N}\rVert$,
  possibly conditioned on probe features (the probe sees whether the sample is near a boundary).
  Valid with high probability, but the tail makes it conservative.
- **(b) End-to-end certification.** Do not chain per-step bounds at all: tune the controller's
  knob so that the *final* $d$ satisfies $P(d > \eta) \le \alpha$ on calibration chains (LTT /
  conformal, as in certified FewSteps, E4). Valid for any caching policy, with no Lipschitz assumption.

And one implication for **usefulness**: an error injected where $\lVert G \rVert \approx 1$ (or $< 1$) is
cheap, while the same error near a basin boundary is expensive. A policy that reuses freely away
from boundaries and recomputes near them is a genuinely *per-sample* allocation, which no uniform
grid can mimic. Whether it pays off is an empirical question (oracle test with the true $G$ first).

## 6. Status of each claim (E7)

| claim | check | result |
|---|---|---|
| Exact recursion $\Delta_N = \sum_k h\, M_{k+1\to N}\, e_k$ | bound with mean-value Jacobians | holds for 99.9–100% of samples (residual = 8-point quadrature) |
| Paper bound invalid | $d \le \sum_k h \lVert e_k \rVert$ | fails for 77–78% (checkerboard), 10–22% (8gaussians) |
| $G$ along the full chain is a good proxy | $d \le \sum_k h \lVert G_{k+1\to N}\rVert \lVert e_k \rVert$ | 86% (checkerboard), 100% (8gaussians); linearisation error matters on checkerboard |
| Tweedie formula | network $\partial v/\partial x$ vs Monte-Carlo posterior covariance | $\lambda_{\max}$ correlation 0.98–1.00 for $t \in [0.3, 0.7]$, median rel. error 3–17% |
| Contraction inside narrow modes, sign flip at $t = 1/(1+s^2)$ | 8gaussians ($s = 0.5$ → flip at $t = 0.8$) | $t = 0.7$: predicted $\lambda = -0.59$, measured median $-0.51$; $t = 0.85$: predicted $+0.31$, measured $+0.34$ |
| Amplification on basin boundaries | map of $\lVert G \rVert$ at $t = 0.2$ | high-$\lVert G \rVert$ samples lie exactly on the 8 rays / the checkerboard cell boundaries |

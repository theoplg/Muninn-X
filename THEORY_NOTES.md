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

Page references are to `Muninn.pdf` (PDF page index) and `muninn_appendices.pdf`; maths is
transcribed from the PDF. The paper uses diffusion notation: time runs backwards ($t: T \to 1$),
$\varepsilon_\theta$ is the network, $\tilde\varepsilon_t$ the value the cached chain actually uses.

**The definitions are fine** (p. 3, left column, "Error variables"): "the trajectory drift
$\Delta_t := \tau_t^{\text{full}} - \tilde\tau_t$ and the denoiser mismatch
$e_t := \varepsilon_\theta(\tilde\tau_t, t, c) - \tilde\varepsilon_t$". Note that $e_t$ is evaluated
**at the cached state**, so $e_t = 0$ at recompute steps by definition. It is exactly our injected
error $e_k$ of §1.

**Assumption 1 is fine** (p. 3): for fixed sampler noise,

$$
\lVert \Phi_t(\tau, \varepsilon, \xi_t) - \Phi_t(\tau', \varepsilon', \xi_t) \rVert \le K_t \lVert \tau - \tau' \rVert + L'_t \lVert \varepsilon - \varepsilon' \rVert ,
$$

with $(K_t, L'_t)$ read off the update coefficients with $\varepsilon$ held fixed ("$\Phi_t$ is affine in
$\tau_t$ and $\varepsilon_t$, so $K_t$ and $L'_t$ follow directly from the update coefficients").
For Euler, $\Phi_k(x, v) = x + h v$ gives $K_k = 1$ and $L'_k = h$.

**The flaw is the last equality of Eq. (1)** (p. 3, right column):

$$
\lVert\Delta_{t-1}\rVert \le K_t \lVert \tau_t^{\text{full}} - \tilde\tau_t\rVert + L'_t \big\lVert \varepsilon_\theta(\tau_t^{\text{full}}, t, c) - \tilde\varepsilon_t \big\rVert
\overset{\text{(!)}}{=} K_t\lVert\Delta_t\rVert + L'_t\lVert e_t\rVert .
$$

The inequality is correct. The equality is not: the left term evaluates the network **at the full
state**, $e_t$ **at the cached state**, and

$$
\varepsilon_\theta(\tau_t^{\text{full}}) - \tilde\varepsilon_t
= \underbrace{\big[\varepsilon_\theta(\tau_t^{\text{full}}) - \varepsilon_\theta(\tilde\tau_t)\big]}_{\text{drift term, dropped}} + e_t ,
$$

so the two coincide only when $\Delta_t = 0$. The dropped bracket is the drift propagation of §2
($h \bar J_k \Delta_k$ in our notation). It is nonzero at **every** step once the chains have
separated, recompute steps included; only because it was dropped does "$e_t = 0$ at recompute
steps" look like "recompute steps add no error".

**Everything after inherits it.** The unrolling (p. 3, "a simple induction shows",
Eq. (2)), the deviation bound $d \le \sum_t \Gamma L_t \lVert e_t\rVert$ (p. 4, Eqs. (4)–(5)), the
global risk argument restricting the sum to reuse steps,
"$d(\tau_0^{\text{full}}, \tilde\tau_0) \le \sum_{t\in\mathcal R} c_t(e_t)$ (from (5))" (p. 6), and
the appendix proof, which takes Eq. (1) as a hypothesis: "Suppose Assumption 1 […] holds and
yields the local recursion […] (94) which is Eq. (1) in the main paper" (appendix p. 36).

**A second instance of the same swap** (p. 6 and appendix p. 28). The risk argument states that
"whenever Muninn reuses at step $t$, the realized denoiser error magnitude $\lVert e_t\rVert$
coincides with the latent potential reuse error $\epsilon_t$". But the calibration label
(Algorithm 2, line 18) is $\epsilon_t = \lVert \hat\varepsilon_t^{\text{full}} - \tilde\varepsilon_t^{\text{cand}}\rVert / \sqrt{Hd}$:
network at the full state, minus a cached value that is itself a full-chain output (lines 22–23).
At deployment, $e_t$ is the network at the cached state minus a value computed earlier on the
drifted chain. They coincide only without drift.

In our notation (Euler, forward time) the paper's bound is

$$
d \;\le\; \sum_{k} h\, \lVert e_k \rVert \qquad \text{(paper)} ,
$$

i.e. exactly the corrected bound of §2 with $M_{k+1\to N}$ replaced by $I$.
E7 tests this version ($e_k$ at the cached state, as in their Eq. (5)); E14 tests the
calibration-label version (network at the full state). Both fail.

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

## 7. Why the paper did not see it

The gap of §3 does not depend on flow matching: it is the last equality of their Eq. (1), which
replaces $\varepsilon_\theta(\tau^{\text{full}}_t)$ by $\varepsilon_\theta(\tilde\tau_t)$ and so drops the
drift term $\varepsilon_\theta(\tau^{\text{full}}_t) - \varepsilon_\theta(\tilde\tau_t)$, for any sampler.
Their calibration does not catch it either: in Algorithm 2 (appendix p. 28) the cached value is
itself a *full-chain* output, so the labels
$\lVert \hat\varepsilon^{\text{full}}_t - \hat\varepsilon^{\text{full}}_{\text{anchor}} \rVert$ contain no drift.

So why does their method look valid? Three reasons, from strongest evidence to weakest.

**(a) Their validation cannot detect violations of a few percentage points** (read from the
paper). Every reported $\hat p_{\text{viol}}$ (0.028–0.049 in Table I) comes from **150 runs**
with a Wilson interval (Appendix C). For 7 violations out of 150 the 95% interval is about
$[0.023, 0.093]$. The probability that a 150-run estimate lands at or below $0.05$ is

| true violation rate | 0.050 | 0.059 (our checkerboard, $\eta = 0.3$) | 0.083 (our checkerboard, $\eta = 0.5$) |
|---|---|---|---|
| $P(\hat p_{150} \le 0.05)$ | 0.52 | 0.34 | 0.06 |

so a method violating its guarantee like ours did on checkerboard at $\eta = 0.3$ would still look
valid in a third of 150-run evaluations. The per-sample claim $d \le \hat D = \sum_{t\in\mathcal R}\hat c_t$
is only shown as a binned reliability plot (their Fig. 10, "conservative, below the diagonal").
A binned average can sit below the diagonal while many samples violate it: on our 8gaussians the
median $d/B$ was $0.79$ while the per-sample bound failed for 22% of samples (E7).

**(b) Their DDPM weights over-count propagation for most samples, and under-count it only on
basin boundaries** (derivation below; measured in E14). Their $K_t$ is computed with $\varepsilon$ held fixed, so
$K_t = 1/\sqrt{\alpha_t} > 1$ and

$$
L_t = L'_t \prod_{j<t} K_j = \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}} \cdot \frac{1}{\sqrt{\bar\alpha_{t-1}}}
= \frac{\beta_t}{\sqrt{\bar\alpha_t}\,\sqrt{1-\bar\alpha_t}} ,
$$

which is very large for early steps ($\bar\alpha_t \to 0$), i.e. in the noisy phase where samples
choose their mode. The step's *true* Jacobian also counts $\partial\varepsilon_\theta/\partial\tau$.
With Tweedie for the VP path, $\varepsilon_\theta(\tau) = \big(\tau - \sqrt{\bar\alpha_t}\, m(\tau)\big)/\sqrt{1-\bar\alpha_t}$
with $\partial m/\partial\tau = \frac{\sqrt{\bar\alpha_t}}{1-\bar\alpha_t}\Sigma$, so the DDPM mean update has Jacobian

$$
\frac{\partial \tau_{t-1}}{\partial \tau_t}
= \frac{1}{\sqrt{\alpha_t}}\left[\Big(1 - \frac{\beta_t}{1-\bar\alpha_t}\Big) I + \frac{\beta_t\,\bar\alpha_t}{(1-\bar\alpha_t)^2}\,\Sigma\right],
$$

which for a settled posterior ($\Sigma \approx 0$) equals $\sqrt{\alpha_t}\,\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t} < 1$:
a contraction, while the paper charges $1/\sqrt{\alpha_t} > 1$. The same expression exceeds
$1/\sqrt{\alpha_t}$ only along directions with $\sigma^2 > (1-\bar\alpha_t)/\bar\alpha_t$, the
VP analogue of the condition in §4. So their weights carry a large, systematic slack that can
absorb the missing drift term unless a sample sits on a basin boundary. With Euler for FM,
$K = 1$ and $L_k = h$ are flat in time, so there is no such slack, and the gap shows directly.

**(c) Conservative per-step bounds and less multimodal tasks** (hypothesis). Per-step $U_t$ at
level $\alpha/|T_{\text{cache}}|$ (≈ 99.9th percentile) plus the triangle inequality add more
slack (this alone hid the gap on our 8gaussians). Their planners are conditioned on start and
goal (inpainting), which removes much of the multimodality, so fewer samples sit on a boundary
and the tail of $\lVert G\rVert$ is smaller than on checkerboard, which is close to a worst case
(flat data, expanding everywhere, many boundaries).

**Separately, usefulness.** Their FewSteps baseline is weak: Diffuser-style DDPM models trained for a
fixed $T$-step stochastic schedule degrade with fewer steps (their Table V: $T' = 90$ already drops
the task score from 68.7 to 56.1 with $\hat p_{\text{viol}} = 0.17$). With FM the paths are nearly
straight and FewSteps (especially AB2) is strong. This explains the speed-up gap between their
results and ours independently of validity.

**What E14 found** (VP $\varepsilon$-model on checkerboard, their exact DDPM / DDIM updates and
$(K_t, L'_t)$, cosine schedule, $T = 100$; see REPORT_NOTES E14). Both explanations are right:

- **The gap is generic.** With an oracle that knows the true errors, the paper's per-sample
  certificate $d \le \sum_{t\in\mathcal R} L_t \lVert \varepsilon_\theta(\tau^{\text{full}}_t) - \tilde\varepsilon_t\rVert$
  fails for **1.3–4.4%** of samples with DDPM and **1.9–6.9%** with DDIM (and 35–79% with FM +
  Euler). The same sum taken over *all* steps, which their Eq. (1) does justify, holds for 100%.
- **Their weights carry a large slack, as predicted in (b).** The ratio true flow Jacobian
  $/ \prod K$ has median **0.04** (DDPM) / **0.19** (DDIM) for an error made at step 5, rising
  to ~1 near the data: for a typical early error the paper over-counts propagation 5–25×. The p99
  at steps 20–60 is still ≈ 5–8 and the maximum reaches $10^2$–$10^3$: on basin boundaries the paper *under*-counts,
  and those are the samples that break the certificate. With FM + Euler the median ratio is
  already 1.1–2.4 (no slack anywhere).
- **Real Muninn is valid on DDPM / DDIM, but because it barely caches.** With the same
  calibration as on FM, test $\hat p_{\text{viol}} \le 0.004$ on 50k samples for every $\eta$,
  but it uses 98–100 NFE out of 100: the large early weights $L_t$ make almost every reuse too
  expensive. On FM + Euler the same code reuses much more and violates $\alpha$ ($0.059$ and
  $0.078$ at $\eta = 0.3, 0.5$).

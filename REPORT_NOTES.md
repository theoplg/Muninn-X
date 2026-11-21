# Muninn-X — experiment notes

Running log of experiments: setup, results and interpretation. `runs/` holds the raw outputs
(tracked in git); **this file is the record of what they mean**. Every entry lists the command that reproduces it.

- Newest findings go in **Key findings** (top). Each experiment gets an entry under **Log**.
- Planned experiments live in **Next experiments** until they are run, then move to the log.
- Conventions follow [GUIDE.md](GUIDE.md): time noise→data (t=0→1), Euler, d = ‖·‖ (Γ = 1 for 2D toys).

---

## Key findings (as of 2026-11-21)

1. **Muninn on velocity + Euler does not beat FewSteps.** At matched mean NFE, uniform Euler
   with fewer steps has a 5–15× smaller E[d] on both toys (E0, E1, E2).
2. **Even a perfect predictor of the reuse error does not help** (E2, oracle). With the true
   ε the error is still ~10× worse than FewSteps. The score and the conformal bound are *not*
   the bottleneck: reusing v with Euler is just Euler with a non-uniform step schedule, and
   that schedule is worse than the uniform one.
3. **The certificate d ≤ Σ ĉ is invalid as derived.** It assumes K_k = 1 for Euler, but at
   recompute steps v is evaluated at the drifted state, and the flow amplifies drift (E3: median
   1.4–1.7×, p99 13–33×, max ~900× for early steps). The oracle, which knows ε exactly, violates
   d ≤ Σ ĉ in 22% (8gaussians) and 77% (checkerboard) of samples.
4. **Muninn's guarantee is broken on checkerboard.** p_viol = 0.059 (CI 0.057–0.061) at η = 0.3
   and 0.083 (0.080–0.085) at η = 0.5, both > α = 0.05. It only held elsewhere because U(s) is
   very conservative (pooled linear fit + α_step = α/90 tail quantile).
5. **Staleness-aware design (Finding 1 in GUIDE) reproduces.** Our E0 numbers on 8gaussians match
   the GUIDE's staleness-aware row.
6. **Certified FewSteps is valid and much cheaper** (E4). Choosing a uniform step count N' by
   conformal / LTT calibration on the final d gives p_viol ≤ α everywhere, with 1.1–17× fewer
   NFE than Muninn at the same (η, α). It is the baseline any adaptive method must beat.
7. **Reusing x̂₁ instead of v changes nothing** (E5). With Euler on straight FM paths, reusing
   any fixed prediction (v, x̂₁, x̂₀) for l steps is exactly one Euler step of size l·Δ
   (verified to 4e-14). The GUIDE's "x̂₁ reuse contracts errors" ablation is a no-op.
8. **Extrapolating beats reusing by ~10×, but uniform AB2 still wins** (E6). An oracle that linearly
   extrapolates from the last two computed velocities reaches, at 18 NFE, a sample error lower
   than full Euler-100 (vs the exact ODE). But uniform AB2 FewSteps gets the same gain with no
   adaptivity: 3–8× lower error than the extrapolation oracle at matched NFE. The gain comes from
   **higher-order integration, not from per-sample adaptivity**.
9. **The certificate target matters.** Euler-100 itself is 0.039 (checkerboard) / 0.023
   (8gaussians) away from the exact ODE solution. AB2 with 15–20 NFE is closer to the exact
   solution than Euler-100 is. Certifying closeness to Euler-100 penalises better solvers.

10. **The cause of the invalid certificate is identified exactly** (E7, [THEORY_NOTES.md](THEORY_NOTES.md)).
   The paper's accounting drops the drift term: the exact recursion is
   $\Delta_{k+1} = (I + h\bar J_k)\Delta_k + h e_k$, so the correct weight is
   $L_k = h\lVert M_{k+1\to N}\rVert$ (flow Jacobian), not $h$. With it, the bound holds for
   99.9–100% of samples (paper: 22–90%). For FM,
   $\partial v/\partial x = \frac{1}{1-t}\big[\frac{t}{(1-t)^2}\mathrm{Cov}(x_1\mid x_t) - I\big]$ (Tweedie),
   which the trained network follows (corr ≥ 0.98): errors are amplified where the endpoint is
   still uncertain, i.e. on the **boundaries between modes**, and contracted inside narrow modes.
11. **End-to-end certification makes Muninn valid, but certified AB2 wins by 6–14×** (E8).
   Choosing the controller's budget $\lambda$ by fixed-sequence LTT on the final $d$ gives
   $p_{\text{viol}} \le \alpha$ on every test row (146/146 with LTT). But at equal certified
   $(\eta, \alpha)$, Muninn needs 64–97 NFE where certified AB2 needs 4–14 and certified Euler
   7–150. Against the exact ODE, Muninn cannot certify $\eta \le 0.05$ (checkerboard) at all,
   since it inherits Euler-100's own error.
12. **Per-sample adaptivity has a real but small ceiling: ~1.1–1.35× over certified AB2** (E9).
   The best oracle (extrapolate on an AB2 base, decide by the true $G$-weighted cost
   $h\lVert G_{k+1\to N} e_k\rVert$, threshold rule) beats certified AB2 only at tight tolerances
   ($\eta \le 0.05$), on both smooth models, with *free* knowledge of $e_k$ and $G$. Being
   $G$-aware matters (1.1–1.5× over the plain cost); a greedy budget rule is far worse than a
   threshold. A practical method must pay for a probe (≈ 8 NFE here) and use conservative
   predictors, which is about the size of the gain: on these toys it would at best tie.
13. **Model smoothness in $t$ confounds solver comparisons** (E9). The original 8gaussians model
   (batch 1024) has a velocity that oscillates in $t$ faster than the grid step (fine-scale
   $\lvert\partial_t v\rvert$ is 1.74× its one-step average; sinusoidal embedding with frequencies
   up to $1000\,t$). Higher-order FewSteps then converges erratically and adaptive oracles look
   4–9× better than they are. Retrained with batch 16384 (`runs/toy_8gaussians_b16k`, ratio 1.20)
   the effect disappears.

**Implication.** (a) *Validity*: the sensitivity weights must account for the flow's
amplification, or calibration must target the final d directly (CertFew already does).
(b) *Usefulness*: on these toys, per-sample adaptivity has shown no value at all. Every gain so far
came from the integration scheme, and a uniform grid captures it. (c) *Target*: the reference
for d should be a near-exact ODE solution, not Euler-100.

---

## Setup

| item | value |
|---|---|
| env | `~/.venvs/muninnx` (uv, Python 3.11, torch 2.11 cu128), RTX 5070 |
| model | `VelocityMLP` dim 2, hidden 256, depth 4, sinusoidal t-embedding (64) |
| training | CondOT FM loss, Adam lr 1e-3 cosine. See per-model rows below |
| solver | Euler, uniform grid, N = 100 (reference chain) |
| probe | `stem(embed_input(x, t))` ≈ ¼ of a forward pass for this MLP |
| score | relative L1, ω = 1e-3, against features at the **last recompute** |
| bound | linear m(s) by OLS + split-conformal q, U = max(m+q, 0), α_step = α / \|T_cache\| |
| controller | forbid prefix = suffix = 5 (T_cache = steps 5..94, 90 steps), max_lag = 10 |
| metric | d = ‖x_N^full − x_N‖ per sample, paired on the same noise; p_viol = P(d > η) with Wilson 95% CI |

**Cost accounting caveat.** All NFE numbers count network forward passes only. Muninn also runs
the probe at every step (~0.25 forward for this MLP), so its real cost is ≈ NFE + 0.25·N.
None of the conclusions below change if the probe is counted; it only makes Muninn look worse.

### Models

| run dir | dataset | iters | batch | final loss | notes |
|---|---|---|---|---|---|
| `runs/toy_8gaussians` | 8gaussians | 20k | 1024 | — | trained by Mamane |
| `runs/toy_checkerboard_staircase` | buggy "checkerboard" (overlapping strips) | 30k | 16384 | ~2.83 | **superseded**, see E1 |
| `runs/toy_checkerboard` | checkerboard (fixed, 8 squares on [-4,4]²) | 30k | 16384 | ~3.35 | GPU data gen, TF32 |

```bash
python scripts/train_toy.py --dataset checkerboard --iters 30000 --batch 16384
python scripts/sample_toy.py --ckpt runs/toy_checkerboard/model.pt --n-steps 100 10 3
```
Sample quality: N=100 reproduces the board. N=10 is visibly blurred on checkerboard (squares
merge), unlike 8gaussians where N=10 already looks clean. N=3 fails on both.

---

## Log

### E0 — Muninn vs FewSteps on 8gaussians (2026-11-18)
```bash
python scripts/run_muninn_toy.py --ckpt runs/toy_8gaussians/model.pt   # n_cal 500, n_test 2000
```
| α | η | Muninn NFE | Muninn E[d] | p_viol | bound_holds | FewSteps E[d] |
|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 97.8 | 0.0035 | 0.001 | 0.999 | 0.0031 |
| 0.05 | 0.15 | 92.8 | 0.0295 | 0.0025 | 0.998 | 0.0036 |
| 0.05 | 0.30 | 85.4 | 0.0679 | 0 | 1.000 | 0.0068 |
| 0.10 | 0.05 | 97.0 | 0.0044 | 0.001 | 0.999 | 0.0018 |
| 0.10 | 0.15 | 92.2 | 0.0324 | 0.0025 | 0.998 | 0.0036 |
| 0.10 | 0.30 | 84.4 | 0.0737 | 0 | 1.000 | 0.0066 |

- **Validity:** OK. p_viol ≪ α.
- **Usefulness:** FewSteps 8–10× better at matched NFE (tie only at η = 0.05). Max speedup 1.18×.
- **Conservatism:** E[d] ≈ η/4 at η = 0.3. Most of the budget is never used.
- `error_vs_lag.png`: ε grows ~linearly with staleness (0.07 at lag 1 → 0.7 at lag 10).
- `calibration_vs_step.png`: ε is largest early (k ≈ 10–30) and decays ~4×. s peaks later
  (k ≈ 30) and only decays ~2×, so the score tracks ε poorly late in the chain.

### E1 — same on the "checkerboard" target (2026-11-18) — **superseded**
```bash
python scripts/run_muninn_toy.py --ckpt runs/toy_checkerboard_staircase/model.pt --n-cal 4000 --n-test 50000
```
The scaffold's `checkerboard()` was buggy (overlapping vertical strips, not a board), so these
numbers are on a staircase-shaped target. Kept for the record only. FewSteps ~15× better,
p_viol 0.003–0.029 (< α), bound_holds dropping to 0.97 as η grows. That was the first hint of the
drift problem confirmed in E2/E3. Fixed in `muninnx/data/toy2d.py` (verified: 8 cells, equal mass).

### E2 — Oracle experiment (2026-11-18)
```bash
python scripts/oracle_toy.py --ckpt runs/toy_checkerboard/model.pt   # n_cal 4000, n_test 50000, α 0.05
python scripts/oracle_toy.py --ckpt runs/toy_8gaussians/model.pt
```
**Question.** Is the gap to FewSteps caused by a bad error predictor (score + U), or by
velocity reuse itself?

**Oracle** (`muninnx/cache/oracle.py`): Muninn's rule with the *true* ε = ‖v(x_k) − v_cache‖
on the chain being run (network evaluated every step for free, only "paid" recomputes counted).
Two variants: same max_lag = 10, and no lag cap. Greedy budget spending, same as Muninn.

Plots: `runs/toy_*/oracle/pareto.png` (E[d] vs mean NFE, full FewSteps curve N' = 2..100).

**Checkerboard** (FewSteps column = uniform Euler with N' = round(NFE))

| η | method | NFE | E[d] | p_viol | bound_holds | FewSteps E[d] |
|---|---|---|---|---|---|---|
| 0.05 | Muninn | 97.0 | 0.0085 | 0.0083 | 0.990 | 0.0013 |
| 0.05 | Oracle | 81.6 | 0.1091 | 0.785 | 0.215 | 0.0084 |
| 0.1 | Muninn | 94.0 | 0.0310 | 0.0200 | 0.978 | 0.0028 |
| 0.1 | Oracle | 69.2 | 0.2054 | 0.779 | 0.211 | 0.0177 |
| 0.15 | Muninn | 91.0 | 0.0691 | 0.0413 | 0.956 | 0.0048 |
| 0.15 | Oracle | 55.4 | 0.2746 | 0.747 | 0.212 | 0.0321 |
| 0.3 | Muninn | 81.9 | 0.1515 | **0.0593** | 0.937 | 0.0084 |
| 0.3 | Oracle | 22.1 | 0.3550 | 0.478 | 0.238 | 0.1358 |
| 0.5 | Muninn | 70.2 | 0.2682 | **0.0826** | 0.916 | 0.0167 |
| 1.0 | Muninn | 41.2 | 0.3544 | 0.0375 | 0.962 | 0.0554 |
| 2.0 | Muninn / Oracle | 18.0 | 0.3576 | 0.0058 | — | 0.1734 |

**8gaussians**

| η | method | NFE | E[d] | p_viol | bound_holds | FewSteps E[d] |
|---|---|---|---|---|---|---|
| 0.05 | Muninn | 97.7 | 0.0036 | 0.0003 | 0.999 | 0.0029 |
| 0.05 | Oracle | 86.5 | 0.0460 | 0.220 | 0.779 | 0.0080 |
| 0.15 | Muninn | 92.7 | 0.0298 | 0.0013 | 0.999 | 0.0034 |
| 0.15 | Oracle | 73.9 | 0.1282 | 0.172 | 0.828 | 0.0105 |
| 0.3 | Muninn | 85.2 | 0.0687 | 0.0001 | 1.000 | 0.0066 |
| 0.3 | Oracle | 29.4 | 0.2362 | 0.098 | 0.900 | 0.0609 |
| 1.0 | Muninn | 51.5 | 0.2203 | 0 | 1.000 | 0.0212 |
| 2.0 | Muninn / Oracle | 18.0 | 0.2243 | 0 | — | 0.1090 |

Full tables (incl. the no-lag-cap oracle and η = 0.01, 0.02, 0.5) are in `results.json`.

**Interpretation**
- On the Pareto plot, Muninn and the oracle lie close together, and both sit roughly an order of
  magnitude above the FewSteps curve at every NFE. A perfect ε predictor buys ~1.3× over Muninn;
  FewSteps is ~10× better. **The loss comes from velocity reuse itself, not from the predictor.**
- Why: with Euler, reusing v for l steps ≡ one Euler step of size l·Δ. Muninn therefore produces a
  non-uniform step schedule (bursts of up to max_lag reuses, then small steps). For a smooth field,
  local error ∝ h², so at fixed NFE a uniform grid minimises the total error.
- The **no-lag-cap oracle is worse** than the capped one at the same NFE. Long reuse runs are the
  most harmful, consistent with the h² argument.
- The NFE floor of 18 = 10 forbidden steps + one recompute every 11 cacheable steps (max_lag 10).
- The oracle's d ≤ Σ ĉ fails massively even though ε is exact, which led to E3.
- Muninn's p_viol exceeds α on checkerboard at η ∈ {0.3, 0.5}, with tight CIs (n = 50k), so this
  is a real violation, not noise.
- Caveat: the oracle is *greedy* (spend budget as soon as affordable), not an optimal allocation.
  A gap of ~10× is too large for a smarter allocation to close, but it is not a formal lower bound.

### E3 — Flow sensitivity / is K_k = 1? (2026-11-18)
```bash
python scripts/flow_sensitivity.py --ckpt runs/toy_checkerboard/model.pt   # n 20000, δ 1e-3
python scripts/flow_sensitivity.py --ckpt runs/toy_8gaussians/model.pt
```
Perturb x_{k+1} by ‖δ‖ = 1e-3 in a random direction, integrate to the end with full Euler, and
measure G_k = ‖Δx_N‖ / ‖δ‖. The Euler bound assumes G_k ≡ 1.

| k | checkerboard median / p90 / p99 / max | 8gaussians median / p90 / p99 / max |
|---|---|---|
| 5 | 1.71 / 3.70 / 16.0 / 938 | 1.47 / 5.78 / 33.2 / 458 |
| 20 | 1.75 / 3.79 / 17.1 / 655 | 1.39 / 5.42 / 31.6 / 214 |
| 40 | 1.43 / 2.85 / 12.7 / 921 | 1.01 / 3.02 / 17.3 / 183 |
| 60 | 1.31 / 1.63 / 6.09 / 95 | 1.01 / 1.30 / 5.55 / 9.6 |
| 80 | 1.20 / 1.24 / 1.82 / 10 | 1.11 / 1.12 / 1.48 / 2.1 |
| 90 | 1.10 / 1.11 / 1.14 / 3.2 | 1.08 / 1.08 / 1.13 / 1.3 |
| 94 | 1.05 / 1.06 / 1.07 / 1.7 | 1.05 / 1.05 / 1.06 / 1.1 |

**Interpretation**
- The true per-step map is x ↦ x + Δ·v(x), so K_k = ‖I + Δ·∂v/∂x‖, not 1. The paper's Assumption 1
  treats v as an input independent of x. That holds at the reuse step itself, but not at the
  recompute steps that follow it.
- The amplification is **heavy-tailed and early**: samples near the boundary between modes
  (which square / which Gaussian they end up in) amplify perturbations by 100–900×. Late
  steps (k ≥ 80) are close to the assumed value of 1.
- This is the GUIDE's "open issue 1", and it is the dominant effect, not a second-order correction.
- It explains both the oracle's failure (E2) and why Muninn's validity degrades on the target
  with sharper mode boundaries (checkerboard).

### E4 — Certified FewSteps (2026-11-19)
```bash
python scripts/certified_fewsteps.py --ckpt runs/toy_checkerboard/model.pt   # n_cal 4000, n_test 50000
python scripts/certified_fewsteps.py --ckpt runs/toy_8gaussians/model.pt
```
**Method** (`muninnx/baselines/fewsteps.py`). On the same 4000 calibration chains Muninn uses,
compute d(N') = ‖x_N^full − x_{N'}^Euler‖ for every N' = 1..100. Choose N' by **fixed-sequence
testing**: go down from N' = N and stop at the first N' that fails the test of
"P(d > η) ≤ α" (no multiple-testing correction needed). Two tests, with k violations out of n:
- **conformal**: (k+1)/(n+1) ≤ α. Marginal guarantee, the same kind Muninn claims.
- **LTT**: binomial p-value P(Binom(n, α) ≤ k) ≤ δ = 0.05. PAC guarantee: with prob ≥ 95% over
  calibration, P(d > η) ≤ α.

The cost is one calibration pass (Σ N' ≈ 5000 evaluations per chain, offline). At sampling time
it is plain Euler with N' steps: no probe, no controller.

**Checkerboard, α = 0.05** (test p_viol, n = 50k)

| η | Muninn NFE | Muninn E[d] | Muninn p_viol | CertFew (conf.) N' | E[d] | p_viol | CertFew (LTT) N' | p_viol | NFE ratio Muninn / CertFew |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 100 | 0 | 0 | 90 | 0.0049 | 0.040 | 92 | 0.023 | 1.1× |
| 0.05 | 97.0 | 0.0085 | 0.008 | 60 | 0.0261 | 0.046 | 61 | 0.043 | 1.6× |
| 0.1 | 94.0 | 0.0310 | 0.020 | 43 | 0.0512 | 0.045 | 44 | 0.042 | 2.2× |
| 0.15 | 91.0 | 0.0691 | 0.041 | 34 | 0.0748 | 0.046 | 35 | 0.042 | 2.7× |
| 0.3 | 81.9 | 0.1515 | **0.059** | 22 | 0.1358 | 0.045 | 23 | 0.042 | 3.7× |
| 0.5 | 70.2 | 0.2682 | **0.083** | 16 | 0.2017 | 0.045 | 17 | 0.039 | 4.4× |
| 1.0 | 41.2 | 0.3544 | 0.038 | 9 | 0.3813 | 0.045 | 10 | 0.035 | 4.6× |

**8gaussians, α = 0.05**

| η | Muninn NFE | Muninn p_viol | CertFew (conf.) N' | p_viol | CertFew (LTT) N' | p_viol | NFE ratio |
|---|---|---|---|---|---|---|---|
| 0.02 | 99.0 | 0.0002 | 92 | 0.012 | 92 | 0.012 | 1.1× |
| 0.05 | 97.7 | 0.0003 | 46 | 0.013 | 90 | 0.040 | 2.1× (conf.) / 1.1× (LTT) |
| 0.1 | 95.0 | 0.0006 | 26 | 0.011 | 26 | 0.011 | 3.7× |
| 0.3 | 85.2 | 0.0001 | 10 | 0.020 | 10 | 0.020 | 8.5× |
| 1.0 | 51.5 | 0 | 3 | 0.034 | 3 | 0.034 | 17× |

α = 0.10 tables are in `results.json`. Same picture: CertFew is valid everywhere (test p_viol
≤ α in all 32 rows) and needs 1.1–17× fewer NFE than Muninn.

**Interpretation**
- **CertFew dominates Muninn on every axis that matters.** It is valid (Muninn is not, on
  checkerboard), needs far fewer NFE for the same (η, α), and has no probe cost. The certificate
  is simple, end-to-end and makes no Lipschitz assumption, so it is immune to the amplification
  problem from E3.
- **CertFew is tight, Muninn is loose.** CertFew's test p_viol sits just below α (0.035–0.046),
  meaning it spends its risk budget. Muninn's p_viol is often 10–100× below α while still using
  most of the NFE. Its E[d] is smaller at the same η, but that is not what the guarantee asks for.
- **Non-monotone d(N') hurts fixed-sequence testing.** On 8gaussians, E[d] is slightly *larger*
  at N' = 90 than at N' = 80 (see E2 FewSteps curve), so at η = 0.05 the LTT test fails at
  N' = 89 and stops (N' = 90) while the conformal test passes through (N' = 46). A fix would be
  to test in an order that is not N, N−1, …, e.g. by error estimated on a separate split.
- **This is now the baseline to beat.** Any adaptive method must offer the same (η, α)
  guarantee at a lower NFE than CertFew, not just than uniform Euler at matched NFE.
- Caveat: the guarantee is per (η, α) and per model. CertFew needs a calibration pass for each
  new tolerance, but that is cheap and offline, like Muninn's.

### E5 — x̂₁ reuse ≡ v reuse (2026-11-19, analysis + numerical check)
```bash
python scripts/check_x1hat_equivalence.py --ckpt runs/toy_checkerboard/model.pt
```
**Claim (GUIDE).** Reusing the predicted endpoint x̂₁ = x + (1−t)v gives the step
Φ(x) = x + Δ(x̂₁ − x)/(1−t) with K = 1 − Δ/(1−t) < 1, so errors contract.

**Result.** For a fixed x̂₁, Euler on dx/dt = (x̂₁ − x)/(1−t) scales the distance to x̂₁ by
exactly (1−t_{k+1})/(1−t_k) per step, which telescopes:
x_k = x̂₁ + (x_j − x̂₁)(1−t_k)/(1−t_j) = x_j + (t_k − t_j)·v_j.
That is one big Euler step with the old velocity, i.e. exactly what v-reuse produces. The
contraction K < 1 is cancelled by the larger sensitivity L' = Δ/(1−t). The same holds for x̂₀ reuse.
Numerically, the same random reuse pattern (20k samples, float64) gives final samples that differ
by at most **4.0e-14** (sample norm ≈ 2.9).

**Consequence.** With Euler on straight FM paths, *any* "copy a cached prediction" scheme is a
non-uniform Euler schedule. Changing the parametrisation only changes the units of the
score / U(s), so it changes Muninn's decisions but cannot beat the E2 oracle. E7 as originally
planned is dropped. To gain anything, the cache must be **combined with new information**
(extrapolation, multistep, Heun stages): see E6.

### E6 — Oracle extrapolation vs higher-order FewSteps (2026-11-19)
```bash
python scripts/extrap_oracle_toy.py --ckpt runs/toy_checkerboard/model.pt   # n_test 50000, RK4-400 reference
python scripts/extrap_oracle_toy.py --ckpt runs/toy_8gaussians/model.pt
```
**Question.** Does adaptive *extrapolation* (forecast v from the last two computed velocities)
beat FewSteps, where adaptive *reuse* failed?

**Methods** (`muninnx/cache/oracle.py::oracle_extrap_sample`, `muninnx/baselines/fewsteps.py`)
- Oracle extrapolate: at a cacheable step, the candidate is
  v_ext = v_b + (t_k − t_b)(v_b − v_a)/(t_b − t_a) from the last two *computed* velocities. It is
  used iff Δ·‖v(x_k) − v_ext‖ ≤ budget and lag < max_lag. Same greedy budget rule as E2. The step is
  the Euler step of the reference grid. Variants: max_lag = 10 and no lag cap.
- Oracle reuse: E2 oracle, for reference.
- FewSteps Euler / AB2 (1 NFE/step, first step Euler) / Heun (2 NFE/step), uniform grids.
- **Two references**: the full Euler-100 chain (Muninn's certificate target) and the near-exact
  ODE solution (RK4, 400 steps). Euler-100 is itself 0.0386 (checkerboard) / 0.0232 (8gaussians)
  away from exact in E[d].

Plot: `runs/toy_*/extrap_oracle/pareto.png` (left: vs Euler-100, right: vs exact).

**Checkerboard**, E[d] vs exact ODE (vs Euler-100 in brackets)

| NFE | FewSteps Euler | FewSteps AB2 | FewSteps Heun | Oracle reuse | Oracle extrapolate (lag 10) |
|---|---|---|---|---|---|
| ~10 | 0.379 | **0.035** | 0.093 | — | — |
| ~18–20 | 0.192 | **0.009** | 0.020 | 0.395 (0.357) @18 | 0.029 (0.045) @18 |
| ~22 | — | — | — | 0.392 (0.355) | 0.030 (0.045) @22 |
| ~47–50 | 0.077 | **0.002** | 0.003 | — | 0.038 (0.035) @47 |
| ~67–70 | 0.055 | **0.001** | — | 0.244 (0.206) @69 | 0.060 (0.031) @67 |
| 100 | 0.039 | **0.0004** | 0.002 | — | — |

**8gaussians**, E[d] vs exact ODE

| NFE | FewSteps Euler | FewSteps AB2 | FewSteps Heun | Oracle reuse | Oracle extrapolate (lag 10) |
|---|---|---|---|---|---|
| ~10 | 0.229 | **0.019** | 0.045 | — | — |
| ~18–20 | 0.113 | **0.010** | 0.016 | 0.247 @18 | 0.082 @18 |
| ~48–50 | 0.050 | **0.009** | 0.014 | — | 0.089 @48 |
| ~70 | — | **0.005** | — | — | 0.075 @70 |

**Interpretation**
- **Extrapolation is a real improvement over reuse:** ~13× (checkerboard) / ~3× (8gaussians)
  lower error than the reuse oracle at 18 NFE. On checkerboard, at 18 NFE it is closer to the
  exact ODE than Euler-100 (0.029 vs 0.039).
- **But it is a higher-order integrator in disguise.** At large η the budget never binds and the
  oracle settles into a fixed pattern: recompute every 11 steps, extrapolate in between. That is
  essentially AB2 with ~9 big steps, plus 10 NFE wasted on the forbidden prefix/suffix. Uniform
  AB2 at 10 NFE gives the same error (0.035 vs 0.029).
- **Uniform AB2 beats the adaptive oracle everywhere** that matters: 3× (checkerboard) and 8×
  (8gaussians) lower error at ~18–20 NFE. The per-sample decisions add nothing on these toys.
- **Without a lag cap, extrapolation diverges.** Linear extrapolation over long spans is unstable,
  so the no-cap oracle is the worst method. The cap is what makes it a stable multistep scheme.
- **More NFE can make the oracle *worse* vs exact.** At small η it recomputes more and tracks the
  Euler-100 chain more closely, inheriting Euler-100's own error (0.039). This is the target
  problem (finding 9) showing up directly.
- **Vs the Euler-100 target**, AB2 and Heun plateau at ≈ 0.038 / 0.023: that is Euler-100's own
  discretisation error, not an AB2 failure. A certificate against Euler-100 with η below that
  value can only be met by tracking Euler's error.
- Heun (2 NFE/step) is worse than AB2 at equal NFE on both toys.

### E7 — Why the certificate breaks: corrected recursion + Tweedie Jacobian (2026-11-21)
```bash
python scripts/theory_check.py --ckpt runs/toy_checkerboard/model.pt   # n 20000, oracle η ∈ {0.05, 0.3}
python scripts/theory_check.py --ckpt runs/toy_8gaussians/model.pt
```
Derivation in [THEORY_NOTES.md](THEORY_NOTES.md). In short: the exact recursion is
Δ_{k+1} = (I + h J̄_k) Δ_k + h e_k, so d = ‖Σ_k h M_{k+1→N} e_k‖. The paper sets M = I (K = 1 for
Euler), dropping the drift term h J̄_k Δ_k. For FM, ∂v/∂x = [t/(1−t)² Cov(x₁|x_t) − I]/(1−t), which
expands perturbations wherever the posterior over the endpoint is still wide.

**Bounds vs actual deviation** (E2 oracle chains, same noise; B = Σ h·w_k‖e_k‖ with weight w_k)

| dataset | η | NFE | w = 1 (paper) | w = ‖G‖ along full chain | w = ‖M‖ mean-value (exact) | median d / B_paper | median d / B_G |
|---|---|---|---|---|---|---|---|
| checkerboard | 0.05 | 81.6 | 0.216 | 0.859 | **0.9989** | 1.61 | 0.73 |
| checkerboard | 0.3 | 22.2 | 0.234 | 0.860 | **1.0000** | 1.26 | 0.73 |
| 8gaussians | 0.05 | 86.5 | 0.776 | 1.000 | **1.0000** | 0.79 | 0.42 |
| 8gaussians | 0.3 | 29.5 | 0.902 | 1.000 | **1.0000** | 0.75 | 0.47 |

(fraction of samples with d ≤ B)

**Flow Jacobian ‖G_{k+1→N}‖_op along the full chain** (median / p90 / p99)

| k | checkerboard | 8gaussians |
|---|---|---|
| 5 | 2.33 / 5.21 / 23.2 | 2.07 / 8.89 / 45.4 |
| 20 | 2.36 / 5.30 / 23.5 | 1.95 / 8.32 / 42.0 |
| 40 | 1.83 / 4.04 / 17.6 | 1.26 / 4.62 / 24.4 |
| 60 | 1.44 / 2.13 / 8.2 | 1.02 / 1.70 / 8.1 |
| 80 | 1.23 / 1.25 / 2.5 | 1.12 / 1.13 / 1.8 |
| 94 | 1.06 / 1.07 / 1.1 | 1.05 / 1.05 / 1.1 |

**Tweedie check**: network ∂v/∂x vs the closed form with Cov(x₁|x_t) estimated by self-normalised
Monte Carlo over 200k data samples (largest eigenvalue, median / p90)

| t | checkerboard net | checkerboard Tweedie | corr | 8gaussians net | 8gaussians Tweedie | corr |
|---|---|---|---|---|---|---|
| 0.10 | −0.30 / −0.27 | −0.30 / −0.28 | 0.87 | −0.01 / +0.08 | −0.02 / +0.01 | 0.79 |
| 0.30 | +1.53 / +1.77 | +1.53 / +1.76 | 1.00 | +2.06 / +4.03 | +2.04 / +4.01 | 0.99 |
| 0.50 | +1.89 / +3.50 | +1.89 / +3.48 | 1.00 | +1.45 / +4.84 | +1.51 / +4.72 | 0.99 |
| 0.70 | +0.81 / +3.79 | +0.81 / +3.79 | 0.99 | −0.51 / +1.84 | −0.54 / +1.12 | 0.98 |
| 0.85 | +1.15 / +1.32 | +1.17 / +1.42 | 0.89 | +0.34 / +0.41 | +0.41 / +0.67 | 0.88 |

Figure: `runs/toy_*/theory/theory.png` (left: d vs both bounds; middle: map of ‖G‖ for an error
made at t = 0.2; right: λ_max(∂v/∂x) vs t, network and Tweedie).

**Interpretation**
- **The theory is exact.** With Jacobians averaged between the two chains (the mean-value form of
  the recursion), the corrected bound holds for 99.9–100% of samples. The residual 0.1% is the
  8-point quadrature of that average. The paper bound fails for 77–78% (checkerboard) and 10–22%
  (8gaussians).
- **Linearising along the full chain is not enough on checkerboard** (86%): the corrected bound
  is fairly tight there (median d/B = 0.73) and the deviation is large enough (linear predictor
  rel. error p90 17–51%) that the Jacobian must be taken between the chains, not on one of them.
  On 8gaussians the slack (d/B ≈ 0.45) absorbs it.
- **The trained network follows the Tweedie formula closely** (λ_max correlation 0.98–1.00 for
  t ∈ [0.3, 0.7]). The amplification is a property of the *data distribution*, not an artefact of
  this MLP. (At t = 0.1 on 8gaussians, v ≈ 0 by symmetry, so relative errors are ill-defined.)
- **Quantitative prediction confirmed.** Inside a Gaussian mode of std s, the sign of λ flips at
  t = 1/(1+s²) = 0.8 for 8gaussians: predicted λ = −0.59 at t = 0.7 (measured −0.51) and +0.31 at
  t = 0.85 (measured +0.34).
- **Amplification sits on the basin boundaries.** The ‖G‖ map shows 8 thin rays for 8gaussians
  and the cell boundaries for checkerboard: the noise points where a small push changes which
  mode the sample ends in. Errors elsewhere are amplified ~1–2×.
- **Data geometry explains E2.** Narrow modes contract late in sampling, so the paper bound is
  often loose-but-valid on 8gaussians. Flat squares expand everywhere (λ > 0 for t ≥ 0.3), so it
  fails on most checkerboard samples.

### E8 — End-to-end certified Muninn vs certified FewSteps Euler / AB2 (2026-11-21)
```bash
python scripts/certified_compare.py --ckpt runs/toy_checkerboard/model.pt   # n_fit 2000, n_cal 4000, n_test 50000
python scripts/certified_compare.py --ckpt runs/toy_8gaussians/model.pt
```
**Question.** Once every method is certified the *same* way, which one needs the least compute
for a given guarantee $P(d > \eta) \le \alpha$?

**Method** (option (b) of [THEORY_NOTES.md](THEORY_NOTES.md) §5). Each method has one knob,
tested by fixed-sequence testing on 4000 calibration chains, from the safest setting to the most
aggressive; the last setting that certifies is deployed and evaluated on 50k fresh test chains.
- **Certified FewSteps Euler / AB2**: knob = step count $N'$ (from 400 down; against the Euler-100
  reference, Euler starts at $N' = 100$, where $d = 0$).
- **Certified Muninn**: knob = the controller's internal budget $\lambda$ (grid 0 … 10, from 0 up).
  $U(s)$ is fitted on 2000 *separate* chains and is now only a decision heuristic: validity comes
  from the calibration, not from the per-step bound.
- Two tests: conformal, $(k+1)/(n+1) \le \alpha$ (marginal), and LTT,
  $P(\mathrm{Binom}(n, \alpha) \le k) \le \delta = 0.05$ (PAC).
- Two references: the exact ODE (RK4-400, main) and the full Euler-100 chain.
- Probe cost (multiply-accumulate count of the stem vs the whole MLP): **0.079** forward passes,
  i.e. $+7.9$ NFE-equivalents per Muninn sample.

Plot: `runs/toy_*/certified_compare/certified_nfe.png` (NFE needed vs $\eta$, $\alpha = 0.05$).

**Validity.** LTT: 0 / 146 certified rows exceed $\alpha$ on test (worst margin $-0.004$).
Conformal: 5 / 146 exceed $\alpha$, by at most $+0.0032$, as expected for a guarantee that only
holds on average over calibration draws. **Muninn is now valid on checkerboard**, where the
per-step certificate failed (E2/E4).

**Mean NFE for $P(d > \eta) \le 0.05$, LTT, reference = exact ODE** (inf = not certifiable)

| $\eta$ | checkerboard Euler | checkerboard AB2 | checkerboard Muninn | 8gaussians Euler | 8gaussians AB2 | 8gaussians Muninn |
|---|---|---|---|---|---|---|
| 0.02 | 370 | **21** | inf | 170 | 170 | inf |
| 0.05 | 150 | **14** | inf | 91 | **90** | 95.0 |
| 0.1 | 74 | **10** | 97.0 | 35 | **10** | 92.7 |
| 0.2 | 40 | **8** | 94.0 | 16 | **6** | 75.6 |
| 0.3 | 29 | **7** | 87.9 | 11 | **4** | 63.8 |
| 0.5 | 19 | **6** | 81.4 | 7 | **4** | 18.0 |

Against the Euler-100 reference (conformal, $\alpha = 0.05$), Muninn is certifiable at small
$\eta$ but still needs 94–98 NFE where certified Euler needs 42–90 (checkerboard). AB2 cannot
certify $\eta < 0.1$ there, because it does not reproduce Euler-100's own error.

**Interpretation**
- **(b) works as a fix for validity.** One scalar knob, chosen on calibration chains, turns any
  caching policy into a certified one, with no Lipschitz assumption. The per-step bound becomes a
  heuristic that only orders the decisions.
- **It also corrects Muninn's conservatism.** On 8gaussians the chosen $\lambda$ is *larger* than
  $\eta$ (0.5 at $\eta = 0.2$, 0.75 at $\eta = 0.3$) because the flow contracts errors there; on
  checkerboard it stays $\le \eta$. The calibration learns what the per-step bound got wrong in
  both directions.
- **But certified Muninn is the most expensive method at every tolerance.** It needs 64–97 NFE
  (plus 7.9 for the probe) where certified AB2 needs 4–14: a 6–14× gap. Even certified Euler
  beats it by 1.3–5× everywhere except very small $\eta$ against the Euler-100 reference.
- **Against the exact ODE, Muninn has an accuracy floor.** It can only approximate the Euler-100
  chain, which is itself 0.039 / 0.023 away from the exact solution, so tolerances below that
  cannot be certified at any budget. Higher-order FewSteps has no such floor.
- **Non-monotone risk still costs the fixed-sequence test.** On 8gaussians at $\eta = 0.05$, LTT
  stops AB2 at $N' = 90$ (the conformal test reaches $N' = 26$), as in E4.
- **Conclusion for toys.** With validity fixed, the ranking is clear: certified AB2 ≪ certified
  Euler < certified Muninn. Velocity reuse with an Euler base is dominated. The only open route
  for adaptivity is a policy that is both higher-order (extrapolation) and boundary-aware ($G$),
  certified the same way: that is what E9 tests, first as an oracle.

### E9 — $G$-aware adaptive extrapolation oracle vs certified AB2 (2026-11-21)
```bash
python scripts/g_aware_oracle.py --ckpt runs/toy_checkerboard/model.pt      # n_cal 4000, n_test 50000
python scripts/g_aware_oracle.py --ckpt runs/toy_8gaussians_b16k/model.pt   # smooth retrain, see finding 13
python scripts/g_aware_oracle.py --ckpt runs/toy_8gaussians/model.pt        # original (rough in t)
```
**Question.** Give adaptivity every advantage: higher-order base, extrapolation, and perfect
knowledge of both the injected error and its downstream amplification. Does it beat certified AB2?

**Policies** (`muninnx/cache/oracle.py::oracle_extrap_ab2_sample`). Base grid: AB2 with
$N = 100$ (with full compute it is 0.0008 from the exact ODE, so no accuracy floor). At each step
$k \ge 2$, per sample, either compute $v$ or extrapolate it linearly from the last two computed
velocities (at most 10 in a row). The state update is AB2 on the velocities actually used. The
decision uses the true error $e_k = v(x_k) - v_{\text{ext}}$ (the network is evaluated every step
for free; only computed steps count as NFE) through one of two costs:
- plain: $c_k = h\lVert e_k\rVert$;
- **$G$-aware**: $c_k = h\lVert G_{k+1\to N}\, e_k\rVert$, with $G$ the flow Jacobian along the
  AB2-100 reference chain (`muninnx/analysis/flow.py`), i.e. the error's actual effect on the
  final sample;

and one of two rules: **threshold** (extrapolate iff $c_k \le \tau$) or **budget** (greedy, as in
Muninn: extrapolate iff $c_k \le$ remaining budget $\lambda$). Sanity check: knob $= 0$ reproduces
plain AB2-100 exactly (max difference 0.0); knob $= \infty$ gives the 10-NFE floor (2 start-up
steps + one compute every 11 steps).

**Certification.** As in E8, against the exact ODE, $\alpha = 0.05$, 4000 calibration / 50k test
chains. Because $d(N')$ is not monotone for AB2 (finding 13, and mildly on checkerboard), the main
selection rule is **order-free LTT with a Bonferroni correction** (`bonferroni_select`): every
knob value is tested at level $\delta/\lvert\text{knobs}\rvert$ and the cheapest certified one is
kept. All 144 certified rows have test $p_{\text{viol}} \le \alpha$.

**Mean NFE for $P(d > \eta) \le 0.05$** (LTT + Bonferroni, exact-ODE reference)

| $\eta$ | checkerboard AB2 | checkerboard $G$-thr. | checkerboard plain-thr. | 8gauss. (smooth) AB2 | 8gauss. (smooth) $G$-thr. | 8gauss. (smooth) plain-thr. |
|---|---|---|---|---|---|---|
| 0.005 | **42** | 47.7 | 64.4 | 55 | **46.0** | 67.5 |
| 0.01 | 33 | **25.8** | 38.1 | 36 | **27.5** | 39.4 |
| 0.02 | 22 | **18.1** | 21.5 | 21 | **19.8** | 22.8 |
| 0.05 | 15 | **11.1** | 12.5 | **11** | 14.9 | 17.1 |
| 0.1 | 11 | **10.0** | 10.0 | **8** | 10.0 | 17.1 |
| 0.2 | **8** | 10 | 10 | **5** | 10 | 10 |
| 0.5 | **6** | 10 | 10 | **3** | 10 | 10 |

Budget-rule policies need 48–95 NFE for $\eta \le 0.05$ on both models (full table in
`results.json`). On the original, rough-in-$t$ 8gaussians model the fixed-sequence test made the
oracles look 4–9× better than AB2; with Bonferroni the gap shrinks to at most 1.15× (and AB2 wins
at most $\eta$), which is how finding 13 was found.

Plot: `runs/toy_*/g_aware_oracle/certified_nfe.png` (left: certified NFE vs $\eta$; right:
uncertified p95 error vs NFE on calibration chains).

**Interpretation**
- **First positive result for adaptivity, but small.** With perfect information, the $G$-aware
  threshold oracle beats certified AB2 by **1.1–1.35×** at tight tolerances ($\eta \le 0.05$ on
  checkerboard, $\eta \le 0.02$ on smooth 8gaussians). In the uncertified view it reaches about
  **2× lower p95 error at equal NFE** between 15 and 60 NFE; because the error falls steeply with
  NFE, that converts to only ~1.3× fewer NFE.
- **$G$ is what makes it work.** With the plain cost $h\lVert e\rVert$ the oracle is roughly tied
  with AB2 (sometimes worse). Weighting by the downstream effect, i.e. extrapolating freely away
  from basin boundaries and computing near them, is worth 1.1–1.5×. This is the per-sample effect
  predicted in THEORY_NOTES §4.
- **Threshold, not budget.** A greedy budget spends everything in the first steps and then must
  compute all remaining steps; at tight $\eta$ it needs 3–8× more NFE than the threshold rule. The
  paper's budget mechanism is the wrong allocation rule.
- **The structure caps the gain at loose tolerances.** With $N = 100$ and at most 10 consecutive
  extrapolations, the oracle cannot go below 10 NFE, while AB2 certifies $\eta \ge 0.2$ with 3–8.
  A coarser base grid would remove this floor (not tested).
- **What a real method would face.** The oracle gets $e_k$ and $G$ for free. A practical version
  needs (i) a probe to predict the extrapolation error (≈ 0.08 NFE per step, i.e. ≈ 8 NFE per
  sample here, as large as the whole gain at 25 NFE), (ii) a predictor of $\lVert G\rVert$ from
  the current state (the future is unknown), and (iii) conservative calibration of both. On these
  toys that would at best tie with certified AB2.

---

## Open questions
- Can the amplification be bounded *with high probability* (conformal on G_k) rather than in the
  worst case? The tail (max ~900×) suggests a per-step worst-case Lipschitz bound would be useless.
- Is there *any* setting where per-sample adaptivity beats a uniform grid of the same order?
  It needs difficulty to vary a lot **between samples** (not just over t, which a fixed
  non-uniform grid captures). Candidates: Maze2D with branching paths, conditional generation.
- Would adaptivity help on top of a higher-order base (e.g. adaptive AB2 / adaptive step size with
  an embedded error estimate, as in classical ODE solvers)? That is classical adaptive step-size
  control, so the novelty would be the certificate, not the speedup.
- What does the paper's 2.7× speedup look like against FewSteps in *their* diffusion setting?
  The DDPM-style samplers there may have a weaker step-reduction baseline than FM + Euler.

## Next experiments (planned)
| id | experiment | hypothesis / what it decides |
|---|---|---|
| E10 | **Can $\lVert G\rVert$ be predicted from the current state?** Regress $\log\lVert G_{k+1\to N}\rVert$ on probe features / $(x_k, t_k)$ on calibration chains | Decides whether the E9 gain is reachable by a real method. If the probe cannot locate basin boundaries, the ceiling is unreachable. |
| E11 | **Realistic $G$-aware extrapolation**: predicted error + predicted $G$ + probe cost, certified end-to-end | Only if E10 is positive. Target: beat certified AB2 at $\eta \le 0.05$ including probe cost. |
| E12 | Coarser base grid ($N = 50$) / larger max_lag for the E9 oracle | Removes the 10-NFE floor; does the ceiling grow at loose $\eta$? |
| E13 | Fix (a) for the record: conformal bound on $h\lVert G\rVert$ in the per-step certificate | Only if the per-step certificate is wanted for the write-up; expected very conservative. |
| E14 | Same E7 check on a DDIM / VP-path sampler | Is the certificate gap generic (not FM-specific)? |
| E15 | **Adaptive step size on an AB2 base** (classical embedded error estimate), certified end-to-end | Classical per-sample adaptivity as a baseline for E11. |
| E16 | **Non-uniform FewSteps** grid (more steps where $\mathbb E[\varepsilon]$ is large) | Captures time-dependent difficulty without per-sample decisions. |
| E17 | Ablations: literal paper rule (max_lag = 1, score vs $F_{k-1}$), per-lag $U$, forbid sizes | Completes GUIDE Finding 1 (a) and the M3 ablations. |
| M1 | Maze2D | The place where per-sample difficulty may actually vary. Worth it if E10/E11 show the E9 gain is reachable, or as the final real-data check. |
| — | ~~x̂₁ reuse~~ | Dropped: provably identical to v reuse with Euler (E5). |

## Changelog of code affecting results
- 2026-11-18 `toy2d.checkerboard` fixed (was a staircase). E1 is invalidated; E2/E3 use the fixed target.
- 2026-11-18 `train_toy.py`: data generated on GPU and TF32 matmuls (training only; eval stays fp32).
- 2026-11-19 merge with upstream: `controller.py` from upstream (full-batch call when all rows
  recompute; `bound_used` accumulated directly). Re-checked: 8gaussians η = 0.3 gives NFE 85.4,
  E[d] 0.068, identical to before.
- 2026-11-21 new model `runs/toy_8gaussians_b16k` (batch 16384, 30k iters) because the original
  8gaussians model is rough in $t$ (finding 13). E0–E8 still use the original model.
- 2026-11-21 `bonferroni_select` (order-free LTT) added; E9 uses it as the main selection rule.
- 2026-11-21 `runs/` is now tracked in git (toy checkpoints, `results.json`, plots: ~7 MB), so every
  number in this file can be checked without rerunning. Heavy future artefacts must be ignored
  explicitly in `.gitignore`.

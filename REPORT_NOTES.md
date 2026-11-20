# Muninn-X — experiment notes

Running log of experiments: setup, results and interpretation. `runs/` is gitignored, so
**this file is the durable record**. Every entry lists the command that reproduces it.

- Newest findings go in **Key findings** (top). Each experiment gets an entry under **Log**.
- Planned experiments live in **Next experiments** until they are run, then move to the log.
- Conventions follow [GUIDE.md](GUIDE.md): time noise→data (t=0→1), Euler, d = ‖·‖ (Γ = 1 for 2D toys).

---

## Key findings (as of 2026-11-19)

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
| E7 | **Certified FewSteps with AB2, against the exact ODE** (E4 method, new reference) | The new bar. Expect far fewer NFE than certified Euler for the same (η, α). |
| E8 | **Adaptive step size on an AB2 base** (embedded error estimate, certified end-to-end) | Last test of per-sample adaptivity on toys. Must beat E7 at equal (η, α). |
| E9 | **Fix Muninn's validity**: conformal bound on the measured sensitivity Δ·G_k, or end-to-end calibration | Needed only if we keep the per-step certificate. Restores p_viol ≤ α on checkerboard. |
| E10 | **Non-uniform FewSteps** grid (more steps where E[ε] is large) | Captures time-dependent difficulty without per-sample decisions. |
| E11 | Ablations: literal paper rule (max_lag = 1, score vs F_{k−1}), per-lag U, forbid sizes | Completes GUIDE Finding 1 (a) and the M3 ablations. |
| M1 | Maze2D | The place where per-sample difficulty may actually vary. Worth it if E8 shows any adaptive gain, or as the final real-data check. |
| — | ~~x̂₁ reuse~~ | Dropped: provably identical to v reuse with Euler (E5). |

## Changelog of code affecting results
- 2026-11-18 `toy2d.checkerboard` fixed (was a staircase). E1 is invalidated; E2/E3 use the fixed target.
- 2026-11-18 `train_toy.py`: data generated on GPU and TF32 matmuls (training only; eval stays fp32).
- 2026-11-19 merge with upstream: `controller.py` from upstream (full-batch call when all rows
  recompute; `bound_used` accumulated directly). Re-checked: 8gaussians η = 0.3 gives NFE 85.4,
  E[d] 0.068, identical to before.

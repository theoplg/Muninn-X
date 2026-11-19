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

**Implication.** Two separate problems: (a) *validity*, so the sensitivity weights must account
for the flow's amplification, or calibration must target the final d directly; (b) *usefulness*,
so we must change what is reused (x̂₁, Heun stage) rather than tune the score / bound.

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

---

## Open questions
- Can the amplification be bounded *with high probability* (conformal on G_k) rather than in the
  worst case? The tail (max ~900×) suggests a per-step worst-case Lipschitz bound would be useless.
- Does x̂₁ reuse (contracting step, K = 1 − Δ/(1−t)) also reduce the effect of the flow's
  amplification, or only the injected error?
- Is there a regime where adaptive reuse beats a **tuned non-uniform** grid, not just uniform
  FewSteps? Per-sample adaptivity only helps if difficulty varies a lot *between samples*.
- What does the paper's 2.7× speedup look like against FewSteps in *their* diffusion setting?
  The DDPM-style samplers there may have a weaker step-reduction baseline than FM + Euler.

## Next experiments (planned)
| id | experiment | hypothesis / what it decides |
|---|---|---|
| E4 | **Certified FewSteps**: pick the smallest N' with a conformal guarantee P(d > η) ≤ α | Trivially valid. Probably the strongest certified baseline, and every adaptive method must beat it. |
| E5 | **Fix validity, option A**: replace L_k = Δ by a conformal upper bound on the measured sensitivity Δ·G_k | Restores p_viol ≤ α on checkerboard. Cost: a much larger ĉ, so fewer reuses. |
| E6 | **Fix validity, option B**: end-to-end calibration on the final d instead of per-step bounds | Avoids the triangle inequality and the amplification issue. Harder with adaptive decisions. |
| E7 | **x̂₁ reuse** instead of v (Φ = x + Δ(x̂₁ − x)/(1−t)) | Contracting K < 1. Does the oracle then beat FewSteps? |
| E8 | **Non-uniform FewSteps** baseline (e.g. more steps near t → 1 or where ε is large) | Is uniform even the right FewSteps reference? |
| E9 | Ablations: literal paper rule (max_lag = 1, score vs F_{k−1}), per-lag U, forbid sizes | Completes GUIDE Finding 1 (a) and the M3 ablations. |
| M1 | Maze2D | Only worth it once E4–E7 show a configuration that is both valid and useful. |

## Changelog of code affecting results
- 2026-11-18 `toy2d.checkerboard` fixed (was a staircase). E1 is invalidated; E2/E3 use the fixed target.
- 2026-11-18 `train_toy.py`: data generated on GPU and TF32 matmuls (training only; eval stays fp32).
- 2026-11-19 merge with upstream: `controller.py` from upstream (full-batch call when all rows
  recompute; `bound_used` accumulated directly). Re-checked: 8gaussians η = 0.3 gives NFE 85.4,
  E[d] 0.068, identical to before.

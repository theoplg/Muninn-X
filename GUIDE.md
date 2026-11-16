# Muninn-X — implementation guide (v0: 2D toy, Euler)

The goal of v0 is to get the whole Muninn pipeline working on a tiny problem where
everything runs in minutes, before moving to Maze2D:

**train FM → sample → probe/score → calibrate → certified reuse → paired evaluation**.

The plumbing (data, model, plotting, scripts, tests) is already written. Every file
marked `>>> YOU IMPLEMENT THIS <<<` contains `TODO(you)` stubs. Implement them **in
order**: each step has a test file, and later steps depend on earlier ones.

```bash
conda env create -f environment.yml      # or: pip install -e ".[dev]" in an existing env
pytest -q                                # everything fails at first; that's expected
pytest -q tests/test_1_fm_loss.py        # run one step's tests at a time
```

## Conventions (read once)
- **Time runs from noise to data**: x₀ ~ N(0, I) at t = 0, data at t = 1. The paper's
  diffusion time runs backwards, so its "t+1" is our "k−1" (the previous step).
- **Grid**: `solver.ts[0..N]`. Step `k` maps x_k → x_{k+1} using the velocity at (x_k, ts[k]).
- **One step as a map**: x_{k+1} = Φ_k(x_k, v). Its Lipschitz constants are
  ‖Φ_k(x,v) − Φ_k(x',v')‖ ≤ K_k‖x−x'‖ + L'_k‖v−v'‖ (the paper's Assumption 1).
- **Distance**: d(a,b) = ‖a−b‖_F/√H, so Γ = 1/√H. For the 2D toy, H = 1 and Γ = 1.

---

## Step 1 — Flow-matching loss · `muninnx/fm/loss.py`
**Theory.** On the straight (CondOT) path x_t = (1−t)x₀ + t·x₁, the conditional velocity
is dx_t/dt. Regressing v_θ(x_t, t) onto it gives a field whose ODE carries N(0, I) to the data.

**Hints**
- `t` has shape (B,). To broadcast it over x of shape (B, ...), use `t.view(-1, *[1]*(x.ndim-1))`.
- `torch.randn_like` takes no `generator`. Use `torch.randn(x1.shape, generator=generator)`
  (the generator lives on CPU), then `.to(x1.device)`. Do the same for `torch.rand`.

**Check**
```bash
pytest -q tests/test_1_fm_loss.py
python scripts/train_toy.py --dataset 8gaussians --iters 20000
```
The loss should fall and then plateau. It does not go to 0: the conditional target
x₁ − x₀ is noisy given x_t, so the minimum loss is strictly positive.

## Step 2 — Euler solver · `muninnx/solvers/euler.py`
**Theory.** Φ_k(x, v) = x + Δ_k·v. Derive (K_k, L'_k) yourself: it takes one line.

**Check**
```bash
pytest -q tests/test_2_solvers.py
python scripts/sample_toy.py --ckpt runs/toy_8gaussians/model.pt --n-steps 100 10 3
```
Open `runs/toy_8gaussians/samples.png` and `paths.png`. The paths should be **almost straight**.
Notice how good N=10 (and even N=3) already looks. That is the reason **FewSteps is the
baseline to beat** for flow matching, and it is a central point of this project.

## Step 3 — Probe and score · `muninnx/cache/probe.py`, `muninnx/cache/score.py`
**Theory.** The probe runs only a cheap prefix of the network:
F_k = Ψ(x_k, t_k) = `stem(embed_input(x, t))`. The score
s_k = ‖F_k − F_{k−1}‖₁ / (‖F_{k−1}‖₁ + ω) measures how much the network's internal input
changed since the last step. It is the signal that predicts the reuse error.

**Hints.** Use per-sample l1 norms (sum over the feature dim) and return shape (B,).
For the MLP the probe costs about 1/4 of a full forward pass. On the U-Net later it will be much less.

**Check.** `pytest -q tests/test_3_probe_score.py`. Later, in step 7, look at
`calibration_vs_step.png`: s_k and ε_k should move together.

## Step 4 — Paired evaluation · `muninnx/eval/paired.py`
**Theory.** Full and cached chains start from the **same noise**: that is what makes d meaningful.
p_viol = fraction of samples with d > η. The guarantee says p_viol ≤ α.

**Hints.** Flatten all dims except the batch. H = x.shape[1] if x.ndim == 3, else 1.
Use the provided `wilson_ci` for the confidence interval.

**Check.** `pytest -q tests/test_4_paired.py`

## Step 5 — Bound and calibration · `muninnx/cache/bound.py`, `muninnx/cache/calibrate.py`
**Theory (bound).** Unrolling the step recursion gives ‖Δ_N‖ ≤ Σ_k L_k‖e_k‖, with
L_k = L'_k·∏_{j>k} K_j. Compute it backwards with a running product.

**Theory (calibration).** We want an upper bound U(s) on the reuse error ε that holds
with probability ≥ 1−α_step, without any distributional assumption:
1. Collect (s, ε) pairs along N_cal full chains for every step k and every staleness
   l = 1..max_lag: s = score(F_k, F_{k−l}) and ε = ‖v_k − v_{k−l}‖. The controller will
   reuse velocities that are several steps old, so calibration must see that too
   (see **Finding 1** below).
2. Fit m(s) = a·s + b by least squares on half of the pairs (`torch.linalg.lstsq`, or closed form).
3. On the other half, compute residuals r = ε − m(s). Then q is the ⌈(n+1)(1−α_step)⌉-th smallest residual.
4. U(s) = max(m(s) + q, 0).

**Why α_step = α/|T_cache|**: a union bound over all steps where reuse can happen.
With α = 0.05 and |T_cache| = 90, α_step ≈ 5.6e-4, so you need n ≥ 1/α_step ≈ 1800
held-out pairs or q becomes +∞. Pairs are pooled over steps, so 500 chains × 90 steps
is plenty. Bear this in mind if you later fit per-step bounds.

**Check.** `pytest -q tests/test_5_bound_calibration.py`. The coverage test checks
both validity (≥ 1−α) and tightness (not wildly conservative).

## Step 6 — Controller (Algorithm 1) · `muninnx/cache/controller.py`
**Theory.** Each sample has its own budget B_rem = η. At a cacheable step, reuse if the
predicted cost ĉ_k = Γ·L_k·U(s_k) fits in the remaining budget **and** the cached
velocity is less than `max_lag` steps old. On reuse, pay ĉ_k out of the budget.
Otherwise recompute and refresh the cache. If no failure event happens, the deviation
satisfies d ≤ Σ ĉ ≤ η.

**Hints**
- Keep per-sample tensors: `budget (B,)`, `v_cache (B, D)`, `f_cache (B, D_probe)`, `lag (B,)`, `nfe (B,)`.
- The score compares the current features with **`f_cache`, the features at the last
  recompute**, not with the previous step's features. That way s measures the actual
  drift since the cached velocity was computed.
- Compute the probe at **every** step, before deciding (as the paper does).
- At k = 0 the cache is empty, so always recompute. `cache_steps()` already excludes k = 0.
- Rule: recompute if `c_hat > budget`, and use the strict `>`. Note that if U(s) = 0
  then ĉ = 0 and reuse is free. Think about whether that is acceptable, because it
  can happen when m(s)+q < 0.
- Call the model only on the rows that recompute: `v_cache[mask] = model(x[mask], t)`.

**Check.** `pytest -q tests/test_6_controller.py`. The tests check four things:
η = 0 reproduces the full chain bit-exactly, an infinite budget reuses every cacheable
step, `max_lag` caps consecutive reuses, and the budget accounting is exact.

## Step 7 — First experiment (E0) · `scripts/run_muninn_toy.py` (already written)
```bash
python scripts/run_muninn_toy.py --ckpt runs/toy_8gaussians/model.pt
```
Output: `runs/toy_8gaussians/muninn/results.json` and `calibration_vs_step.png`.

**Questions to answer from E0**
1. **Validity**: is p_viol ≤ α (inside its Wilson CI) for every (η, α)? Is
   `bound_holds_frac` ≈ 1? If not, which assumption breaks? See "Open issues" below.
2. **Usefulness**: at the same mean NFE, does Muninn give a smaller E[d] than FewSteps?
   If FewSteps wins, that is a genuine finding for FM: think about why. Straight paths
   make the velocity almost constant along each path, so skipping steps and reusing
   velocity are nearly the same thing.
3. **Conservatism**: how far is the actual E[d] below η? The union bound plus the
   triangle-inequality accumulation can make the bound very loose.

---

## Why flow matching changes things (research notes)
- **The sensitivity weights become trivial.** For Euler on a uniform grid, K_k = 1 and
  L_k = Δ for every k, so the paper's "sensitivity-aware" weighting does nothing (in the
  paper it gave only 2.70× vs 2.63× anyway). Non-uniform grids (for example more steps
  near t=1) or a different quantity to reuse bring it back.
- **Reuse the predicted endpoint x̂₁ = x + (1−t)v instead of v.** The step becomes
  Φ(x, x̂₁) = x + Δ(x̂₁ − x)/(1−t), so K = 1 − Δ/(1−t) < 1 and the errors **contract**.
  This is a cheap ablation once v0 works.
- **Heun (M2).** Its second stage is evaluated at t+Δ, exactly where the next step's
  first stage is needed. Reusing it there (FSAL-style) is a natural FM-specific cache.
  The bound then involves the network's Lipschitz constant in x.

## Finding 1: the paper's rule read literally is not valid under chained reuse (already observed)
While validating the scaffold, I ran a private reference implementation on 8gaussians:
Euler with N=100, a model trained for only 3k iterations, and α = 0.05. With the paper's
rule read literally, the score compares consecutive steps (F_k vs F_{k−1}) and
calibration only ever sees 1-step-stale velocities. The controller then chains dozens
of reuses, each one looks cheap, and the cached velocity becomes very stale:

| variant | η | NFE | E[d] | p_viol |
|---|---|---|---|---|
| literal paper rule | 0.05 / 0.15 / 0.30 | 87 / 62 / 21 | 0.11 / 0.85 / 3.4 | **≈ 1.0** |
| staleness-aware, max_lag = 10 | 0.05 / 0.15 / 0.30 | 97 / 92 / 85 | 0.005 / 0.04 / 0.085 | ≤ 0.009 ✓ |
| FewSteps, Euler at the same NFE | | 97 / 92 / 85 | 0.003 / 0.003 / 0.009 | ≈ 0 |

So the design in this repo is staleness-aware: the score is measured against the
features at the last recompute, calibration covers lags 1..max_lag, and the controller
caps the lag. Rerun these numbers yourself once your implementation passes the tests.
Two follow-ups:
- (a) Reproduce the literal rule as an ablation: set `max_lag=1` in calibration and
  score against F_{k−1}.
- (b) **FewSteps currently wins clearly.** Getting an advantage over it is the real
  research question for FM. Ideas: per-step or per-lag bounds instead of one pooled
  regression (pooling makes U very conservative), reusing x̂₁ instead of v, and harder
  targets where the paths are curved (checkerboard, Maze2D).

## Open issues in the paper's argument (worth testing empirically)
1. The recursion bounds the error with ‖v(x_full) − ṽ‖, but calibration measures
   ‖v(x_cached) − ṽ‖. Closing that gap needs a Lipschitz constant L_v of the network
   in x. Log both quantities and compare.
2. Even with the lag-aware calibration, the pairs come from the **full** chain, while
   deployment runs on a slightly drifted chain. So the calibration distribution is
   still not exactly the deployment distribution.

## Next milestones (after v0)
- **M1**: Maze2D through Minari (`D4RL/pointmaze/*`), a Diffuser-style temporal U-Net
  trained with your FM loss, start/goal inpainting, and success rate measured by rollouts.
- **M2**: the Heun solver, with the two reuse variants above.
- **M3**: baselines (FixedSkip, ProbeThresh), then ablations: v vs x̂₁ reuse, time
  grids, probe depth, N_cal, forbidden regions.

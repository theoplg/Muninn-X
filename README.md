# Muninn-X

Certified, training-free caching for **flow-matching** samplers. This project adapts
*Muninn: Your Trajectory Diffusion Model But Faster* ([arXiv 2605.09999](https://arxiv.org/abs/2605.09999)).

Current stage: **v0**, a 2D toy with the Euler solver. Implementation steps are in [GUIDE.md](GUIDE.md).

```
muninnx/
  data/toy2d.py        toy targets (8gaussians, moons, checkerboard)
  models/mlp.py        time-conditioned velocity MLP (exposes .stem for the probe)
  fm/loss.py           flow-matching loss                   
  solvers/             Solver API + Euler                    
  sampling.py          full-compute sampling
  cache/               probe, score, bound, calibration, controller 
  eval/paired.py       paired metrics (d, p_viol, NFE)           
scripts/               train_toy.py, sample_toy.py, run_muninn_toy.py
tests/                 one test file per GUIDE step
```

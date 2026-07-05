# i3pw — Informed Inference of Inverse Probability Weights

Correcting **outcome-dependent selection (ascertainment) bias** by
**inverse-probability weighting (IPW)** when the population prevalences of the
outcomes are known a priori.

## The problem, and the idea

You have a biased sample — units were selected in a way that depends on their
outcomes (e.g. cases oversampled in a case-control or volunteer cohort), so
outcome prevalences in the sample are skewed, and so is everything estimated
from it. The standard fix models each unit's participation probability
`P(selected | X)` from covariates (socioeconomic features, via LASSO) and
reweights by `1 / P`.

**That participation model works poorly for many disease outcomes.** Write the
selection log-odds as `a(X) + θ·Y`: participation depends on *having the disease*
(`θ·Y`), a signal largely orthogonal to the covariates, so a covariate-only model
learns `a(X)` but not `θ·Y`, the propensities barely vary, and the weights barely
correct anything.

i3pw's idea: **use the known population prevalences to supply the missing `θ·Y`.**
Knowing `Pr(Y_q)` a priori (from a registry or census) is exactly the information
the covariate model lacks, and injecting it as a **calibration constraint** — force
the reweighted sample to reproduce the known prevalences — recovers the
disease-driven selection.

## Methods

| Method | Function | Idea |
| --- | --- | --- |
| No correction | `no_correction` | Naive prevalence in the observed sample. |
| LASSO IPW | `lasso_ipw` | Covariate-only participation model (`cv.glmnet` analogue) — *the approach that fails for disease outcomes*. |
| **Calibration IPW** | `calibration_ipw` | **Recommended.** Calibrate weights so the reweighted sample reproduces the known prevalences *exactly* (entropy balancing), optionally on top of the covariate model. |
| Penalized IPW | `penalized_ipw` | The original R project's softer precursor: a logistic inclusion model with a quadratic prevalence penalty, cross-validated `(λ, γ)`, numba-JIT compiled. |

### Calibration IPW (the principled version)

Given base weights `d_i` (uniform, or the covariate-model IPW weights), solve

```
min_w  Σ_i d_i · KL(w_i / d_i)
s.t.   Σ_i w_i Y_iq / Σ_i w_i = Pr(Y_q)   for each anchored outcome q
```

The solution is exponential tilting, `w_i ∝ d_i · exp(Σ_q λ_q Y_iq)`, with `λ` from
a small convex dual (entropy balancing; Hainmueller 2012, Deville & Särndal 1992).
Because that tilt is log-linear in `Y` — the same functional form as the selection
mechanism — calibrating on the `Q` known prevalences recovers the disease-driven
selection weights that a covariate model cannot. Using the covariate model for the
base weights `d_i` keeps the covariate-driven part too (a doubly-robust flavour).

`shrinkage=` adds a ridge on the tilt (exact calibration → shrink toward the base
weights, trading bias for variance); `calibration_ipw` reports the Kish **effective
sample size**, since strong ascertainment concentrates weight on few units.

### Penalized IPW (numba)

The softer precursor fits, for each outcome (with `p = sigmoid(Xβ)`):

```
f(β) = -mean( s·log(p) + (1-s)·log(1-p) )     # logistic negative log-likelihood
       + λ · ‖β‖₁                              # LASSO penalty
       + γ · (logit(mean(p)) - logit(π))²      # informed prevalence penalty
```

where `s` is the sample indicator and `π` the outcome's known population
prevalence. `γ = 0` recovers ordinary L1-penalized IPW; larger `γ` drags the
mean predicted inclusion probability toward `π`.

## Install

```bash
pip install -e .           # from a clone of this repo
pip install -e '.[test]'   # with the test dependencies
```

Requires Python ≥ 3.10 and numpy / scipy / scikit-learn / numba.

## Quick start

```python
import i3pw

# 1. Simulate a population and draw a biased sample.
ds = i3pw.make_dataset(
    seed=97,
    population_size=20000,
    n_features=15,
    n_outcomes=2,
    target_population_prevalence=(0.4, 0.05),
    target_sample_prevalence=(0.2, 0.005),   # what the biased sample looks like
    sample_size=4000,
)

print(ds.population_prevalence)   # truth we want to recover
print(ds.sample_prevalence)       # biased, naive estimate

# 2. Correct the bias.
naive = i3pw.no_correction(ds)
res   = i3pw.penalized_ipw(ds, lambdas=(0.001, 0.01, 0.1),
                           gammas=(0.0, 0.1, 1.0, 10.0), K=5)

print(naive.summary())
print(res["mean"].summary())      # weighted prevalence, per outcome
print("selected (λ, γ):", res["best_lambda"], res["best_gamma"])
```

Run the full benchmark comparison:

```bash
python examples/benchmark.py
```

Typical output (8k population, one common + one rare outcome; ~3.5s total incl.
one-time numba compile):

```
method                         % diff Y1   % diff Y2
----------------------------------------------------
no_correction                      48.25       91.32
lasso_ipw                          12.99       33.96
penalized_ipw[mean]                13.20       32.83
penalized_ipw[harmonic]            13.20       32.83
----------------------------------------------------
```

## Using the estimator directly

```python
from i3pw import PenalizedIPW

X_train, _, s_train = ds.split("train")
est = PenalizedIPW(lam=0.01, gamma=1.0, optimizer="gd")  # or "bfgs", "lbfgs"
est.fit(X_train, s_train, ds.population_prevalence)

X_test, Y_test, s_test = ds.split("test")
P = est.predict_inclusion(X_test)   # (n, Q) fitted inclusion probabilities
W = est.weights(X_test, s_test)     # (n, Q) per-outcome IPW weights
```

Three optimizers are available, mirroring the R scripts:

- `"gd"` — numba gradient descent with learning-rate decay (`Parallel_methods.R`);
- `"bfgs"` — SciPy BFGS (`BFGS.R`);
- `"lbfgs"` — SciPy L-BFGS-B with optional box `bounds` on coefficients (`L_BFGS.R`).

When several outcome models share one sample, their per-outcome weights are
combined with one of four rules (`combine=` in `penalized_ipw`): `"mean"`,
`"product"`, `"harmonic"`, `"absdiff"` (inverse-abs-difference weighting).

Two weighting schemes are available (`weighting=` in `lasso_ipw` / `penalized_ipw`,
or `scheme=` in `PenalizedIPW.weights`):

- `"odds"` — the R construction: selected units get `(1 - P) / P`, unselected units
  get weight 1, and the weighted mean runs over the whole test set.
- `"inverse"` — the textbook Horvitz–Thompson / Hájek estimator: selected units get
  `1 / P`, unselected units get weight 0, so only the **sample** is used.

Very large weights can be tamed with `trim=` (clip at a quantile, standard IPW practice).

## Why the covariate model fails and calibration works

Run `python examples/monte_carlo.py` — it repeats the whole pipeline over 20 random
populations and reports mean absolute % error (± SD) for each method, using the
deployable sample-only estimator:

```
method                     Y1 %err         Y2 %err
no_correction        46.81±5.70      79.53±6.49
lasso_ipw            44.66±5.93      78.50±6.61      <- covariate model, barely helps
calibration_ipw       0.00±0.00       0.00±0.00      <- uses the known prevalences
                                    (Kish effective sample size: 155 ± 32)
```

The covariate-only participation model (`lasso_ipw`) barely dents the bias, because
selection here is driven by the *outcomes* and the covariates are only a weak proxy
— exactly the situation that motivated the project. `calibration_ipw` reproduces the
known prevalences essentially exactly, because it is *given* them and enforces them
as constraints. That is the point: the known prevalences carry information the
covariate model cannot recover.

Two honest caveats:

- **The anchored outcomes are recovered by construction.** The value is not that
  calibration "predicts" a prevalence it was told, but that it produces *weights*
  that are correct along the ascertained dimensions — which then de-bias downstream
  estimands (associations, coefficients) that are correlated with those outcomes.
- **Transfer is not automatic.** Calibrating on outcome A helps estimands correlated
  with A; for a target driven by factors independent of the anchored outcomes there
  is little to transfer. And the correction costs variance — `calibration_ipw`
  reports the Kish effective sample size, which shrinks as ascertainment strengthens.
- **Feasibility.** A target prevalence is reachable only if the ascertained sample
  contains cases of that outcome. For a very rare outcome in a small sample the
  cases can be absent, and no reweighting reaches the target; `shrinkage=` (or
  pooling outcomes) helps degrade gracefully.

### Weighting schemes for the IPW baselines

`lasso_ipw` / `penalized_ipw` accept `weighting=`, and `PenalizedIPW.weights` accepts
`scheme=`:

- `"inverse"` — the textbook Horvitz–Thompson / Hájek estimator (`1 / P`, sample only).
  **Deployable**, and the default lens for honest evaluation.
- `"odds"` — the original R construction (`(1 - P) / P` for selected, weight 1 for
  unselected, mean over the whole test set). It reads unselected outcomes, so it
  flatters the method; treat it as an oracle diagnostic.

Very large weights can be tamed with `trim=` (clip at a quantile, standard IPW practice).

## Notable differences from the R code

These are deliberate corrections/improvements, documented so results are
comparable:

- **Intercept in the inclusion model.** The R gradient methods omitted an
  intercept, which crippled the prevalence penalty (with mean-zero covariates
  there is no lever to shift the mean prediction). `PenalizedIPW` fits an
  unpenalized intercept by default (`fit_intercept=True`).
- **Exact penalty gradient.** The gradient of the prevalence penalty is
  implemented in closed form and verified against numerical differentiation
  (`scipy.optimize.check_grad`, error < 1e-7).
- **Cross-validation criterion.** The R code selects `(λ, γ)` by minimizing the
  *penalized* objective, whose scale grows with `γ` — so it structurally favors
  `γ = 0`. `penalized_ipw` defaults to `cv_criterion="prevalence"`, scoring each
  fold by how well the reweighted validation prevalence recovers the known
  population value, which makes the informed penalty genuinely selectable. The
  faithful `cv_criterion="objective"` is still available.

## Package layout

```
src/i3pw/
├── dgm.py          # data-generating mechanism + biased sampling
├── calibration.py  # calibration_ipw + entropy_balance (the recommended method)
├── methods.py      # no_correction, lasso_ipw / lasso_propensity, penalized_ipw
├── penalized.py    # PenalizedIPW estimator (gd / bfgs / lbfgs)
├── _kernels.py     # numba-compiled objective, gradient, gradient descent
├── weights.py      # per-outcome weight combination rules
├── evaluation.py   # Monte Carlo comparison across many replications
├── metrics.py      # weighted prevalence, % difference, weighted MSE
└── _links.py       # stable sigmoid / logit
tests/              # pytest suite (calibration, gradient checks, DGM, methods)
examples/           # benchmark.py, monte_carlo.py
```

## Calibration in one snippet

```python
import i3pw

ds = i3pw.make_dataset(seed=0, n_outcomes=2)

# Covariate model alone barely corrects an outcome-driven selection...
print(i3pw.lasso_ipw(ds, weighting="inverse").summary())

# ...so inject the known population prevalences as calibration constraints.
res = i3pw.calibration_ipw(ds, base="lasso")   # base weights from the covariate model
print(res.summary())
print("effective sample size:", round(res.ess))

# Anchor only the diseases whose prevalence you actually know:
res = i3pw.calibration_ipw(ds, anchor_outcomes=[0], base="lasso", shrinkage=0.0)
```

## Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).

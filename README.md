# i3pw — Informed Inference of Inverse Probability Weights

A Python reimplementation of the key methods from the `SelectionBias` R project:
simulating **outcome-dependent selection bias** and correcting it with
**inverse-probability weighting (IPW)**, including a novel *prevalence-penalized*
IPW estimator that uses known population prevalences to inform the weights.

The compute-heavy inner loops (penalized objective, exact gradient, and
gradient descent) are JIT-compiled with [numba](https://numba.pydata.org/).
Compilation happens once per environment and is cached to disk; call
`i3pw.warmup()` to pay that one-time cost up front.

## The problem

You have a biased sample from a population — units were selected in a way that
depends on their outcomes, so outcome prevalences in the sample are skewed. You
want to recover the **population** prevalence of each outcome.

IPW corrects this by modelling each unit's probability of inclusion,
`P(selected | X)`, and reweighting sampled units by the inverse odds
`(1 - P) / P`. The **informed** twist: when the population prevalence of an
outcome is known (e.g. from a registry or census), add a penalty that pulls the
inclusion model's average prediction toward that known value.

## Methods

| Method | Function | Idea |
| --- | --- | --- |
| No correction | `no_correction` | Naive prevalence in the observed sample (baseline). |
| LASSO IPW | `lasso_ipw` | One L1-penalized logistic inclusion model (the `cv.glmnet` analogue), one weight per unit. |
| **Penalized IPW** | `penalized_ipw` | Per-outcome inclusion models with an L1 penalty **and** an informed prevalence penalty; cross-validated `(λ, γ)`. |

The penalized objective fit for each outcome (with `p = sigmoid(Xβ)`):

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

## Does the correction actually work? A caveat worth reading

Run `python examples/monte_carlo.py` — it repeats the whole pipeline over 20 random
populations and reports mean absolute % error (± SD) for each method under both
weighting schemes:

```
weighting = 'odds'     (oracle — reads unselected outcomes)
method                     Y1 %err         Y2 %err
no_correction        46.76±4.83      87.49±6.69
lasso_ipw            14.25±2.98      29.37±4.46
penalized_ipw        14.12±2.73      27.57±4.76

weighting = 'inverse'  (deployable — sample only)
method                     Y1 %err         Y2 %err
no_correction        46.76±4.83      87.49±6.69
lasso_ipw            44.28±5.07      87.28±6.62
penalized_ipw        45.61±4.86      87.41±6.67
```

Under the **`odds`** scheme the methods look excellent — but that weighted mean
includes unselected units, contributing their outcomes, which you would never
observe in a real study. Under the **`inverse`** scheme (the deployable estimator
that uses only the sampled units) the correction almost vanishes.

This is not a bug; it is the fundamental limitation of IPW here. Selection in this
DGM is driven directly by the *outcomes*, and the covariates `X` are only a weak
proxy for them. Inverse-probability weighting on `X` can only remove the part of the
selection that `X` explains — it cannot recover prevalence when units are selected on
the outcome itself (a missing-not-at-random problem). The headline numbers reported
by the original R scripts rely on the `odds` construction, so they flatter the method.
Treat `odds` as an oracle diagnostic and `inverse` as the honest estimate.

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
├── dgm.py         # data-generating mechanism + biased sampling
├── penalized.py   # PenalizedIPW estimator (gd / bfgs / lbfgs)
├── _kernels.py    # numba-compiled objective, gradient, gradient descent
├── methods.py     # no_correction, lasso_ipw, penalized_ipw, cross_validate
├── weights.py     # per-outcome weight combination rules
├── evaluation.py  # Monte Carlo comparison across many replications
├── metrics.py     # weighted prevalence, % difference, weighted MSE
└── _links.py      # stable sigmoid / logit
tests/             # pytest suite (gradient checks, DGM, methods, Monte Carlo)
examples/          # benchmark.py, monte_carlo.py
```

## Tests

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).

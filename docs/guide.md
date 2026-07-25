# i3pw — user guide

The estimators, how to [check a weighting](#checking-the-weights-balance-as-a-falsification-test),
and how to [put error bars on it](#uncertainty).

The recipe these support is in the
[README](../README.md#conclusions-and-recommendations), and the menu of what you might
calibrate on — from a single prevalence up to comorbidity and severity — is the
[ladder](theory.md#a-ladder-of-prevalence-informed-weights) in `theory.md`, which is worth
reading first if you have a real cohort and are deciding what to constrain. Justification
for any of it is in [theory.md](theory.md); sources are in its
[bibliography](theory.md#references).


## Methods

| Method | Function | Idea |
| --- | --- | --- |
| No correction | `no_correction` | Naive prevalence in the observed sample. |
| LASSO IPW | `lasso_ipw` | Covariate-only participation model (`cv.glmnet` analogue) — *the approach that fails for disease outcomes*. |
| **Calibration IPW** | `calibration_ipw` | **The method.** Calibrate weights so the reweighted sample reproduces the known prevalences *exactly* (entropy balancing), optionally on top of the covariate model. |

### Calibration IPW (the principled version)

Given base weights `d_i` (uniform, or the covariate-model IPW weights), solve

```
min_w  Σ_i d_i · KL(w_i / d_i)
s.t.   Σ_i w_i Y_iq / Σ_i w_i = Pr(Y_q)   for each anchored outcome q
```

The solution is exponential tilting, `w_i ∝ d_i · exp(Σ_q λ_q Y_iq)`, with `λ` from
a small convex dual (entropy balancing; Hainmueller 2012, Deville & Särndal 1992).
Because that tilt is log-linear in `Y`, calibrating on the `Q` known prevalences supplies
the disease-driven part of the reweighting a covariate model cannot — and it *equals* the
true inverse-probability weights when the population-to-sample density ratio lies in this
tilt family (base `d(X)` for the covariate-driven part, `exp(λ·Y)` for the outcome-driven
part; see [What is identified?](theory.md#what-is-identified)). Otherwise it is the
minimum-divergence weighting that matches the known moments. (This is **not** doubly
robust in the AIPW sense; rather, it is consistent when the base weights capture the
covariate-driven part of selection *and* the calibration functions span the
remaining outcome-driven part — two ingredients covering different pieces.)

`shrinkage=` adds a ridge on the tilt (exact calibration → shrink toward the base
weights, trading bias for variance). `calibration_ipw` returns **diagnostics**
(`res.diagnostics_summary()`): optimizer convergence, the max calibration residual
(non-zero flags an infeasible target — e.g. an anchored outcome with no cases sampled),
per-anchor case/control support, the Kish **effective sample size**, and how much weight
the top 1% of units carry. It warns (`CalibrationWarning`) when the solve fails to
converge, a target is unreachable, or `trim=` breaks the exact calibration.

## Checking the weights: balance as a falsification test

Every diagnostic in the section above describes the **solve** — did it converge, did it
hit the targets, how concentrated are the weights. None of them can tell you the weights
are *wrong*, because a calibration matches its constraints exactly whether or not the
underlying density-ratio model is right. A broken run can even look healthier: a base
model that misses a real participation driver can leave the residual at machine precision
with a **larger** effective sample size than the correct one.

`balance_report` supplies the missing test. Check the reweighted sample against
population quantities the calibration **did not** use:

```python
rep = i3pw.balance_report(
    features,                       # (n, k) quantities on the sampled units
    weights,                        # calibration weights
    population_means,               # what they should average to
    constrained=[True, False, ...], # which ones were calibration targets
)
print(rep.summary())
print(rep.worst_held_out, rep.passed())   # verdict uses ONLY the held-out columns
```

Constrained moments match by construction and carry no information — they are excluded
from the verdict. *Unconstrained* ones do not have to match, so when they do the model
has survived a test it could have failed. This is an **overidentification test**: supply
more known population quantities than you calibrate on, and the surplus becomes evidence.
For a biobank, calibrate to known disease prevalences and check against register margins
you held back (age, sex, region). A large held-out `|SMD|` refutes the tilt-family
assumption that [What is identified?](theory.md#what-is-identified) rests on.

Measured on a population where participation depends on a covariate `X0` *and* the
outcome, with the prevalence anchored either way:

| | converged | residual | ESS | worst held-out \|SMD\| |
| --- | --- | --- | --- | --- |
| base model sees `X0` | ✔ | 4e-11 | 3165 | **0.17** |
| base model misses `X0` | ✔ | 6e-11 | **10405** | **1.11** |

Every shipped diagnostic prefers the broken run — its ESS is 3× "healthier". Only the
held-out covariate separates them.

## Uncertainty

Point estimates and the ESS are not enough. `i3pw.uncertainty` adds three pieces:

- `weighted_mean_se(values, weights)` — the design-based linearization (sandwich) SE of a
  Hájek weighted mean or prevalence, `Var = Σ wᵢ²(yᵢ−μ)² / (Σ wᵢ)²`. Exact for
  independent units, but it treats the weights as *fixed*, so it does not describe the
  uncertainty of a calibration estimate — and it is **not a bound in either direction**.
  On an anchored margin it badly *overstates* (calibration reproduces the known
  prevalence exactly, so the true sampling variability is zero while the formula still
  returns ≈0.04); on an estimand uncorrelated with the anchors it is about right; it can
  understate when the weights are noisy. Use the bootstrap when the weights were
  estimated.
- `calibration_mean_se(values, weights, features)` — the SE that **does** account for the
  estimated calibration, in closed form. The calibration estimator is asymptotically a
  regression (GREG) estimator, so its influence function is the residual of the outcome
  on the constrained functions, `e = y − μ − β·(g − ḡ)`. Constraining `g` removes exactly
  the part of `y` that `g` explains, which gets the anchored case right for the right
  reason: an anchored outcome *is* a column of `g`, so its residual is identically zero
  and the SE collapses to 0. Validated against the bootstrap on a held-out estimand
  (0.0759 closed-form vs 0.0785 from 400 replicates) — same answer, no re-solving.
- `apply_tilt(features, tilt, targets)` — the fitted dual `λ` is exposed on
  `CalibrationDiagnostics.tilt`; it *is* the estimated `θ` of the selection decomposition
  `a(X) + θ·g(Y)` ([what that means](theory.md#what-is-identified), and for one binary
  outcome it is just the log odds-ratio between register and sample). Feed it
  here to weight a held-out fold or newly recruited participants under the same
  calibration without re-solving. Transferred weights are *not* re-calibrated, so their
  achieved moments miss the targets by ordinary sampling error — which is what makes this
  a usable check rather than a tautology.
- `bootstrap_calibration_ipw(dataset, ...)` — a nonparametric bootstrap over the sampled
  units that re-solves the calibration each replicate, so it captures the
  weight-*estimation* variability the linearization SE omits; `refit_base=True` also
  refits the LASSO participation model per replicate. Anchored outcomes come back with
  near-zero SE **by construction** — the honest read is that, conditional on the known
  prevalences, the anchored margins carry no sampling uncertainty; the variance lives in
  the *unanchored* and downstream estimands:

  ```
  bootstrap (100 reps, 95% percentile CI):
    Y1: 0.4085 ± 0.0000 [0.4085, 0.4085] (anchored)
    Y2: 0.0312 ± 0.0096 [0.0151, 0.0508]
  ```

- `prevalence_sensitivity(dataset, ...)` — registry prevalences are not exact constants
  (age/period, ascertainment, diagnostic, linkage error), so this scales the known `K`
  by `1 + δ` across a grid and reports how each estimand and the ESS move. The anchored
  outcome tracks its perturbed target by construction; the informative response is in the
  unanchored outcomes and the ESS.

## Downstream estimands: doubly-robust estimation

Calibration fixes the *ascertained outcome*, which is not otherwise identified.
But most analyses target a **downstream** quantity — the population mean of a
trait or biomarker measured only on participants. When that is missing at random
given the covariates (`S ⊥ V | X`), it is recoverable, and the efficient, robust
estimator is augmented IPW (`aipw_mean`):

```
μ_AIPW = mean_i m(X_i)  +  Σ_{i in sample} w_i (V_i − m(X_i))
```

with an outcome model `m(X) = E[V|X]` fit on the sample and self-normalized
weights `w` (from a participation model *or* from `calibration_ipw`). It is
**doubly robust** — consistent if *either* `m` or `w` is correct — and lower
variance than weighting alone.

The doubly-robust guarantee is conditional: it needs the MAR structure `S ⊥ V | X`,
and — for the usual √n inference with a *flexible* outcome model — the fit to be
independent of the point it scores. Fitting `m` on the whole sample and predicting
in-sample (the default) is fine for the simple models here, but for ML outcome models
pass `aipw_mean(..., crossfit=K)`: it fits `m` out-of-fold (Chernozhukov et al. 2018),
so each unit's residual comes from a model that never saw it. `crossfit=1` keeps the
exact in-sample behaviour.

### A doubly-robust demo (`examples/doubly_robust_trait.py`)

A binary outcome is ascertained (cases over-represented; population prevalence `K`
known); a trait `V` — a biomarker, say — is measured only on participants and
correlates with the outcome's liability, so the sample's mean `V` is inflated.
Recovering `E[V]` over 20 replications (bias from the truth, `|bias|`):

```
method          mean bias    |bias|
naive             -0.096      0.101     <- ascertainment inflates the trait
ipw_lasso         -0.019      0.065
calibration       +0.084      0.103     <- weights tuned to the ascertained margin, noisy here
aipw              +0.003      0.050     <- doubly robust: best and most stable
```

Two honest lessons: (1) `calibration_ipw`'s job is the ascertained margin —
using its weights as a raw weighted mean for an *unrelated* quantity can be
noisy; `aipw` is the right downstream estimator. (2) Pure case-control
ascertainment leaves *logistic slopes* unbiased (Prentice & Pyke 1979) but biases
means, absolute risks, and liability-correlated traits — which is what these
estimators repair.

## Stratified calibration: prevalence known within strata

A single pooled prevalence is often too crude. In registers and biobanks, prevalence
varies strongly by **sex, birth cohort, age, ancestry, region, or calendar time**, and
participation varies across those same strata. When the registry reports prevalence
*within* strata, calibrate to it directly rather than to the pooled margin.
`stratified_calibration_weights(Y, strata, within_stratum_prevalence, stratum_share)`
matches, for every stratum `a` and outcome `q`,

```
E_w[1(A = a)]        = P(A = a)              (stratum shares)
E_w[Y_q · 1(A = a)]  = P(Y_q = 1, A = a)     (within-stratum prevalence)
```

so the reweighted sample reproduces both the stratum sizes and the per-stratum
prevalences. This matters whenever selection acts *through* the strata: pooled outcome
calibration reweights as a function of the outcome only, so it cannot restore a distorted
stratum composition, and any estimand that depends on the strata (not just on the anchored
outcome) stays biased. Calibrating within strata pins the composition and recovers it —
and, being calibration to richer moments, it also reaches past purely marginal selection
toward the interaction structure marginal calibration cannot represent. It reduces to
`outcome_calibration_weights` when there is a single stratum, and composes with covariate
base weights exactly like the other calibrators.

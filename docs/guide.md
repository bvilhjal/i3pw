# i3pw — user guide

If you have a real cohort, [`calibrate`](#if-you-have-a-cohort-start-here-calibrate) is
the entry point and the rest of this page is background. Then: the estimators, how to
[check a weighting](#checking-the-weights-a-held-out-balance-diagnostic), and how to
[put error bars on it](#uncertainty).

The recipe these support is in the
[README](../README.md#conclusions-and-recommendations), and the menu of what you might
calibrate on — from a single prevalence up to comorbidity and severity — is the
[ladder](theory.md#a-ladder-of-prevalence-informed-weights) in `theory.md`, which is worth
reading first if you have a real cohort and are deciding what to constrain. Justification
for any of it is in [theory.md](theory.md); the symbols are defined once in its
[notation table](theory.md#notation); sources are in its
[bibliography](theory.md#references).


## If you have a cohort, start here: `calibrate`

Everything else on this page takes a `Dataset` — a *simulated* population carrying
ground-truth coefficients, with its population prevalence computed from outcomes observed
on everybody. Your cohort has none of that, and does not need it. `calibrate` takes what
you actually have:

```python
import i3pw

fit = i3pw.calibrate(
    Y=case_status[:, None],              # outcomes for participants only
    targets=[0.012],                     # register prevalence
    base_weights=i3pw.inverse_probability_weights(p_hat),   # your participation model
    holdout={"mean age": (age, 41.7),    # register margins you did NOT calibrate on
             "% female": (female, 0.508)},
)

print(fit.summary())            # solve diagnostics + held-out balance diagnostic
print(fit.mean(bmi).summary())  # a weighted mean, with an SE that knows about the tilt
fit.weights                     # if you would rather take the weights and go
```

Four arguments carry the recipe, and it is worth knowing which does what.

**`base_weights=`** is the covariate-driven half of the correction, and it comes from your
participation model — any model, fitted however you like on whatever sampling frame you
have. Pass `1/P̂`, or hand the predicted probabilities to `inverse_probability_weights`
(the `scheme="odds"` option, `(1−P)/P`, targets nonparticipants rather than the full
population and is appropriate only when that is the stated target). Omit it and you get
pure calibration from uniform weights: correct when selection
acts only through the outcome, silent about the covariate channel when it does not.

**`targets=`** are the known population prevalences — the outcome-driven half.

**`strata=`** calibrates prevalence *within* sex, birth year, ancestry or severity instead
of only the pooled margin, and pins the stratum shares as well. Pass integer labels plus
`stratum_share`, and give `targets` as the `(A, Q)` within-stratum prevalences. A known
prevalence fixes the *number* of cases, not their *type*
([case mix](theory.md#prevalence-sets-the-scale-not-the-case-mix)); for psychiatric
cohorts this is usually the step that matters most.

**`holdout=`** is the only part of the output that can tell you the weighting is wrong.
Name quantities whose population value you know and are deliberately *not* constraining;
they become a [balance report](#checking-the-weights-a-held-out-balance-diagnostic)
whose verdict ignores the constrained columns. Leave it out and `fit.summary()` will say,
in as many words, that nothing was tested — which is not the same as a pass.

`fit.mean(values)` gives a weighted mean whose standard error accounts for the estimated
tilt (an exactly constrained margin gets approximately zero only when `shrinkage=0`; see
[Uncertainty](#uncertainty)), and
`fit.apply_to(...)` re-weights fresh rows under the already-fitted tilt without re-solving.

### The recipe, run end to end

`examples/real_cohort_workflow.py` walks the whole thing on a population where
participation depends on the disease *and* on age, so each half of the correction has
something only it can fix. Scored on a trait no weighting was given (truth 25.112):

| weighting | disease | held-out age | trait mean | ESS |
| --- | --- | --- | --- | --- |
| naive | 0.1738 | 51.02 | 25.650 | 45099 |
| participation model only | 0.1615 | 45.03 | 25.322 | 34139 |
| calibration only | 0.0519 | 50.81 | 25.389 | 40865 |
| **both** | 0.0519 | **44.79** | **25.083** | 30861 |
| *truth* | *0.0519* | *44.97* | *25.112* | |

The disease column is exact for anything calibrated — it was the constraint — and is not
evidence of anything. The model alone corrects age and leaves the disease at 0.16; the
calibration alone corrects the disease and leaves age at 50.8. Only the combination gets
both, and it cuts the trait error from 0.54 to 0.03. The cost is visible in the last
column: the ESS falls from 45k to 31k, which is what the correction is paying.

## Methods (the simulation harness)

These take a `Dataset` from `make_dataset` and exist to compare estimators against a known
truth. `calibration_ipw` is a thin wrapper over `calibrate` — same estimator, different
front door.

| Method | Function | Idea |
| --- | --- | --- |
| No correction | `no_correction` | Naive prevalence in the observed sample. |
| LASSO IPW | `lasso_ipw` | Covariate-only participation model (`cv.glmnet` analogue) — *the approach that fails for disease outcomes*. |
| **Calibration IPW** | `calibration_ipw` | **The method.** Calibrate weights so the reweighted sample reproduces the known prevalences *exactly* (entropy balancing), optionally on top of the covariate model. |

### What the solve actually does

Given base weights `d_i` (uniform, or the covariate-model IPW weights), find the weights
that stay as close as possible to `d` while hitting the known prevalences:

```
min_w   Σ_i w_i log(w_i / d_i)
s.t.    (Σ_i w_i Y_iq) / (Σ_i w_i) = P(Y_q)     for each anchored outcome q
```

The objective is the Kullback–Leibler divergence of `w` from `d`; the solution is
exponential tilting,

```
w_i  ∝  d_i · exp( Σ_q λ_q Y_iq )
```

with `λ` from a small convex dual — one parameter per constraint, not per unit
(entropy balancing; Hainmueller 2012, Deville & Särndal 1992; the
[dual, written out](theory.md#where-this-sits-density-ratios-i-projection-and-label-shift)).
Because that tilt is log-linear in `Y`, calibrating on the `Q` known prevalences supplies
the disease-driven part of the reweighting a covariate model cannot.

What that buys and what it assumes — when these weights equal the true inverse-probability
weights, why they are *not* doubly robust in the AIPW sense, and what has to be true
instead — is [What is identified?](theory.md#what-is-identified). That section is the
canonical statement; this page does not restate it.

`shrinkage=` adds a ridge on the tilt (exact calibration → shrink toward the base
weights, trading bias for variance). `calibration_ipw` returns **diagnostics**
(`res.diagnostics_summary()`): optimizer convergence, the max calibration residual
(non-zero flags an infeasible target — e.g. an anchored outcome with no cases sampled),
per-anchor case/control support, the Kish **effective sample size**, and how much weight
the top 1% of units carry. It warns (`CalibrationWarning`) when the solve fails to
converge, a target is unreachable, `trim=` breaks the exact calibration, or the design
asks for more constraints than the sample can support (fewer than ten units per
constraint — the trap a fine-grained `strata=` walks into, since `A` strata by `Q`
outcomes grows much faster than the cells backing them).

## Checking the weights: a held-out balance diagnostic

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

If you got your weights from [`calibrate`](#if-you-have-a-cohort-start-here-calibrate),
pass `holdout={name: (values, population_mean)}` instead and skip the assembly: the
constrained columns and their flags are already known, so the report comes back on
`fit.balance` with the right ones excluded from the verdict. This is the same function,
called for you.

Constrained moments match by construction and carry no information — they are excluded
from the verdict. *Unconstrained* ones do not have to match, so they can expose a bad
weighting. This is a held-out specification diagnostic inspired by overidentifying
restrictions, not a formal Sargan/Hansen J-test: `balance_report` has no
covariance-weighted statistic, reference distribution, degrees of freedom, or p-value.
For a biobank, calibrate to known disease prevalences and check against register margins
you held back (age, sex, region). A large held-out `|SMD|` is evidence against the
tilt-family assumption that [What is identified?](theory.md#what-is-identified) rests on;
a small one is not proof.

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
  Hájek weighted mean or prevalence,
  `Var = Σ w_i²(y_i − μ)² / (Σ w_i)²`. This independent-unit linearization
  treats the weights as *fixed*, so it does not describe the
  uncertainty of a calibration estimate — and it is **not a bound in either direction**.
  On an anchored margin it badly *overstates* (calibration reproduces the known
  prevalence exactly, so the true sampling variability is zero while the formula still
  returns ≈0.04); on an estimand uncorrelated with the anchors it is about right; it can
  understate when the weights are noisy. Use the bootstrap when the weights were
  estimated.
- `calibration_mean_se(values, weights, features, ridge=...)` — a first-order SE that
  accounts for estimation of the calibration tilt. For exact calibration its influence
  function uses the GREG residual
  `e_i = y_i − μ − β·(g_i − ḡ)`, with `β = Cov_w(g)^{-1}Cov_w(g,y)`.
  For a penalized solve it instead uses
  `β_ρ = [Cov_w(g)+ρI]^{-1}Cov_w(g,y)`. Thus an anchored outcome has zero residual
  only under exact feasible calibration (`ρ=0`); with shrinkage the target is not pinned
  and its SE is generally nonzero.
  `fit.mean(values)` on a [`calibrate`](#if-you-have-a-cohort-start-here-calibrate) result
  carries the fitted ridge and constraints into this calculation.
- `apply_tilt(features, tilt, targets)` — the fitted dual `λ` is exposed on
  `CalibrationDiagnostics.tilt`; it is the outcome coefficient in the fitted
  population-to-participant log density ratio, not generally a participation-logit
  coefficient. For one binary outcome it is the log odds-ratio between register and
  sample. Feed it
  here to weight a held-out fold or newly recruited participants under the same
  calibration without re-solving. Transferred weights are *not* re-calibrated, so their
  achieved moments miss the targets by ordinary sampling error — which is what makes this
  a usable check rather than a tautology.
- `bootstrap_calibration_ipw(dataset, ...)` — a nonparametric bootstrap over the sampled
  units that re-solves the calibration each replicate, so it captures the
  weight-*estimation* variability the fixed-weight SE omits; `refit_base=True` also
  refits the LASSO participation model per replicate. Under exact feasible calibration,
  anchored outcomes come back with near-zero SE **by construction**. With
  `shrinkage>0` their achieved margins and bootstrap estimates vary:

  ```
  bootstrap (100 reps, 95% percentile CI):
    Y1: 0.4085 ± 0.0000 [0.4085, 0.4085] (anchored)
    Y2: 0.0312 ± 0.0096 [0.0151, 0.0508]
  ```

  **Read `failure_rate` before quoting the interval.** A replicate that draws no cases of
  a rare anchored outcome cannot meet its target and is discarded rather than folded in
  (one infeasible constraint corrupts the other outcomes' estimates too). But that rule
  fires on exactly the resamples poorest in rare cases — the ones that would have
  populated the tail — so a non-zero failure rate means the printed interval is **too
  narrow**, and by more the higher the rate. It is a truncation the bootstrap cannot
  correct for itself, only report. Treat the interval as a lower bound on the uncertainty,
  or remove the cause: drop the rare anchor, or set `shrinkage > 0` so replicates stop
  failing.

- `prevalence_sensitivity(dataset, ...)` — a simulation-harness helper for a `Dataset`;
  it is not a general-array companion to `calibrate`. Registry prevalences are not exact constants
  (age/period, ascertainment, diagnostic, linkage error), so this scales the known `K`
  by `1 + δ` across a grid and reports how each estimand and the ESS move. The anchored
  outcome tracks its perturbed target by construction; the informative response is in the
  unanchored outcomes and the ESS.

## Downstream estimands: doubly-robust estimation

Calibration fixes the *ascertained outcome*, which is not otherwise identified.
But most analyses target a **downstream** quantity — the population mean of a
trait or biomarker measured only on participants. When that is missing at random
given the covariates (`S ⊥ V | X`), it can be recovered with augmented IPW
(`aipw_mean`):

```
μ̂_AIPW  =  (1/N) Σ_i m(X_i)  +  Σ_{i : S_i = 1} w_i ( V_i − m(X_i) )
              over the whole frame        over participants only
```

with an outcome model `m(X) = E[V | X]` fit on the sample and self-normalized
weights `w` (from a participation model *or* from `calibration_ipw`); the first sum runs
over everyone in the frame, the second only over participants. It is
**doubly robust** — consistent if either `m` is a correct outcome regression or `w` is
proportional to the correct full-population density ratio, under the sampling-frame and
regularity conditions of the AIPW theory. This is a *downstream* estimator for a
participant-only trait when `X` is known on the frame. It does not repair a misspecified
tilt family: if `g(Y)` misses the outcome-driven part of selection, the weighting piece
and the outcome-regression piece can fail together. Calibration to a few prevalences does
not by itself make `w` correct, and lower variance than weighting alone is not guaranteed
in every finite sample.

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
Recovering `E[V]` over 20 replications (bias from the truth, `|bias|`), copied
from the `doubly_robust_trait.py` rows of
[`report/validation_results.tsv`](../report/validation_results.tsv):

```
method          mean bias    |bias|
naive             -0.0905     0.1008    <- ascertainment inflates the trait
ipw_lasso         -0.0252     0.0636
calibration       +0.0537     0.0864    <- weights tuned to the ascertained margin, noisy here
aipw              -0.0009     0.0584    <- best mean bias in this design, not a ranking
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
E_w[ 1(A = a) ]        =  P(A = a)             (stratum shares)
E_w[ Y_q · 1(A = a) ]  =  P(Y_q = 1, A = a)    (within-stratum prevalence)
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

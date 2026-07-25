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
the reweighted sample to reproduce the known prevalences — supplies the
disease-driven part of the reweighting that the covariate model cannot. What that
does and does not identify is made precise in [What is identified?](#what-is-identified)
below: calibration recovers *minimum-divergence* weights matching the known moments,
which coincide with the true inverse-probability weights under a stated condition.

## Install

```bash
pip install -e .           # from a clone of this repo
pip install -e '.[test]'   # with the test dependencies
```

Requires Python ≥ 3.10 and numpy / scipy / scikit-learn.

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

# 2. Correct the bias by calibrating the weights to the known prevalences.
naive = i3pw.no_correction(ds)
cal   = i3pw.calibration_ipw(ds, base="lasso")   # the recommended estimator

print(naive.summary())
print(cal.summary())              # weighted prevalence, per outcome
print("effective sample size:", round(cal.ess))

# Anchor only the diseases whose prevalence you actually know; the rest are left
# free, so evaluating them measures transfer rather than restating an input.
cal = i3pw.calibration_ipw(ds, anchor_outcomes=[0], base="lasso")

# Check the weighting against population quantities it was NOT given — the only
# way it can fail, and so the only real evidence it worked.
X, Y, s = ds.split("test")
sel = s == 1
print(i3pw.balance_report(
    X[sel][:, :3], cal.extra["weight"][sel], X[:, :3].mean(axis=0),
).summary())
```

Run the full benchmark comparison:

```bash
python examples/benchmark.py
```

Typical output (8k population, one common + one rare outcome; ~2s total):

```
method                       % diff Y1   % diff Y2
--------------------------------------------------
no_correction                    48.25       91.32
lasso_ipw                        45.64       91.44
calibration_ipw                   0.00        0.00
--------------------------------------------------
```

**Read those two zeros carefully.** Every outcome here is anchored, so `calibration_ipw`
is reproducing prevalences it was *handed* — an algebraic identity, not a result. And
`lasso_ipw` looks useless because this simulation gives selection no covariate channel to
learn. `examples/honest_benchmark.py` measures the questions that can actually fail —
transfer to an *unanchored* outcome, and balance against quantities never supplied — and
[its section](#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy) is
the one to trust.

## How to read the rest of this

What follows is long. It contains four *kinds* of material, somewhat interleaved, and
most readers need only two of them. Follow the links rather than reading straight through.

1. **Theory** — [what is identified](#what-is-identified), [where the construction comes
   from](#where-this-sits-density-ratios-i-projection-and-label-shift), [why the standard
   errors behave as they do](#calibration-is-a-regression-estimator-and-why-the-ses-look-the-way-they-do),
   and [what could prove the whole thing wrong](#what-makes-this-falsifiable). Read this
   before quoting a number from the package in a paper. It is also the part that says, at
   some length, what the method *cannot* do.
2. **Practice** — the [two tasks](#two-separable-tasks-predict-selection-then-anchor-to-the-population)
   the package deliberately keeps apart, a [ladder of weights](#a-ladder-of-prevalence-informed-weights)
   from crude to defensible, and how to [check](#checking-the-weights-balance-as-a-falsification-test)
   and [put error bars on](#uncertainty) the result. If you have a real cohort and want a
   recipe, start at the ladder and ignore everything else.
3. **Evidence** — simulation studies, beginning with the
   [one that tries hardest to make the method look bad](#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy).
   Skippable unless you doubt a specific claim, in which case there is probably a table for it.
4. **Reference** — [layout](#package-layout), [tests](#tests), [bibliography](#references).

In a hurry? [**Conclusions and recommendations**](#conclusions-and-recommendations)
condenses the whole document into a five-step recipe, a list of things not to do, and the
reason for each — with links back to the evidence, so you can check any of it.

A warning about reading order that applies throughout: several sections report a `0.00`
or an exact match, and *every one of those is an identity rather than a finding*. The
interesting numbers in this document are all elsewhere.

## What is identified?

### A case small enough to check by hand

Before the general statement, the simplest instance, which the reader can verify with a
pencil. One binary outcome, no covariates, a sample that is 20% cases, and a population
known to be 40% cases. Calibration solves for a single number `λ` and assigns weights
proportional to `exp(λ·Y)`. Working the constraint through gives

```
λ = log[ odds(K) / odds(P) ] = log[ (0.40/0.60) / (0.20/0.80) ] = 0.9808
```

and therefore weights proportional to `K/P = 2.00` for each case and
`(1−K)/(1−P) = 0.75` for each control. Those are exactly the classical weights for a
choice-based sample with known population shares (Manski & Lerman 1977), which is
reassuring: the machinery had better reproduce what was already correct.
Running `entropy_balance(Y, [0.40])` returns `λ = 0.980829` and precisely those two
weights.

Two things are worth noticing, because both generalize:

- **`λ` is interpretable.** It is the log odds-ratio between the population prevalence and
  the sample prevalence — a measure of how hard the sample had to be pushed. It is the `θ`
  of the selection model, recovered rather than assumed, and `CalibrationDiagnostics.tilt`
  reports it.
- **Nothing was estimated.** Given `K`, the answer is a closed form. The sample supplied
  `P`; the register supplied `K`; no model intervened. This is why the anchored margin
  later turns out to have *zero* sampling variance — there was never anything to estimate.

Everything below is this calculation with more moments, an arbitrary base weight, and
consequently no closed form.

### The general statement

Write the **population-to-sample
density ratio** — the reweighting that turns the biased sample back into the population —
as log-linear in the covariates and outcomes:

```
log dP_population/dP_sample (X, Y) = a(X) + θ·g(Y)
```

Calibration returns the **minimum-divergence** weights `w ∝ d(X)·exp(λ·g(Y))` (base `d(X)`
tilted by the smallest exponential factor) that reproduce the supplied population moments
`g(Y)`. This is a *density-ratio* model, not a claim to have recovered each unit's inclusion
probability. Three consequences:

- **Anchored margins are exact by construction.** The reweighted sample reproduces each
  known prevalence exactly — because that is the constraint. It is not evidence the method
  "works", only that it did what it was told.
- **Coincidence with true IPW is conditional.** These weights equal the true
  inverse-probability weights `1/π(X, Y)` *only* when the density ratio genuinely lies in
  the tilt family — the base `d(X)` captures the covariate-driven part, `g(Y)` spans the
  outcome-driven part — and positivity holds (every relevant `(X, Y)` region has sample
  support).
- **Transfer to other estimands is conditional too.** Downstream means, variance
  components, and effect sizes are recovered only insofar as this density-ratio model is
  adequate for them (see the effect-size/collider section for where it is *not*).

So the honest one-liner is **not** "we infer the true inverse-probability weights" but:
*we estimate minimum-divergence weights that reproduce the known population moments, and
they equal the true IPW weights when the population-to-sample density ratio is spanned by
the base weights plus those moments.*

**Inverse vs odds base weights (`base_scheme`).** That separability is *exact* for one
familiar choice. Under logistic participation `logit π = a(X) + θ·Y`, the **inverse-odds**
weight `(1−π)/π = exp(−a(X))·exp(−θ·Y)` is exactly multiplicatively separable and
log-linear, so it composes cleanly with the `exp(λ·Y)` calibration tilt. The
Horvitz–Thompson weight `1/π = 1 + exp(−a(X)−θ·Y)` is **not** separable; it only approaches
the tilt family as inclusion becomes rare (`π → 0`, where `1/π ≈ (1−π)/π`).
`calibration_ipw(base_scheme="odds")` uses the exactly-composing form; `"inverse"` (the
default) is the standard IPW weight and is very close under strong selection. When selection
is on the outcome alone (no covariates in the base), the two agree exactly — the reweighting
is a per-class constant either way, which is why the liability-model `K/P` weights [below](#a-probit--liability-threshold-model-the-lee-et-al-transform-vs-ipw)
are exact IPW, not an approximation.

## Where this sits: density ratios, I-projection, and label shift

The construction is not ad hoc — it is one object seen through three established literatures,
and that is where its guarantees (and its boundary) come from.

- **An exponential-tilt density-ratio model.** Writing
  `log dP_population/dP_sample = a(X) + θ·g(Y)` is exactly the semiparametric *density-ratio*
  (exponential-tilt) model of Qin (1998) — the same tilt that underlies retrospective
  case-control sampling, where only the intercept shifts between prospective and separate-sample
  logistic fits (Anderson 1972; Prentice & Pyke 1979). The liability `K/P` weights are its
  one-outcome special case.
- **Calibration is an I-projection.** The "minimum-divergence weights" are the *information
  projection* of the base weights onto the set of distributions meeting the moment constraints
  `E_w[g(Y)] = target`: minimize Kullback–Leibler divergence subject to linear constraints
  (Csiszár 1975; the minimum-discrimination-information principle, Kullback 1959). Its convex
  dual is precisely the exponential tilt `w ∝ d(X)·exp(λ·g(Y))` that `entropy_balance` solves,
  so entropy balancing (Hainmueller 2012) and empirical-likelihood calibration (Qin & Lawless
  1994) are two views of the same optimization. Matching population moments by reweighting is
  also what kernel mean matching does for covariate shift (Gretton et al. 2009).
- **This is label shift.** With no covariates in the base — pure
  `outcome_calibration_weights(Y, [K])` — i3pw *is* the classic correction for **prior
  probability shift / label shift**: sample and population differ only in the label marginal
  `P(Y)`, and the fix is to reweight the sample to the known priors (Saerens et al. 2002;
  Storkey 2009; Lipton et al. 2018). i3pw generalises it two ways: (i) it tilts an arbitrary
  base weight `d(X)` from a participation model rather than uniform weights, and (ii) where
  black-box label-shift estimators must *infer* `P(Y)` from a classifier, i3pw takes `P(Y)` as
  a **known register quantity** — the regime where the correction is exact rather than estimated.

The placement also re-derives the honesty boundary. Label shift assumes the class-conditional
`P(X | Y)` is stable between sample and population — selection acts only *through* `Y` — which is
the exact analogue of "the density ratio lies in the tilt family" above. When selection also acts
*within* outcome classes, that assumption fails and so does the guarantee: this is precisely the
[case-mix](#prevalence-sets-the-scale-not-the-case-mix) caution (selection on severity within
cases) and the [collider](#participation-bias-and-effect-sizes-what-known-prevalences-cannot-fix)
boundary (selection on the exposure alongside the outcome), stated in a second language.

## Calibration is a regression estimator (and why the SEs look the way they do)

One classical result does more practical work here than any other, and it is what
`calibration_mean_se` implements. Deville & Särndal (1992) showed that **every** member of
the calibration family — raking, linear calibration, the entropy tilt used here — is
asymptotically equivalent to the **generalized regression (GREG)** estimator, to
`O(n^-3/2)`. They therefore all share one asymptotic variance, and the GREG variance
formula can be used for all of them. That is what rescues a package whose weights come
from a solve with no closed form.

Concretely, calibrating on `g` and then taking a weighted mean of `y` is asymptotically
the same as *regressing `y` on `g` and correcting the prediction*. So the influence
function is a **residual**:

```
e_i = y_i − μ − β·(g_i − ḡ),     β = weighted least squares of y on g
Var(μ̂) = Σ wᵢ² eᵢ² / (Σ wᵢ)²
```

Three consequences, all visible in the package's output:

- **An anchored margin gets SE exactly 0, for the right reason.** If `y` is itself a
  column of `g`, the regression fits it perfectly, the residual is identically zero, and
  the variance vanishes. That is not a numerical accident — it is the correct answer:
  conditional on a known prevalence, the reweighted margin has no sampling variability.
  The fixed-weight formula cannot see this and reports ≈0.04 instead.
- **An estimand orthogonal to the constraints is unaffected.** Then `β = 0`, the residual
  is `y − μ`, and the calibration SE reduces to the ordinary Hájek one.
- **Everything in between is shrunk** by exactly the variance the constraints absorbed —
  which is also why calibration *reduces* variance for quantities correlated with the
  anchors, not just corrects bias.

**Efficiency.** This is not merely convenient. Chan, Yam & Zhang (2016) show that
empirical balancing calibration weighting attains the **semiparametric efficiency bound**
globally — without nonparametric estimation of either the propensity or the outcome
regression, whose finite-sample behaviour is the usual weak point of efficient estimators.
The moment constraints inherit what the unknown propensity function would have supplied.

**Two robustness claims that look contradictory.** Zhao & Percival (2017) prove that
entropy balancing *is* doubly robust — with respect to a **linear outcome regression** and
a **logistic propensity model** in the balancing functions — and attains the semiparametric
variance bound when both hold. Elsewhere this package says it is *not* doubly robust. Both
are true, because they answer different questions, and it is worth being exact about which:

| | the question | answer |
| --- | --- | --- |
| Zhao & Percival | given the balancing functions `g`, may one of the two models over them be wrong? | **yes, either may be** |
| [What is identified?](#what-is-identified) | is `g` rich enough to span the outcome-driven part of selection at all? | **no guarantee** |

Their result buys robustness *within* a chosen `g`; it says nothing about choosing `g`
badly. If `g` misses the outcome-driven part, both models are wrong together and no
robustness property rescues either. That second question is the one this package cannot
settle by assumption — which is what the next section is about.

## What makes this falsifiable

An identification assumption that cannot fail is not doing scientific work. The one above
— the density ratio lies in the tilt family — is **untestable when the system is
just-identified**: supply exactly as many known moments as you calibrate on, and the
weights reproduce them by construction, whatever the truth. Every diagnostic the solve
emits (convergence, residual, ESS) is then a statement about the optimizer, not the world.

Supply **more** known population quantities than you constrain, and the surplus becomes
evidence. The unconstrained ones did not have to match; that they do is a test the model
could have failed. This is exactly the logic of a test of **overidentifying restrictions**
(Sargan 1958; Hansen 1982), transplanted from GMM: the constraints identify, the surplus
moments test. `balance_report` implements it, and deliberately bases its verdict *only* on
the held-out columns — see [balance as a falsification
test](#checking-the-weights-balance-as-a-falsification-test) for the demonstration where
every other diagnostic prefers the broken weighting.

The practical reading for a biobank: calibrate to the disease prevalences you need
corrected, and **hold back** register margins (age, sex, region, birth cohort) as tests. A
large held-out `|SMD|` refutes the tilt family; a small one is the strongest positive
evidence this framework can produce. It is not proof — passing an overidentification test
never is — but it is the difference between an assumption and a checked assumption.

## Two separable tasks: predict selection, then anchor to the population

It clarifies everything to split selection-bias correction into two tasks that i3pw
deliberately keeps separate:

1. **Predict who is in the sample** — an individual-level participation model
   `P(S = 1 | X)`, inverted to base weights. The predictors `X` can be socioeconomic
   *and* clinical or genetic (any measured proxy of participation), not just
   demographics. This corrects selection on *measured* covariates but is blind to
   selection on the disorder itself.
2. **Anchor the weighted sample to the target population** — calibrate those base
   weights so the reweighted sample reproduces known register quantities: disease
   prevalence, and prevalence *within* demographic and clinical strata. This is the
   task the known prevalences make possible, and where register data (e.g. iPSYCH,
   Danish registers) supplies what a selected genetic sample (UK Biobank, PGC-style
   cohorts) cannot.

`calibration_ipw` / `entropy_balance` implement task 2 on top of *any* task-1 base
weights — `entropy_balance(Y_sample, targets, base_weights=1/P̂)`. Keeping the two
apart is what makes the method defensible: the participation model handles the part of
selection it can see, and the register prevalences anchor the rest. It also states the
honest division of labour — *prediction* of selection at the individual level, and
*anchoring* of the weighted sample to the target population — rather than hoping one
model does both.

### A ladder of prevalence-informed weights

From simplest to most defensible, with the i3pw entry point for each. Each rung adds
constraints; `shrinkage=` (the entropy-balancing ridge) stabilises any of them against
extreme weights.

| Constraint you add | What it fixes | i3pw entry point |
| --- | --- | --- |
| Case/control prevalence `P(Y)=K` | overall case fraction | `outcome_calibration_weights(Y, [K])`; the `K/P` form in `estimate_liability_r2(method="ipw")` |
| Prevalence within strata | case mix across sex / birth year / ancestry / parental history / region | `stratified_calibration_weights` |
| Several known margins (raking) | multiple population totals at once | `outcome_calibration_weights` / `entropy_balance` with base weights |
| Calibrated IPW model | a fitted participation model **anchored** to `K` | `calibration_ipw(base="lasso")`, or `entropy_balance(Y, K, base_weights=1/P̂)` |
| Comorbidity / disease-state prevalence | joint case patterns, not just margins | `outcome_calibration_weights(..., joint_prevalences=...)` |
| Severity prevalence `P(severity given Y=1)` | over-/under-representation of severe cases | severity as a stratum in `stratified_calibration_weights` |
| Outcome model **and** weights | robustness if either is roughly right | `aipw_mean` (doubly robust) |
| Sensitivity to the assumed `K` | how much the answer leans on the register number | `prevalence_sensitivity` |

The natural recommended recipe for a register-linked genetic cohort is the middle of
the ladder: estimate base weights from demographic, clinical, and genetic predictors of
participation, then calibrate them to register prevalences by diagnosis, sex, birth
year, and severity — conventional IPW, prevalence-calibrated IPW, entropy-balanced
weights, and a doubly-robust estimator are all directly comparable here because they
share the same two-task structure.

### Prevalence sets the scale, not the case mix

The central caution. Calibrating to a known prevalence fixes the **number** of cases in
the weighted sample, not their **type**. If the sampled cases differ systematically from
the population's — e.g. UK Biobank holding a milder, higher-functioning subset of
schizophrenia — then matching the overall prevalence leaves that *within-case* selection
untouched: right count, wrong mix, and any estimand that depends on severity or
comorbidity stays biased. The fixes climb the ladder:

- calibrate prevalence **within severity / comorbidity strata**
  (`stratified_calibration_weights`), not just the marginal, so the weighted case mix
  matches the register's — this is usually the single most important step for
  psychiatric cohorts;
- past that, within-case selection on things you *cannot* stratify on (unmeasured
  severity, differential survival) is the residual risk that prevalence cannot fix.
  Fold the available proxies into the task-1 participation model, and report a
  sensitivity analysis (`prevalence_sensitivity`, plus varying the assumed within-case
  selection).

This is the same boundary as the [effect-size / collider section](#participation-bias-and-effect-sizes-what-known-prevalences-cannot-fix):
marginal prevalence anchors marginal quantities; anything driven by the *joint*
structure of selection needs richer constraints or a selection model that captures it.

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
part; see [What is identified?](#what-is-identified)). Otherwise it is the
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

## What the headline benchmark does *not* show (`examples/honest_benchmark.py`)

The 0.00% row above is an **algebraic identity**, not a result: every outcome is anchored,
so the estimator reproduces the prevalences it was handed. Two things follow, and both
are measured by `examples/honest_benchmark.py` rather than asserted.

**Transfer is the honest question.** Anchor outcome 1, evaluate outcome 2 — a disease
whose prevalence you never supplied. Mean absolute % error over 12 replications:

| | unanchored outcome | no correction |
| --- | --- | --- |
| `calibration_ipw(base="lasso")` | 77.2 ± 6.1 | — |
| `calibration_ipw(base="uniform")` | 78.5 ± 6.2 | — |
| naive sample prevalence | — | 78.4 ± 6.8 |

Calibrating on one disease does essentially **nothing** for another. That is not a bug —
it is the [case-mix caution](#prevalence-sets-the-scale-not-the-case-mix) and the
marginal-vs-joint boundary showing up as a number. Marginal calibration fixes marginal
quantities; anchor what you want corrected.

**The default's benefit depends on the simulation.** The shipped DGM had *no covariate
channel at all* — selection was a function of `Y` alone, so `P(S|X)` was only weakly
learnable second-hand through `X → Y → S` (AUC ≈ 0.58). On that setting a covariate base
model has nothing to fit and merely adds noise, and no comparison run there can
discriminate between the two channels the package is about.
`SimConfig.selection_covariate_strength` adds the missing channel, and with it the base
model earns its place — on worst held-out `|SMD|`, lower is better:

| | no covariate channel (default) | channel on (`strength=1.5`) |
| --- | --- | --- |
| `base="lasso"` | 0.084 ± 0.029 | **0.211 ± 0.115** |
| `base="uniform"` | 0.078 ± 0.037 | 0.566 ± 0.236 |

So the `base="lasso"` default is right *when selection actually has a covariate
component* — 2.7× better balance and 2.6× lower error on a held-out covariate mean. It
looked useless only because the simulation gave it nothing to learn. Benchmark on a
setting with the channel enabled before drawing conclusions about the base model.

Across 20 populations (`python examples/monte_carlo.py`), mean absolute % error ± SD:

```
method                     Y1 %err         Y2 %err
no_correction        46.81±5.70      79.53±6.49
lasso_ipw            44.66±5.93      78.50±6.61      <- see the caveat above
calibration_ipw       0.00±0.00       0.00±0.00      <- by construction
                                    (Kish effective sample size: 155 ± 32)
```

The value of calibration is *not* that it "predicts" a prevalence it was told. It is
that the resulting **weights** are correct along the ascertained dimensions, which
de-biases downstream estimands correlated with those outcomes. That correction costs
variance — hence the effective sample size, which shrinks as ascertainment strengthens.

One feasibility limit: a target is reachable only if the sample contains cases of that
outcome. For a very rare outcome the cases can be absent, and no reweighting gets there;
`shrinkage=` (or pooling outcomes) degrades gracefully instead of failing.

### Weighting schemes for the IPW baseline

`lasso_ipw` (and `monte_carlo`) accept `weighting=`:

- `"inverse"` — the Hájek (self-normalized) estimator (`1 / P`, sample only).
  **The default and the only deployable choice.**
- `"oracle_odds"` — `(1 - P) / P` for selected, weight 1 for unselected, mean over
  the whole test set. It reads unselected outcomes, so it flatters the method; it is
  a simulation-only oracle diagnostic.

Very large weights can be tamed with `trim=` (clip at a quantile, standard IPW practice).

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
assumption that [What is identified?](#what-is-identified) rests on.

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
  `CalibrationDiagnostics.tilt`; it *is* the estimated `θ` of `a(X) + θ·g(Y)`. Feed it
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

## Modifying a Schoeler-style weight to leverage known prevalences

A covariate participation model (Schoeler et al. 2023; van Alten et al. 2024)
weights the UK Biobank by `1/P̂(S | X_socio)` — correcting the sociodemographic
tilt but blind to selection that depends on the disorder itself. The modification:
use those weights as a **base**, then calibrate (rake) them to the known population
prevalence. Equivalently, add a `θ·Y` term to the log-participation model whose
coefficient is identified by the known prevalence — the
calibration-for-nonignorable-nonresponse construction of Kott & Chang (2010).
`examples/schoeler_plus_prevalence.py`, selection `= α + X_socio·c + θ·Y`,
recovering the disorder's liability-scale variance explained `R²_L`:

```
method       R²_L
truth        0.512
naive        2.478   (ascertainment uncorrected)
schoeler     3.747   (covariate IPW only: disease ascertainment still uncorrected)
modified     0.514   (Schoeler base + rake to known prevalence — recovers truth)
oracle       0.508   (1 / P(S | X_socio, Y))
```

The modification strictly extends Schoeler (it reduces to it when `θ → 0`) and is
already in the package: `outcome_calibration_weights(Y, K, base_weights=1/P̂)`.

How much can the prevalences buy you? Each known marginal prevalence pins down one
number in the selection model — the outcome's own participation effect — which is
enough to fix marginal quantities (prevalence, absolute risk, means, and the
liability-scale variance explained here). What it *cannot* pin down is how selection
depends on two things at once; recovering that (the interaction terms behind
effect-size bias) needs richer inputs — known co-occurrence rates, or prevalences
broken down by covariate strata. Marginals alone suffice above only because the
estimand is a variance component, not an effect size — see below.

## Inferring selection probabilities from many outcomes

The realistic version: a *latent* selection variable `U` drives participation
(`logit P(S|U) = α + γU`); there are `N` outcomes, each a noisy proxy for `U`; only
`k` are observed frame-wide (registry-linked); but the population **means** of all
`N` are known. How best to infer the selection probabilities (equivalently, the
weights `1/P(S)`)? `examples/selection_inference_extensive.py` runs four studies — a
comparison across selection regimes, sweeps over `N` and `k`, and a Schoeler-style
covariate comparison — scoring each method by held-out bias, effective sample size,
and how well its log-weights track the oracle's.

The one-line recipe, and the choice that is robust across every regime below: model
`P(S | outcomes observed frame-wide)` for the base weights, then calibrate to *all*
known population means — `entropy_balance(Y_sample, means, base_weights=1/P̂)`.

### Study A — the regime decides, and Lee-style weights are a bet

Alongside `registry`/`calib_all`/`combined` the benchmark adds a **Lee et al.
(2011)-style** analytic weight (`lee_cc`): the product over all `N` outcomes of the
case-control ratios `K_j/P_j` (case) and `(1−K_j)/(1−P_j)` (control) — model-free,
using the same `N` known means as calibration but assuming each outcome is an
*independent* case-control axis. The generative model dials between selection driven
purely by the latent `U` (`latent`), purely by a few observed outcomes
(`case_control`), or both (`mixed`). Held-out `|E[Z]−truth|`, 20 reps (lower is
better):

```
scenario        naive   lee_cc  registry  calib_all  combined   oracle
latent          0.582   0.036    0.330     0.256     0.254     0.013
case_control    0.344   0.326    0.008     0.009     0.008     0.008
mixed           0.540   0.139    0.197     0.158     0.150     0.011
```

The headline is that **no method is uniformly best**:

- **Latent regime** (every outcome proxies one hidden driver): `lee_cc` is
  startlingly good — averaging `N` simple case-control corrections reconstructs `U`
  with low variance and *beats exact joint calibration*, which chases sampling noise
  in each of the `N` margins.
- **Case-control regime** (a few *correlated* outcomes drive selection): `lee_cc`
  now **over-corrects** — it applies an independent correction for every outcome even
  though most only correlate with the true drivers — and is barely better than naive.
  A registry model or exact calibration, which cannot push past the true margins, are
  near-exact.
- **`combined` is the robust choice**: never catastrophic in any regime (0.25 / 0.008
  / 0.15). `lee_cc` swings from best (latent) to nearly-naive (case-control), and its
  effective sample size is low (≈0.4 of `n`) — it is a high-variance bet that pays off
  only when selection really is a latent factor cleanly proxied by all your outcomes.
- **Studies B/C (sweeps)**: calibration bias falls *monotonically* as the number of
  known means `N` grows, but `lee_cc` is *non-monotonic* — it improves then degrades
  once many weak correlated outcomes each add an over-correction. Only the registry
  (and `combined`) benefit from more frame-wide outcomes `k`.

Practical read: if you know selection is case-control on a specific known-prevalence
disorder, the analytic Lee/case-control weight is exact and cheap. If you don't know
the mechanism — the usual biobank situation — prefer `combined`: it never blows up,
and unlike `lee_cc` it keeps improving as you learn more prevalences.

### Study D — where Schoeler et al. fits in: covariate model and calibration are complementary

The methods above see only *outcomes*; the [Schoeler et al. (2023)](https://doi.org/10.1038/s41562-023-01579-9)
approach instead fits a participation model on **socioeconomic covariates** `X` — a
LASSO `P(S | X)`, inverted. Study D gives it a fair fight: a
population where selection depends on *both* a socioeconomic index `X@b` **and** the
disease latent `U` (with `X` independent of `U`), and a held-out trait `Z` that loads
on both channels. `schoeler` = LASSO `1/P̂(S|X)`; `sch+prev` = those weights used as a
base, then calibrated to the `N` known outcome means. Held-out `|E[Z]−truth|`:

```
selection channel   naive   schoeler  calib_all  sch+prev   oracle
socioeconomic       0.689    0.208     0.600      0.094      0.022
balanced            0.741    0.421     0.551      0.206      0.021
disease             0.689    0.548     0.424      0.276      0.020
```

- When selection is **socioeconomic**, the Schoeler covariate model removes most of the
  bias and prevalence calibration barely helps (the outcomes don't proxy an `X`-driven
  mechanism).
- When selection is **disease-driven**, the covariate model is nearly useless — this is
  the project's motivating failure, participation driven by *having the disease*, a
  signal orthogonal to `X` — and prevalence calibration does the work instead.
- The two are **complementary**: `sch+prev` (Schoeler weights calibrated to the known
  means) is best in *every* channel mix. So the recommended UK Biobank recipe is
  literally *Schoeler-plus-prevalences* — `entropy_balance(Y_sample, means,
  base_weights=1/P̂(S|X))` — with the covariate model handling the socioeconomic part
  and the known prevalences handling the disease part. (See also
  `examples/schoeler_plus_prevalence.py`.)

## Participation bias and effect sizes: what known prevalences cannot fix

The scientific target is usually an **effect size** (an exposure→outcome or genetic
association, an MR estimate), not a prevalence — and there the known-prevalence tool
mostly does not apply. `examples/ukb_participation.py` estimates a true effect `β` of an
exposure `E` on an outcome `Y` under participation `logit P(S|E,Y) = α + δ_E·E + δ_Y·Y`:

```
                     outcome-only (δ_E=0)     collider (δ_E=0.8)
β_truth                    1.096                    1.096
β_naive                    1.094  (unbiased)        1.274  (+16%)
β_prev_calib               1.107                    1.313  (calibration doesn't fix it)
β_model_ipw  P̂(S|E,Y)      1.107                    1.110  (recovers β)
β_oracle     1/P(S|E,Y)    1.107                    1.110
```

Two facts, both important:

1. **Selection on the outcome alone does not bias the effect size** (Prentice & Pyke 1979:
   the logistic slope is consistent) — `β_naive ≈ β_truth`. There is nothing to correct,
   and reweighting only adds variance.
2. **The effect-size bias that matters is collider bias** — participation depending on the
   exposure *and* the outcome (the regime behind Schoeler et al.'s distorted genetic
   associations and MR estimates). There, **prevalence calibration does not help** (`1.313`
   vs truth `1.096`): an effect size is a *conditional association* (a joint moment), and
   matching the outcome's *marginal* leaves the exposure-outcome *joint* selection untouched.
   Only weights from a sampling model that **includes the exposure**, `P(S|E,Y)`, recover it.

So: known prevalences are the right information for **prevalences, absolute risk and means**
(where calibration is exact) — and essentially the **wrong** information for **effect sizes**.
Correcting effect-size (collider) bias needs a participation model that captures the variables
driving selection (Schoeler-style IPW), and its validity rests entirely on that model being
right — something known prevalences cannot supply or verify.

## Several case-control outcomes at once: joint calibration is optimal

Now `Q` outcomes are ascertained together, and every combination of their
case/control statuses can be recruited at its own rate — so the selection
probability `π(y)` is one unknown number per outcome pattern (`2^Q` of them). The
known prevalences give us moments to pin those numbers down. `outcome_calibration_weights`
calibrates the weights to all the outcomes jointly; `examples/multi_outcome_calibration.py`
tests how well the reweighted sample then recovers two targets it did *not* calibrate
to — an additive `E[L1+L2]` and a joint `E[L1·L2]` (bias, 10 reps):

```
                    independent selection (g=1)     comorbid interaction (g=2.5)
method               E[L1+L2]   E[L1*L2]             E[L1+L2]   E[L1*L2]
naive                  2.30       1.44                 2.88       2.05
mean-combine           0.79       0.22                 0.82       0.25
product-combine       -0.29      -0.09                -0.41      -0.13
calib_marginal        -0.005     +0.002               -0.010     +0.032
calib_joint           -0.005     +0.003               +0.003     -0.005
oracle (1/π)          -0.005     +0.002               +0.008     -0.004
```

The optimum has a precise characterization:

- **Joint calibration dominates the per-outcome heuristics.** Combining separate
  case/control weights by `mean` or `product` is biased; jointly solving the
  marginal constraints (entropy balancing) is the principled combination.
- **Match the calibration terms to the selection structure.** When selection is
  multiplicative in the outcomes (each outcome scales the inclusion odds
  independently, `g = 1`), `log π(y)` is linear in `y`, the `Q` known marginals
  identify it, and **marginal calibration equals the oracle** — even on the joint
  target.
- **Coupled selection needs the joint moments.** When comorbid cases are recruited
  specially (`g > 1`), `log π` has an interaction term that the `Q` marginals cannot
  represent, so `calib_marginal` is biased on the joint target (`+0.032`). Adding the
  known co-occurrence `P(Y1=1, Y2=1)` as a constraint (`calib_joint`) restores the
  oracle. In general you must calibrate to every population moment the selection
  model needs — marginals for independent ascertainment, plus co-occurrences (and
  higher-order joints) when the outcomes are sampled in a coupled way.

`outcome_calibration_weights(Y, prevalences, joint_prevalences={(0,1): k12})` builds
these constraints; if you actually *know* the per-outcome sampling design, the exact
weights `1/π(y)` dominate everything.

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

## A probit / liability-threshold model: the Lee et al. transform vs IPW

A separate, self-contained study (`i3pw.liability`, benchmarked in
`examples/probit_selection_lee_vs_ipw.py`). Latent Gaussian liability
`L = f(X) + e`, binary outcome `Y = 1[L > t]`, prevalence `K`; the estimand is the
liability-scale variance explained `R²_L = Var(f)/Var(L)`. The sample is
ascertained on `Y` (cases over-represented), so the sample case fraction `P ≠ K`.
Two corrections:

- **Lee et al. (2011)** — estimate `R²` on the observed 0/1 scale, then multiply by
  `[K(1-K)/z²] · [K(1-K)/(P(1-P))]` (observed→liability × an analytic ascertainment
  factor).
- **IPW** — reweight the case fraction back to `K` (weights `K/P`, `(1-K)/(1-P)` —
  the exact inverse-probability weights for selection on `Y` alone), run a weighted
  moment estimator, then apply only the population `K(1-K)/z²` factor.

Both correct the ascertainment; the Lee factor is the *analytic* counterpart of what
IPW does by *reweighting*. Benchmark (25 reps, strong-ascertainment `P = 0.5` rows):

```
 true R²_L    K      naive        lee          ipw
   0.50     0.01   12.21±0.40   0.483±0.02   0.494±0.02
   0.50     0.10    1.40±0.11   0.503±0.04   0.505±0.04
   0.80     0.01   19.46±0.71   0.771±0.03   0.807±0.03
   0.80     0.05    4.08±0.15   0.775±0.03   0.788±0.03
```

Findings:

- **Ignoring ascertainment is catastrophic** — the naive estimate is inflated up to
  ~24× (it is worse for rarer `K` and more balanced `P`).
- **Lee and IPW both work**, and agree closely at moderate `R²` / mild ascertainment.
- **They diverge exactly where theory predicts.** IPW removes the selection *exactly*
  at any strength (it is design-based); the Lee ascertainment factor is a
  linearization, so as effects grow (`R²_L = 0.8`) *and* ascertainment is strong,
  Lee drifts low (−3 to −4%) while IPW stays within ~1–2%. Both still share the
  observed→liability approximation, so both sit slightly low at high `R²`.
- **No variance penalty** for IPW here — the SDs match Lee's. (The design-based /
  moment route is the same idea as PCGC regression, which is the ascertainment-exact
  fix to the Lee transform.)

### When selection depends on more than the outcome

The comparison above is a level playing field: selection is a pure function of the
outcome, so Lee and IPW have the same information. But IPW's real advantage appears
when selection is *more complex*. `examples/complex_selection_ipw.py` makes selection
depend slightly on the latent liability too, `logit P(S=1|Y,L) = a_Y + δ·L` (e.g.
severity-dependent recruitment, super-normal controls):

```
delta   truth      lee   ipw_simple   ipw_fitted   ipw_oracle
  0.0   0.605    0.594      0.602        0.595        0.599
  0.6   0.602    0.399      0.407        0.571        0.604
  1.2   0.598    0.279      0.284        0.542        0.586
```

- `δ = 0` (pure case-control): everything works.
- `δ > 0`: **Lee and simple `K/P` IPW fail identically** — both know only `(K, P)`, so
  both assume selection is a function of the outcome alone and miss the within-group
  liability selection.
- **`ipw_fitted`** — weights from a *fitted* `P(S|X, Y)` — recovers most of it, limited
  by how well the predictors `X` proxy the latent liability.
- **`ipw_oracle`** — weights `1/P(S=1|Y,L)` from the *true* inclusion probabilities —
  is exact.

The lesson: a closed-form transform is stuck with the selection model it assumes, but
IPW is only as good as the sampling probabilities you can supply — and if you can
*estimate* or *know* them, it keeps working where the transform cannot.

## Conclusions and recommendations

Everything above, condensed. Each recommendation is followed by the reason for it, and
the reason links to the section that establishes it — if you disagree with a
recommendation, the disagreement is really with the evidence behind it, so go there.

### The recipe

1. **Fit a participation model on whatever predicts participation** — demographic,
   socioeconomic, clinical, genetic, or outcomes observed frame-wide — and invert it for
   base weights.
   *Why:* it corrects the part of selection that is visible in covariates, and it is the
   only ingredient that can. On a population where participation genuinely depends on a
   covariate, dropping this step costs a factor of 2.7 in held-out balance
   ([benchmark](#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy)).
2. **Calibrate those weights to known register quantities** rather than trusting the
   model alone — `entropy_balance(Y, targets, base_weights=1/P̂)`.
   *Why:* participation depends on *having the disease*, which the covariates barely
   proxy, so the covariate model alone leaves the ascertainment uncorrected. The two
   ingredients are complementary and the combination is best in **every** channel mix
   tested ([Study D](#study-d--where-schoeler-et-al-fits-in-covariate-model-and-calibration-are-complementary)).
3. **Calibrate within strata — sex, birth year, ancestry, severity — not just the pooled
   margin** (`stratified_calibration_weights`).
   *Why:* a known prevalence fixes the *number* of cases, not their *type*. If the
   sampled cases are milder than the population's, matching the margin leaves that
   uncorrected ([case mix](#prevalence-sets-the-scale-not-the-case-mix)). For psychiatric
   cohorts this is usually the single most important step.
4. **Hold back some known margins and test against them** (`balance_report`).
   *Why:* a just-identified calibration reproduces its constraints whatever the truth, so
   nothing it reports can refute it. Held-out moments are the only thing that can
   ([falsifiability](#what-makes-this-falsifiable)) — and in a worked case every other
   diagnostic *preferred* the broken weighting, whose ESS looked 3× healthier
   ([demonstration](#checking-the-weights-balance-as-a-falsification-test)).
5. **Report the effective sample size, a sensitivity sweep over `K`, and an interval that
   accounts for the estimated weights** (`calibration_mean_se`, or the bootstrap).
   *Why:* the correction costs variance, register prevalences are not exact constants, and
   the fixed-weight SE is not a bound in either direction ([uncertainty](#uncertainty)).

### What not to do

- **Do not use known prevalences to fix effect sizes.** Selection on the outcome alone
  does not bias a logistic slope (Prentice & Pyke 1979), so there is nothing to correct;
  and the bias that *does* matter — collider bias, from participation depending on
  exposure and outcome together — is untouched by calibration, which moved the estimate
  from 1.274 to 1.313 against a truth of 1.096. Only a participation model including the
  exposure recovers it
  ([evidence](#participation-bias-and-effect-sizes-what-known-prevalences-cannot-fix)).
  Known prevalences are the right information for prevalences, absolute risks and means,
  and the wrong information for associations.
- **Do not expect calibrating on one disease to help with another.** Anchoring outcome 1
  left outcome 2 at 77.2% error against 78.4% for no correction at all. Marginal
  calibration fixes marginal quantities; anchor what you need corrected
  ([transfer](#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy)).
- **Do not quote a `0.00` as evidence.** An anchored margin is reproduced by construction.
  This is worth repeating because the package's own headline table invites the mistake.
- **Do not reach for the analytic Lee-style product weight unless you know the mechanism
  is case-control on known-prevalence disorders.** It is excellent when selection really
  is one latent factor cleanly proxied by every outcome, and barely better than naive when
  a few correlated outcomes drive selection — a high-variance bet either way
  ([Study A](#study-a--the-regime-decides-and-lee-style-weights-are-a-bet)).

### The one risk that outranks the others

Everything here assumes the population-to-sample density ratio lies in the tilt family —
that the base weights span the covariate-driven part of selection and the supplied moments
span the outcome-driven part. That assumption is not testable from the calibration itself,
and no robustness property covers it: if the moments miss the outcome-driven part, the
outcome model and the propensity model fail *together*
([what is identified](#what-is-identified)).

The practical consequence is recommendation 4, which is why it is not optional. Passing an
overidentification test is not proof — it never is — but it is the difference between an
assumption and a checked assumption, and it is the strongest positive evidence this
framework can produce.

## Package layout

```
src/i3pw/
├── calibration.py  # the core: entropy_balance, calibration_ipw, apply_tilt,
│                   #   calibration_mean_se, stratified/joint calibration, diagnostics
├── balance.py      # balance_report: the falsification test (held-out moments)
├── uncertainty.py  # bootstrap, fixed-weight SE, prevalence-sensitivity
├── aipw.py         # aipw_mean: doubly-robust downstream estimation (+ cross-fitting)
├── liability.py    # probit / liability-threshold model: Lee et al. transform vs IPW
├── methods.py      # baselines: no_correction, lasso_ipw / lasso_propensity
├── dgm.py          # simulated population + biased sampling
├── evaluation.py   # Monte Carlo comparison across many replications
├── metrics.py      # weighted (Hájek) prevalence, % difference
└── _links.py       # stable sigmoid / logit / probability clamp
tests/              # pytest suite
examples/           # honest_benchmark.py is the one to read first
```

## Tests

```bash
pytest

# If unrelated third-party pytest plugins in your environment interfere,
# disable plugin autoload:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## References

**Selection / participation bias in volunteer cohorts (the applied motivation):**

- Schoeler, T. et al. (2023). Participation bias in the UK Biobank distorts genetic
  associations and downstream analyses. *Nature Human Behaviour* 7, 1216–1227.
  [doi:10.1038/s41562-023-01579-9](https://doi.org/10.1038/s41562-023-01579-9)
- van Alten, S., Domingue, B. W., Faul, J., Galama, T., Marees, A. T. (2024).
  Reweighting UK Biobank corrects for pervasive selection bias due to volunteering.
  *International Journal of Epidemiology* 53(3), dyae054.
- Schoeler, T. et al. (2025). Correcting for volunteer bias in GWAS increases SNP
  effect sizes and heritability estimates. *Nature Communications* 16.
- Munafò, M. R. et al. (2018). Collider scope: when selection bias can substantially
  influence observed associations. *Int. J. Epidemiol.* 47(1), 226–235.
- Elliott, M. R. & Valliant, R. (2017). Inference for nonprobability samples.
  *Statistical Science* 32(2), 249–264.

**Inverse-probability weighting and calibration (the machinery):**

- Horvitz, D. G. & Thompson, D. J. (1952). A generalization of sampling without
  replacement from a finite universe. *JASA* 47(260), 663–685. *(the IPW estimator)*
- Hájek, J. (1971). Comment on a paper by D. Basu. In *Foundations of Statistical
  Inference*, eds. V. P. Godambe & D. A. Sprott. Holt, Rinehart & Winston.
  *(the self-normalized ratio estimator)*
- Deville, J.-C. & Särndal, C.-E. (1992). Calibration estimators in survey sampling.
  *JASA* 87(418), 376–382.
- Hainmueller, J. (2012). Entropy balancing for causal effects. *Political Analysis*
  20(1), 25–46. *(the exact form `entropy_balance` solves)*
- Kott, P. S. & Chang, T. (2010). Using calibration weighting to adjust for
  nonignorable unit nonresponse. *JASA* 105(491), 1265–1275. *(the prevalence-informed
  base-weight modification)*
- Manski, C. F. & Lerman, S. R. (1977). The estimation of choice probabilities from
  choice based samples. *Econometrica* 45(8), 1977–1988.
- Kish, L. (1965). *Survey Sampling.* Wiley. *(effective sample size / design effect)*
- Csiszár, I. (1975). I-divergence geometry of probability distributions and
  minimization problems. *Annals of Probability* 3(1), 146–158. *(the I-projection /
  minimum-KL-subject-to-moment-constraints that calibration solves)*
- Kullback, S. (1959). *Information Theory and Statistics.* Wiley (Dover reprint 1968).
  *(the minimum-discrimination-information principle)*
- Qin, J. & Lawless, J. (1994). Empirical likelihood and general estimating equations.
  *Annals of Statistics* 22(1), 300–325. *(calibration as empirical likelihood under
  moment constraints)*
- Qin, J. (1998). Inferences for case-control and semiparametric two-sample density
  ratio models. *Biometrika* 85(3), 619–630. *(the exponential-tilt density-ratio model
  the identification section uses)*
- Anderson, J. A. (1972). Separate sample logistic discrimination. *Biometrika* 59(1),
  19–35. *(logistic participation under retrospective / separate sampling)*
- Zhao, Q. & Percival, D. (2017). Entropy balancing is doubly robust. *Journal of Causal
  Inference* 5(1), 41–55. *(double robustness w.r.t. a linear outcome regression and a
  logistic propensity model — the axis i3pw's caveat is **not** about)*
- Chan, K. C. G., Yam, S. C. P., Zhang, Z. (2016). Globally efficient non-parametric
  inference of average treatment effects by empirical balancing calibration weighting.
  *JRSS-B* 78(3), 673–700. [doi:10.1111/rssb.12129](https://doi.org/10.1111/rssb.12129)
  *(calibration weighting attains the semiparametric efficiency bound)*
- Sargan, J. D. (1958). The estimation of economic relationships using instrumental
  variables. *Econometrica* 26(3), 393–415.
- Hansen, L. P. (1982). Large sample properties of generalized method of moments
  estimators. *Econometrica* 50(4), 1029–1054. *(the overidentification test whose logic
  `balance_report` transplants)*
- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of baseline
  covariates between treatment groups in propensity-score matched samples. *Statistics in
  Medicine* 28(25), 3083–3107. *(standardized mean differences; the |SMD| < 0.1 rule)*

**Distribution shift: density-ratio and label-shift correction (the same problem in
machine learning):**

- Saerens, M., Latinne, P., Decaestecker, C. (2002). Adjusting the outputs of a
  classifier to new a priori probabilities: a simple procedure. *Neural Computation*
  14(1), 21–41. *(prior-probability / label-shift correction — the covariate-free case
  of calibrating to known `P(Y)`)*
- Storkey, A. (2009). When training and test sets are different: characterizing learning
  transfer. In *Dataset Shift in Machine Learning*, eds. Quiñonero-Candela et al., ch. 1,
  3–28. MIT Press. *(the taxonomy naming "prior probability shift")*
- Lipton, Z. C., Wang, Y.-X., Smola, A. (2018). Detecting and correcting for label shift
  with black box predictors. *ICML*, PMLR 80, 3128–3136. *(estimating `P(Y)` from a
  classifier — the regime where i3pw instead takes it as known)*
- Gretton, A., Smola, A., Huang, J., Schmittfull, M., Borgwardt, K., Schölkopf, B.
  (2009). Covariate shift by kernel mean matching. In *Dataset Shift in Machine
  Learning*, ch. 8, 131–160. MIT Press. *(moment-matching reweighting, the covariate-shift
  analogue)*

**Doubly-robust and nonprobability-sample inference:**

- Robins, J. M., Rotnitzky, A., Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *JASA* 89(427), 846–866.
- Bang, H. & Robins, J. M. (2005). Doubly robust estimation in missing data and causal
  inference models. *Biometrics* 61(4), 962–973.
- Chen, Y., Li, P., Wu, C. (2020). Doubly robust inference with nonprobability survey
  samples. *JASA* 115(532), 2011–2021.
- Yang, S., Kim, J. K., Song, R. (2020). Doubly robust inference when combining
  probability and non-probability samples with high-dimensional data. *JRSS-B* 82(2),
  445–465. *(data integration: a non-probability sample anchored to population
  quantities)*
- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and
  structural parameters. *Econometrics Journal* 21(1), C1–C68. *(cross-fitting)*

**Case-control ascertainment and the liability-threshold model:**

- Prentice, R. L. & Pyke, R. (1979). Logistic disease incidence models and
  case-control studies. *Biometrika* 66(3), 403–411.
- Dempster, E. R. & Lerner, I. M. (1950). Heritability of threshold characters.
  *Genetics* 35(2), 212–236. *(the observed→liability transform)*
- Haseman, J. K. & Elston, R. C. (1972). The investigation of linkage between a
  quantitative trait and a marker locus. *Behavior Genetics* 2(1), 3–19. *(the
  method-of-moments variance-component estimator)*
- Lee, S. H., Wray, N. R., Goddard, M. E., Visscher, P. M. (2011). Estimating missing
  heritability for disease from genome-wide association studies. *AJHG* 88(3), 294–305.
- Golan, D., Lander, E. S., Rosset, S. (2014). Measuring missing heritability:
  inferring the contribution of common variants. *PNAS* 111(49), E5272–E5281 (PCGC).

**Software:**

- Friedman, J., Hastie, T., Tibshirani, R. (2010). Regularization paths for generalized
  linear models via coordinate descent. *Journal of Statistical Software* 33(1), 1–22.
  *(`glmnet`, the LASSO/coordinate-descent baseline)*

## License

MIT — see [LICENSE](LICENSE).


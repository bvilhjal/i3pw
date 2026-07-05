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
├── calibration.py  # calibration_ipw, entropy_balance, outcome_calibration_weights
├── aipw.py         # aipw_mean: doubly-robust downstream estimation
├── liability.py    # probit / liability-threshold model: Lee et al. transform vs IPW
├── methods.py      # no_correction, lasso_ipw / lasso_propensity, penalized_ipw
├── penalized.py    # PenalizedIPW estimator (gd / bfgs / lbfgs)
├── _kernels.py     # numba-compiled objective, gradient, gradient descent
├── weights.py      # per-outcome weight combination rules
├── evaluation.py   # Monte Carlo comparison across many replications
├── metrics.py      # weighted prevalence, % difference, weighted MSE
└── _links.py       # stable sigmoid / logit
tests/              # pytest suite (calibration, AIPW, liability, DGM, methods)
examples/           # benchmark.py, monte_carlo.py, doubly_robust_trait.py,
                    #   probit_selection_lee_vs_ipw.py, complex_selection_ipw.py,
                    #   multi_outcome_calibration.py, ukb_participation.py,
                    #   schoeler_plus_prevalence.py, selection_inference_extensive.py
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

## References

Selection / participation bias and reweighting:

- Schoeler, T. et al. (2023). Participation bias in the UK Biobank distorts genetic
  associations and downstream analyses. *Nature Human Behaviour* 7, 1216–1227.
- van Alten, S., Domingue, B. W., Faul, J., Galama, T., Marees, A. T. (2024).
  Reweighting UK Biobank corrects for pervasive selection bias due to volunteering.
  *International Journal of Epidemiology* 53(3), dyae054.
- Schoeler, T. et al. (2025). Correcting for volunteer bias in GWAS increases SNP
  effect sizes and heritability estimates. *Nature Communications* 16.
- Munafò, M. R. et al. (2018). Collider scope: when selection bias can substantially
  influence observed associations. *Int. J. Epidemiol.* 47(1), 226–235.

Calibration / weighting methodology:

- Deville, J.-C. & Särndal, C.-E. (1992). Calibration estimators in survey sampling.
  *JASA* 87(418), 376–382.
- Kott, P. S. & Chang, T. (2010). Using calibration weighting to adjust for
  nonignorable unit nonresponse. *JASA* 105(491), 1265–1275. *(the modification here)*
- Hainmueller, J. (2012). Entropy balancing for causal effects. *Political Analysis*
  20(1), 25–46.
- Manski, C. F. & Lerman, S. R. (1977). The estimation of choice probabilities from
  choice based samples. *Econometrica* 45(8), 1977–1988.
- Chen, Y., Li, P., Wu, C. (2020). Doubly robust inference with nonprobability survey
  samples. *JASA* 115(532), 2011–2021.
- Robins, J. M., Rotnitzky, A., Zhao, L. P. (1994). Estimation of regression
  coefficients when some regressors are not always observed. *JASA* 89(427), 846–866.

Case-control / liability-threshold model:

- Prentice, R. L. & Pyke, R. (1979). Logistic disease incidence models and
  case-control studies. *Biometrika* 66(3), 403–411.
- Lee, S. H., Wray, N. R., Goddard, M. E., Visscher, P. M. (2011). Estimating missing
  heritability for disease from genome-wide association studies. *AJHG* 88(3), 294–305.
- Golan, D., Lander, E. S., Rosset, S. (2014). Measuring missing heritability:
  inferring the contribution of common variants. *PNAS* 111(49), E5272–E5281 (PCGC).

## License

MIT — see [LICENSE](LICENSE).

# i3pw — simulation studies

The evidence behind the claims in the [README](../README.md) and
[theory.md](theory.md). Every study here is a simulation with a known truth, run by a
script in `examples/`, so any number can be reproduced by running it. Between them these
studies answer research questions [1 and 3](../README.md#the-research-question) — *can
known prevalences supply what the covariate model misses*, and *which estimands does that
fix*.

Read the [first section](#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy)
before any of the others: it is the one that tries hardest to make the method look bad,
and it is where the front page's `0.00` gets its caveat. Symbols are defined in the
[notation table](theory.md#notation); sources for the methods compared here are in the
[bibliography](theory.md#references).

**The short version of all of it.** Calibration is exact for what you anchor and does
essentially nothing for what you don't ([transfer](#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy));
it is complementary to — not a replacement for — a covariate participation model, and the
combination wins in every regime tested ([Study D](#study-d--where-schoeler-et-al-fits-in-covariate-model-and-calibration-are-complementary));
and it is the right information for prevalences, means and variance components, and the
*wrong* information for effect sizes ([collider](#participation-bias-and-effect-sizes-what-known-prevalences-cannot-fix)).


## What the headline benchmark does *not* show (`examples/honest_benchmark.py`)

The benchmark table on the [front page](../README.md#quick-start--the-simulations) reports `0.00` for
`calibration_ipw`. That row is an **algebraic identity**, not a result: every outcome
there is anchored, so the estimator reproduces the prevalences it was handed. Two things
follow, and both are measured by `examples/honest_benchmark.py` rather than asserted.

**Transfer is the honest question.** Anchor outcome 1, evaluate outcome 2 — a disease
whose prevalence you never supplied. Mean absolute % error over 12 replications:

| | unanchored outcome | no correction |
| --- | --- | --- |
| `calibration_ipw(base="lasso")` | 77.2 ± 6.1 | — |
| `calibration_ipw(base="uniform")` | 78.5 ± 6.2 | — |
| naive sample prevalence | — | 78.4 ± 6.8 |

Calibrating on one disease does essentially **nothing** for another. That is not a bug —
it is the [case-mix caution](theory.md#prevalence-sets-the-scale-not-the-case-mix) and the
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
lasso_ipw            44.66±5.93      78.50±6.61      <- no covariate channel to learn
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

- **Lee et al. (2011)** — estimate $R^2$ on the observed 0/1 scale, then multiply by

  ```math
  \underbrace{\frac{K(1-K)}{z^2}}_{\text{observed} \to \text{liability}}
  \times
  \underbrace{\frac{K(1-K)}{P(1-P)}}_{\text{ascertainment}}
  ```

  where $z$ is the standard-normal density at the threshold.
- **IPW** — reweight the case fraction back to $K$ (weights $K/P$ for cases and
  $(1-K)/(1-P)$ for controls — the exact inverse-probability weights for selection on $Y$
  alone), run a weighted moment estimator, then apply only the population
  $K(1-K)/z^2$ factor.

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

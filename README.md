# i3pw — Informed Inference of Inverse Probability Weights

Correcting **outcome-dependent selection (ascertainment) bias** by reweighting a biased
sample, when the population prevalences of the outcomes are known a priori.

## Aim

**Make a cohort that recruited the wrong mix of people answer questions about the
population it was drawn from — using disease prevalences from a register as the
information the cohort itself cannot supply.**

Biobanks and volunteer cohorts are not random samples of their populations. Who agreed to
take part depends, among other things, on whether they were ill. Any prevalence, mean, or
absolute risk computed from such a cohort is therefore an estimate about *participants*,
not about the population — and no amount of data collected inside the cohort reveals by
how much, because the people who declined are simply absent. What *is* available, in a
country with registers, is the answer at the population level: the true prevalence of the
disease. i3pw is the machinery for turning that external number into weights.

## The research question

Three questions, in the order they have to be answered. Each is the subject of one of the
three documents below.

1. **Can a known population prevalence supply the part of selection that a covariate model
   cannot see?** The standard correction fits `P(participate | X)` on socioeconomic
   covariates. If people participate partly because they *have the disease*, that channel
   is invisible to `X` — so is it recoverable from the register instead?
   → *[docs/studies.md](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md), and the sketch under [The problem, and the
   idea](#the-problem-and-the-idea).*
2. **What do the resulting weights actually identify?** They reproduce the register's
   prevalence by construction, which proves nothing. So under what condition are they the
   true inverse-probability weights, and what are they when that condition fails?
   → *[What is identified?](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#what-is-identified).*
3. **Which estimands does this fix, and which does it leave broken — and how would you
   find out on a real cohort?** Prevalences and means, yes; effect sizes, no. And since a
   calibration always matches its own constraints, what evidence could ever refute it?
   → *[What makes this falsifiable](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#what-makes-this-falsifiable) and
   [studies](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md).*

Question 3 is the one most often skipped and the one that decides whether a number from
this package belongs in a paper.

## The problem, and the idea

You have a biased sample — units were selected in a way that depends on their
outcomes (e.g. cases oversampled in a case-control or volunteer cohort), so
outcome prevalences in the sample are skewed, and so is everything estimated
from it. The standard fix models each unit's participation probability
`P(selected | X)` from covariates (socioeconomic features, via LASSO) and
reweights by `1 / P`.

**That participation model works poorly for many disease outcomes.** Participation can
depend on *having the disease*, a signal largely orthogonal to the covariates, so a
covariate-only model can miss an outcome-dependent part of the population-to-participant
density ratio and the weights barely correct anything.

i3pw's idea: **use known population prevalences to estimate that missing density-ratio
tilt.**
Knowing `Pr(Y_q)` a priori (from a registry or census) is exactly the information
the covariate model lacks, and injecting it as a **calibration constraint** — force
the reweighted sample to reproduce the known prevalences — supplies the
disease-driven part of the reweighting that the covariate model cannot. What that
does and does not identify is made precise in
[What is identified?](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#what-is-identified): calibration recovers
*minimum-divergence* weights matching the known moments, which coincide with the true
inverse-probability weights under a stated condition.

### What it computes, stated exactly

Not a recovered per-unit inclusion probability, but the **minimum-divergence weights that
reproduce the known population moments**. Those equal the true inverse-probability weights
under a condition that is stated, not assumed —
[What is identified?](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#what-is-identified) — and are the closest reweighting
to your starting weights otherwise. The package name predates the precision; the docs are
the authority on what it does.

If you would rather see it than read it, the whole method is one calculation on one binary
outcome, small enough to do with a pencil:
[a case small enough to check by hand](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#a-case-small-enough-to-check-by-hand).
The symbols used throughout are collected in one table:
[notation](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#notation).

### If you are a statistical geneticist

You already use the one-outcome case of this method. When a case-control study over-samples
cases — sample fraction `P`, population prevalence `K` — the weights that undo it are `K/P`
for cases and `(1−K)/(1−P)` for controls. That is the ascertainment factor
`K(1−K)/(P(1−P))` in the Lee et al. (2011) transform, and it is what PCGC does by
reweighting instead of by a closed form. i3pw is that correction with three generalisations,
each of which matters for a register-linked cohort:

| | classical ascertainment correction | i3pw |
| --- | --- | --- |
| starting weights | uniform | **any participation model** — Schoeler-style `1/P̂(S \| X_socio)`, or one built on clinical and genetic predictors |
| constraints | one prevalence `K` | **several diagnoses jointly**, comorbidity rates, and prevalence *within* sex / birth year / ancestry / severity strata |
| the tilt | assumed to be `K/P` | **solved for**, and reported: `λ` is the outcome coefficient in the log population-to-participant density ratio; it is not generally the participation-logit coefficient |

The setting it was built for is the Danish one: **iPSYCH and the national registers supply
`K`** — by diagnosis, by sex, by birth cohort — for a population in which a selected genetic
sample sits, and calibration is how you make the second answer questions about the first.

The estimands it repairs are the ones on the **liability and absolute scales**: prevalence,
absolute risk, trait and biomarker means, and liability-scale variance components
(`h²_l`, `R²_L`) — see the
[liability-threshold study](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#a-probit--liability-threshold-model-the-lee-et-al-transform-vs-ipw),
where IPW matches Lee under pure case-control ascertainment and beats it once ascertainment
is strong or depends on liability within case status.

The estimand it does **not** repair is the one you probably care about most: **GWAS effect
sizes**. Case-control ascertainment leaves a logistic slope unbiased (Prentice & Pyke 1979),
so under outcome-only selection there is nothing to correct. The distortion that Schoeler et
al. (2023) document in UK Biobank is *collider* bias — participation depending on genotype/
exposure *and* outcome — and a known marginal prevalence cannot touch it, because an effect
size is a joint moment and `K` is a marginal one. In the simulation it moves the estimate the
wrong way, 1.274 → 1.313 against a truth of 1.096
([evidence](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#participation-bias-and-effect-sizes-what-known-prevalences-cannot-fix)).
For that you need a participation model containing the exposure; prevalences will not
substitute.

## Install

```bash
pip install -e .           # from a clone of this repo
pip install -e '.[test]'   # with the test dependencies
```

Requires Python ≥ 3.10 and numpy / scipy / scikit-learn.

## Quick start — a real cohort

`calibrate` takes what a cohort actually has: outcomes for the people who participated,
prevalences from a register, and optionally base weights from your own participation
model. No simulation object, no ground truth.

```python
import i3pw

fit = i3pw.calibrate(
    Y=case_status[:, None],          # outcomes, participants only
    targets=[0.012],                 # what the register says the prevalence is
    base_weights=i3pw.inverse_probability_weights(p_hat),   # your participation model
    holdout={"mean age": (age, 41.7), "% female": (female, 0.508)},   # NOT calibrated on
)

print(fit.summary())              # solve diagnostics + held-out balance diagnostic
print(fit.mean(bmi).summary())    # weighted mean, SE that accounts for the tilt
fit.weights                       # or just take the weights
```

`holdout=` is the argument that matters. Constrained margins match by construction and
cannot refute anything; held-out register margins can, and are the only positive evidence
this framework produces. Omit it and `fit.summary()` says so rather than implying a pass.
Add `strata=` to calibrate prevalence *within* sex / birth year / ancestry / severity —
usually the most important step for psychiatric cohorts, because a known prevalence fixes
the number of cases and not their type.

Full walkthrough: [the guide](https://github.com/bvilhjal/i3pw/blob/main/docs/guide.md#if-you-have-a-cohort-start-here-calibrate).

## Quick start — the simulations

```python
import i3pw

# 1. Simulate a population and draw a biased sample.
ds = i3pw.make_dataset(
    seed=97,
    population_size=20000,
    n_features=15,
    n_outcomes=2,
    target_population_prevalence=(0.4, 0.05),
    target_sample_prevalence=(0.2, 0.005),   # expected margins under selection
    sample_size=4000,                        # expected Bernoulli participant count
)

print(ds.population_prevalence)   # truth we want to recover
print(ds.sample_prevalence)       # biased, naive estimate

# 2. Correct the bias by calibrating the weights to the known prevalences.
naive = i3pw.no_correction(ds)
cal   = i3pw.calibration_ipw(ds, base="lasso")   # `calibrate` with a Dataset around it

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

Fixed-seed output (8k population, one common + one rare outcome; runtime is
machine-dependent):

```
method                       % diff Y1   % diff Y2
--------------------------------------------------
no_correction                    49.48       91.82
lasso_ipw                        46.86       90.83
calibration_ipw                   0.00        0.00
--------------------------------------------------
```

**Read those two zeros carefully.** Every outcome here is anchored, so `calibration_ipw`
is reproducing prevalences it was *handed* — an algebraic identity, not a result. And
`lasso_ipw` looks useless because this simulation gives selection no covariate channel to
learn. `examples/honest_benchmark.py` measures the questions that can actually fail —
transfer to an *unanchored* outcome, and balance against quantities never supplied — and
[its section](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy) is
the one to trust.

## Where everything is

The rest lives in four documents, so that this page stays a page. Most readers need one
of them — and each one is the answer to one of the
[three questions](#the-research-question) above.

| | | |
| --- | --- | --- |
| [**docs/theory.md**](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md) | *question 2.* [notation](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#notation), what the method identifies, where the construction comes from, why the standard errors behave as they do, [what could prove it wrong](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#what-makes-this-falsifiable), and the [bibliography](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#references) | read before quoting a number in a paper |
| [**docs/guide.md**](https://github.com/bvilhjal/i3pw/blob/main/docs/guide.md) | *how to run it.* [`calibrate`, start to finish](https://github.com/bvilhjal/i3pw/blob/main/docs/guide.md#if-you-have-a-cohort-start-here-calibrate), the estimators, how to [check a weighting](https://github.com/bvilhjal/i3pw/blob/main/docs/guide.md#checking-the-weights-a-held-out-balance-diagnostic), how to [put error bars on it](https://github.com/bvilhjal/i3pw/blob/main/docs/guide.md#uncertainty) | read if you have a cohort |
| [**docs/studies.md**](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md) | *questions 1 and 3.* the simulations behind every number claimed here, starting with [the one that tries hardest to make the method look bad](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy) | read if you doubt a claim |
| [**report/i3pw_report.tex**](https://github.com/bvilhjal/i3pw/blob/main/report/i3pw_report.tex) | a self-contained mathematical review, implementation audit, fresh validation table, and cited bibliography for statistical geneticists | read before developing or reviewing the method |

**In a hurry?** [Conclusions and recommendations](#conclusions-and-recommendations), just
below, is the whole thing in a page: a five-step recipe, a list of things not to do, and
the reason for each — every reason linked to the evidence for it.

One warning that applies everywhere: several tables report a `0.00` or an exact match, and
*every one of those is an identity rather than a finding*. The interesting numbers are all
elsewhere.

## Conclusions and recommendations

All three documents, condensed to a page. Each recommendation is followed by the reason
for it, and the reason links to the section that establishes it — if you disagree with a
recommendation, the disagreement is really with the evidence behind it, so go there.

### The recipe

Steps 1–4 are one `calibrate` call — `base_weights=` for step 1's output, `targets=` for
step 2, `strata=` for step 3, `holdout=` for step 4 — so the reasons below are the reasons
for each argument.

1. **Fit a participation model on whatever predicts participation** — demographic,
   socioeconomic, clinical, genetic, or outcomes observed frame-wide — and invert it for
   base weights (`inverse_probability_weights(p_hat)` → `base_weights=`).
   *Why:* it corrects the part of selection that is visible in covariates, and it is the
   only ingredient that can. On a population where participation genuinely depends on a
   covariate, dropping this step costs a factor of 2.3 in held-out balance
   ([benchmark](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy)).
2. **Calibrate those weights to known register quantities** rather than trusting the
   model alone — `targets=`.
   *Why:* participation depends on *having the disease*, which the covariates barely
   proxy, so the covariate model alone leaves the ascertainment uncorrected. The two
   ingredients are complementary and the combination is best in **every** channel mix
   tested ([Study D](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#study-d--where-schoeler-et-al-fits-in-covariate-model-and-calibration-are-complementary)).
3. **Calibrate within strata — sex, birth year, ancestry, severity — not just the pooled
   margin** (`strata=` with `stratum_share=`).
   *Why:* a known prevalence fixes the *number* of cases, not their *type*. If the
   sampled cases are milder than the population's, matching the margin leaves that
   uncorrected ([case mix](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#prevalence-sets-the-scale-not-the-case-mix)). For psychiatric
   cohorts this is usually the single most important step. Watch the constraint count as
   the strata get finer — `A` strata by `Q` outcomes outgrows the cells backing them, and
   the solve reports success either way, so the package warns below ten units per
   constraint.
4. **Hold back some known margins and check against them** (`holdout=`, which returns a
   `balance_report`).
   *Why:* a just-identified calibration reproduces its constraints whatever the truth, so
   nothing it reports can refute it. Held-out moments are the only thing that can
   ([falsifiability](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#what-makes-this-falsifiable)) — and in a worked case every other
   diagnostic *preferred* the broken weighting, whose ESS looked 3× healthier
   ([demonstration](https://github.com/bvilhjal/i3pw/blob/main/docs/guide.md#checking-the-weights-a-held-out-balance-diagnostic)).
5. **Report the effective sample size, a sensitivity sweep over `K`, and an interval that
   accounts for the estimated weights** (`fit.mean(values)`, or the bootstrap).
   *Why:* the correction costs variance, register prevalences are not exact constants, and
   the fixed-weight SE is not a bound in either direction ([uncertainty](https://github.com/bvilhjal/i3pw/blob/main/docs/guide.md#uncertainty)).
   If you bootstrap, read its `failure_rate` first: discarded replicates are dropped
   selectively from the tail, so a non-zero rate means the interval is too narrow.

### What not to do

- **Do not use known prevalences to fix effect sizes.** Selection on the outcome alone
  does not bias a logistic slope (Prentice & Pyke 1979), so there is nothing to correct;
  and the bias that *does* matter — collider bias, from participation depending on
  exposure and outcome together — is untouched by calibration, which moved the estimate
  from 1.274 to 1.313 against a truth of 1.096. Only a participation model including the
  exposure recovers it
  ([evidence](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#participation-bias-and-effect-sizes-what-known-prevalences-cannot-fix)).
  Known prevalences are the right information for prevalences, absolute risks and means,
  and the wrong information for associations.
- **Do not expect calibrating on one disease to help with another.** Anchoring outcome 1
  left outcome 2 at 81.6% error against 80.8% for no correction at all. Marginal
  calibration fixes marginal quantities; anchor what you need corrected
  ([transfer](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy)).
- **Do not quote a `0.00` as evidence.** An anchored margin is reproduced by construction.
  This is worth repeating because the package's own headline table invites the mistake.
- **Do not reach for the analytic Lee-style product weight unless you know the mechanism
  is case-control on known-prevalence disorders.** It is excellent when selection really
  is one latent factor cleanly proxied by every outcome, and barely better than naive when
  a few correlated outcomes drive selection — a high-variance bet either way
  ([Study A](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#study-a--the-regime-decides-and-lee-style-weights-are-a-bet)).

### The one risk that outranks the others

Everything here assumes the population-to-sample density ratio lies in the tilt family —
that the base weights span the covariate-driven part of selection and the supplied moments
span the outcome-driven part. That assumption is not testable from the calibration itself,
and no robustness property covers it: if the moments miss the outcome-driven part, the
outcome model and the propensity model fail *together*
([what is identified](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#what-is-identified)).

The practical consequence is recommendation 4, which is why it is not optional. Passing a
held-out balance check is not proof, but it is the difference between an assumption and a
checked assumption, and it is the strongest positive evidence this framework currently
produces. The package reports a standardized-mean-difference diagnostic, not a formal
Sargan/Hansen overidentification test.

## Package layout

```
src/i3pw/
├── fit.py          # calibrate: the entry point for a real cohort (arrays in, weights,
│                   #   diagnostics and the held-out balance check out)
├── calibration.py  # the core: entropy_balance, calibration_ipw, apply_tilt,
│                   #   calibration_mean_se, stratified/joint calibration, diagnostics
├── balance.py      # balance_report: held-out SMD specification diagnostic
├── uncertainty.py  # bootstrap, fixed-weight SE, prevalence-sensitivity
├── aipw.py         # aipw_mean: augmented downstream estimation (+ cross-fitting)
├── liability.py    # probit / liability-threshold model: Lee et al. transform vs IPW
├── methods.py      # baselines: no_correction, lasso_ipw / lasso_propensity
├── dgm.py          # simulated population + Bernoulli-logistic participation
├── evaluation.py   # Monte Carlo comparison across many replications
├── metrics.py      # weighted (Hájek) prevalence, % difference
└── _links.py       # stable sigmoid / logit / probability clamp
docs/               # theory.md, guide.md, studies.md
tests/              # pytest suite
examples/           # real_cohort_workflow.py if you have a cohort;
                    #   honest_benchmark.py if you are judging the method
report/             # cited LaTeX review + frozen local validation results
CITATION.cff        # software citation metadata
CHANGELOG.md        # release-level scientific and API changes
```

## Tests

```bash
pytest

# If unrelated third-party pytest plugins in your environment interfere,
# disable plugin autoload:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## License

MIT — see [LICENSE](https://github.com/bvilhjal/i3pw/blob/main/LICENSE).

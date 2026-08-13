# i3pw — prevalence-calibrated density-ratio weighting

The import name is historical. The estimator is **minimum-divergence
reweighting that matches known population prevalences**, optionally on top of
a participation-model base. Those weights equal full-population inverse-probability
weights only under a stated density-ratio family. They are not unrestricted
inference of per-unit inclusion probabilities.

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

Fixed-seed illustration from
[`report/validation_results.tsv`](report/validation_results.tsv)
(`examples/benchmark.py`, seed 97, i3pw 0.3.0 freeze):

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
learn. The numbers that can fail — unanchored transfer and held-out balance — are the
`honest_benchmark.py` rows of the same TSV, narrated in
[studies.md](docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy).

## Where everything is

The rest lives in four documents, so that this page stays a page. Most readers need one
of them — and each one is the answer to one of the
[three questions](#the-research-question) above.

| | | |
| --- | --- | --- |
| [**docs/theory.md**](docs/theory.md) | *question 2.* [notation](docs/theory.md#notation), what the method identifies, where the construction comes from, why the standard errors behave as they do, [what could prove it wrong](docs/theory.md#what-makes-this-falsifiable), and the [bibliography](docs/theory.md#references) | read before quoting a number in a paper |
| [**docs/guide.md**](docs/guide.md) | *how to run it.* [`calibrate`, start to finish](docs/guide.md#if-you-have-a-cohort-start-here-calibrate), the estimators, how to [check a weighting](docs/guide.md#checking-the-weights-a-held-out-balance-diagnostic), how to [put error bars on it](docs/guide.md#uncertainty) | read if you have a cohort |
| [**docs/studies.md**](docs/studies.md) | *questions 1 and 3.* the simulations behind every number claimed here, starting with [what breaks and when](docs/studies.md#the-benchmark-suite-what-breaks-and-when) and [the honest benchmark](docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy). Numbers are copied from [`report/validation_results.tsv`](report/validation_results.tsv) and [`report/benchmark_results.tsv`](report/benchmark_results.tsv) | read if you doubt a claim |
| [**benchmarks/**](benchmarks/README.md) | *when does it fail?* seven benchmarks over the recruitment mechanism, the register information, the target error, the case mix, the support and the ridge — every one scored on estimands nobody was given, against oracle weights | read if you are judging the method |
| [**report PDF**](output/pdf/i3pw_report.pdf) ([LaTeX source](report/i3pw_report.tex)) | methods note: estimand, identification, and both layers of simulation evidence in full | read before quoting a number in a paper |

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
for each argument. Steps 5 and 6 are what to report afterwards.

1. **Fit a participation model on whatever predicts participation** — demographic,
   socioeconomic, clinical, genetic, or outcomes observed frame-wide — and invert it for
   base weights (`inverse_probability_weights(p_hat)` → `base_weights=`).
   *Why:* it corrects the part of selection that is visible in covariates, and it is the
   only ingredient that can. On a population where participation genuinely depends on a
   covariate, dropping this step costs a factor of 2.3 in held-out balance
   ([benchmark](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy)).
   *But it is not free.* Where recruitment acts through the diagnosis alone, a
   covariate base makes the estimate **11× worse** than a uniform one, because
   `1/P̂(S|X)` still varies with `X` — through `X → Y → S` — and imports a covariate
   tilt the outcome constraint cannot remove. The base weights are part of the
   specification, so omit them when you believe selection is outcome-driven
   ([benchmark suite](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#neither-ingredient-is-enough-and-the-base-model-is-not-free)).
2. **Calibrate those weights to known register quantities** rather than trusting the
   model alone — `targets=`.
   *Why:* participation depends on *having the disease*, which the covariates barely
   proxy, so the covariate model alone leaves the ascertainment uncorrected. The two
   ingredients are complementary and the combination is best in **every simulated**
   channel mix in Study D
   ([Study D](docs/studies.md#study-d--where-schoeler-et-al-fits-in-covariate-model-and-calibration-are-complementary)).
3. **Calibrate along the axis recruitment acts on — sex, birth year, ancestry,
   severity — not just the pooled margin** (`strata=` with `stratum_share=`, or
   severity-specific prevalences as extra outcome columns).
   *Why:* a known prevalence fixes the *number* of cases, not their *type*. If the
   sampled cases are milder than the population's, matching the margin leaves that
   uncorrected ([case mix](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#prevalence-sets-the-scale-not-the-case-mix)). Where cases are recruited
   unevenly *across strata*, a pooled margin leaves a within-stratum prevalence wrong by
   more than the prevalence itself (0.169 against `K = 0.10`) and stratifying nearly
   closes the held-out gap to the oracle. But **the wrong axis does not help**:
   demographic strata do nothing for severity-dependent recruitment, where separate
   mild- and severe-case prevalences cut the case-mix error 3.5×
   ([evidence](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#prevalence-fixes-the-case-count-not-the-case-mix--and-strata-are-not-a-cure-all)).
   Which axis is right is a claim about the recruitment mechanism, and the calibration
   cannot supply it. Watch the constraint count as the strata get finer — `A` strata by
   `Q` outcomes outgrows the cells backing them, and the solve reports success either
   way, so the package warns below ten units per constraint.
4. **Hold back some known margins and check against them** (`holdout=`, which returns a
   `balance_report`).
   *Why:* a just-identified calibration reproduces its constraints whatever the truth, so
   nothing it reports can refute it. Held-out moments are the only thing that can
   ([falsifiability](https://github.com/bvilhjal/i3pw/blob/main/docs/theory.md#what-makes-this-falsifiable)) — and in a worked case every other
   diagnostic *preferred* the broken weighting, whose ESS looked 3× healthier
   ([demonstration](https://github.com/bvilhjal/i3pw/blob/main/docs/guide.md#checking-the-weights-a-held-out-balance-diagnostic)).
   *Read it as an alarm, not a ranking.* Across the benchmark suite the held-out
   `|SMD|` correctly flagged one broken weighting and ranked two others backwards —
   preferring an estimator with 3× the bias — so a large discrepancy is evidence of
   misspecification, and a small one is not a reason to choose one weighting over
   another ([evidence](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#neither-ingredient-is-enough-and-the-base-model-is-not-free)).
5. **Report the effective sample size, a sensitivity sweep over `K`, and an interval that
   accounts for the estimated weights** (`fit.mean(values)`, or the bootstrap).
   *Why:* the correction costs variance, register prevalences are not exact constants, and
   the fixed-weight SE is not a bound in either direction ([uncertainty](https://github.com/bvilhjal/i3pw/blob/main/docs/guide.md#uncertainty)).
   Register error is the cheapest of the three: ±30% on `K` moved a held-out estimand by
   about 0.011 SD per 10 percentage points, against a 0.479 SD naive bias — but the
   sweep *bounds* the estimate and does not locate the truth inside the bound, since
   the smallest error in that sweep occurred at a 30% wrong target.
   If you bootstrap, read its `failure_rate` first: discarded replicates are dropped
   selectively from the tail, so a non-zero rate means the interval is too narrow. That
   begins at about **five sampled cases** — roughly an order of magnitude in prevalence
   before the point estimate stops solving
   ([support](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#a-wrong-register-prevalence-is-survivable-a-rare-disease-under-recruited-is-not)).
6. **Do not read the interval as evidence that the weighting is right.** At nominal 95%
   it covered 0.95 when the tilt family contained the truth and **0.64** under the
   ordinary combination of a fitted base and a marginal constraint — with almost no
   change in width, because the loss is bias and no variance formula covers a bias
   ([coverage](https://github.com/bvilhjal/i3pw/blob/main/docs/studies.md#the-intervals-cover-only-when-the-tilt-family-is-right)).
   An interval belongs next to the held-out check of step 4 and the sweep of step 5,
   never alone.

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
benchmarks/         # the evidence base: seven benchmarks over the report's
                    #   validation matrix, plus the generator for the report's
                    #   tables and figures. Not part of the installed package.
report/             # cited LaTeX methods note + both frozen results artifacts
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

## Benchmarks

The evidence base. Seven benchmarks over the recruitment mechanism, the register
information, the target error, the case mix, interval coverage, support and the ridge
— all seeds fixed, all scored on estimands the estimators were never given. See
[benchmarks/README.md](benchmarks/README.md).

```bash
python -m benchmarks.run_all          # rewrites report/benchmark_results.tsv (~17 min)
python -m benchmarks.make_figures     # rewrites the report's tables and figures
```

```bash
python -m benchmarks.run_all --quick
```

The quick run is a smoke test — far too noisy to quote — and writes to a separate
file so it cannot be mistaken for the freeze.

## License

MIT — see [LICENSE](https://github.com/bvilhjal/i3pw/blob/main/LICENSE).

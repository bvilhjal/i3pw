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
does and does not identify is made precise in [What is identified?](docs/theory.md#what-is-identified)
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
[its section](docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy) is
the one to trust.

## Where everything is

The rest lives in three documents, so that this page stays a page. Most readers need one
of them.

| | | |
| --- | --- | --- |
| [**docs/theory.md**](docs/theory.md) | what the method identifies, where the construction comes from, why the standard errors behave as they do, and [what could prove it wrong](docs/theory.md#what-makes-this-falsifiable) | read before quoting a number in a paper |
| [**docs/guide.md**](docs/guide.md) | the estimators, how to [check a weighting](docs/guide.md#checking-the-weights-balance-as-a-falsification-test), how to [put error bars on it](docs/guide.md#uncertainty) | read if you have a cohort |
| [**docs/studies.md**](docs/studies.md) | the simulations behind every number claimed here, starting with [the one that tries hardest to make the method look bad](docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy) | read if you doubt a claim |

**In a hurry?** [Conclusions and recommendations](#conclusions-and-recommendations), just
below, is the whole thing in a page: a five-step recipe, a list of things not to do, and
the reason for each — every reason linked to the evidence for it.

One warning that applies everywhere: several tables report a `0.00` or an exact match, and
*every one of those is an identity rather than a finding*. The interesting numbers are all
elsewhere.

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
   ([benchmark](docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy)).
2. **Calibrate those weights to known register quantities** rather than trusting the
   model alone — `entropy_balance(Y, targets, base_weights=1/P̂)`.
   *Why:* participation depends on *having the disease*, which the covariates barely
   proxy, so the covariate model alone leaves the ascertainment uncorrected. The two
   ingredients are complementary and the combination is best in **every** channel mix
   tested ([Study D](docs/studies.md#study-d--where-schoeler-et-al-fits-in-covariate-model-and-calibration-are-complementary)).
3. **Calibrate within strata — sex, birth year, ancestry, severity — not just the pooled
   margin** (`stratified_calibration_weights`).
   *Why:* a known prevalence fixes the *number* of cases, not their *type*. If the
   sampled cases are milder than the population's, matching the margin leaves that
   uncorrected ([case mix](docs/theory.md#prevalence-sets-the-scale-not-the-case-mix)). For psychiatric
   cohorts this is usually the single most important step.
4. **Hold back some known margins and test against them** (`balance_report`).
   *Why:* a just-identified calibration reproduces its constraints whatever the truth, so
   nothing it reports can refute it. Held-out moments are the only thing that can
   ([falsifiability](docs/theory.md#what-makes-this-falsifiable)) — and in a worked case every other
   diagnostic *preferred* the broken weighting, whose ESS looked 3× healthier
   ([demonstration](docs/guide.md#checking-the-weights-balance-as-a-falsification-test)).
5. **Report the effective sample size, a sensitivity sweep over `K`, and an interval that
   accounts for the estimated weights** (`calibration_mean_se`, or the bootstrap).
   *Why:* the correction costs variance, register prevalences are not exact constants, and
   the fixed-weight SE is not a bound in either direction ([uncertainty](docs/guide.md#uncertainty)).

### What not to do

- **Do not use known prevalences to fix effect sizes.** Selection on the outcome alone
  does not bias a logistic slope (Prentice & Pyke 1979), so there is nothing to correct;
  and the bias that *does* matter — collider bias, from participation depending on
  exposure and outcome together — is untouched by calibration, which moved the estimate
  from 1.274 to 1.313 against a truth of 1.096. Only a participation model including the
  exposure recovers it
  ([evidence](docs/studies.md#participation-bias-and-effect-sizes-what-known-prevalences-cannot-fix)).
  Known prevalences are the right information for prevalences, absolute risks and means,
  and the wrong information for associations.
- **Do not expect calibrating on one disease to help with another.** Anchoring outcome 1
  left outcome 2 at 77.2% error against 78.4% for no correction at all. Marginal
  calibration fixes marginal quantities; anchor what you need corrected
  ([transfer](docs/studies.md#what-the-headline-benchmark-does-not-show-exampleshonest_benchmarkpy)).
- **Do not quote a `0.00` as evidence.** An anchored margin is reproduced by construction.
  This is worth repeating because the package's own headline table invites the mistake.
- **Do not reach for the analytic Lee-style product weight unless you know the mechanism
  is case-control on known-prevalence disorders.** It is excellent when selection really
  is one latent factor cleanly proxied by every outcome, and barely better than naive when
  a few correlated outcomes drive selection — a high-variance bet either way
  ([Study A](docs/studies.md#study-a--the-regime-decides-and-lee-style-weights-are-a-bet)).

### The one risk that outranks the others

Everything here assumes the population-to-sample density ratio lies in the tilt family —
that the base weights span the covariate-driven part of selection and the supplied moments
span the outcome-driven part. That assumption is not testable from the calibration itself,
and no robustness property covers it: if the moments miss the outcome-driven part, the
outcome model and the propensity model fail *together*
([what is identified](docs/theory.md#what-is-identified)).

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
docs/               # theory.md, guide.md, studies.md
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

## License

MIT — see [LICENSE](LICENSE).

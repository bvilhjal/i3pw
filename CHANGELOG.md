# Changelog

## Unreleased

- Harden the calibration solve against optimizer stop-flag drift. For the
  unpenalized dual, `entropy_balance` now certifies convergence by the
  constraint residual — which is the dual gradient at the returned point —
  rather than by L-BFGS-B's `success` flag, whose line-search "ABNORMAL"
  termination fires at machine precision on some scipy versions (observed on
  1.18; the frozen artifacts were built on 1.17.1). Previously such a solve was
  reported as non-converged and the bootstrap discarded the replicate, a
  selective tail loss the module itself warns against;
  `test_bootstrap_anchored_outcome_has_near_zero_se` failed off the freeze
  environment for this reason and now asserts the discard *rate* is small
  instead of asserting zero discards outright.
- Reject NaN/inf inputs in `weighted_mean_se` and `weighted_prevalence`, the
  two estimators that lacked the finite-value guard the calibration module
  applies everywhere else.
- Validate that outcomes are 0/1 in the prevalence-constraint design builders
  (`calibrate`, `outcome_calibration_weights`,
  `stratified_calibration_weights`). A continuous column was silently accepted
  and could be falsely flagged unreachable; continuous calibration targets
  belong to `entropy_balance` directly, which makes no prevalence claims.
- Note in `similarity_matrix`'s docstring that it is dense O(n^2), intended for
  the package's simulation studies rather than biobank-scale cohorts.
- Add `benchmarks/`, a seven-benchmark evidence suite covering the report's
  validation matrix: recruitment mechanism, register information, target error,
  case mix, interval coverage, support, and the ridge. Every benchmark scores
  held-out estimands against oracle weights `1/π` and writes to one artifact,
  `report/benchmark_results.tsv`, with provenance in
  `report/benchmark_environment.txt`. The suite is not part of the installed
  package; `tests/test_benchmarks.py` holds its contract tests.
- Generate the report's data tables and figures from that artifact
  (`benchmarks/make_figures.py` → `report/figures/`), so a number in the PDF
  cannot drift from the table it came from. CI regenerates them and fails on a
  diff. Redraw every figure in one shared visual language
  (`report/figures/i3pw-viz.tex`): a fixed hue and marker per estimator, a
  colour-vision-validated palette, achromatic reference rows for the
  uncorrected and oracle weightings, and no constrained quantity plotted
  anywhere.
- Four documented findings change how the estimator should be used. Base
  weights are part of the specification, not a free improvement — under
  outcome-only recruitment a covariate base is 11× worse than a uniform one.
  Interval coverage is nominal only when the tilt family contains the truth and
  falls to 0.64 under ordinary applied misspecification, with no change in
  width. Stratification must follow the axis recruitment acts on; demographic
  strata do not repair severity-dependent recruitment. And the bootstrap starts
  discarding replicates about an order of magnitude in prevalence before the
  point estimate stops solving. README, `docs/studies.md` and the report are
  updated accordingly, including a sixth recommendation on how to read an
  interval.

- The package is named prevalence-calibrated density-ratio weighting. The
  import `i3pw` is unchanged. The methods PDF is recast as a research note
  (estimand, identification, frozen 0.3.0 simulation table); software review
  and the analysis protocol are appendices. README and studies copy numbers
  from `report/validation_results.tsv` rather than implying a 0.3.1 rerun.

## 0.3.1 — 2026-08-11

- Add a rendered PDF of the statistical-genetics report, with a vector workflow
  diagram and a three-panel summary of the frozen synthetic benchmarks.
- Correct report float ordering and pagination, and link the rendered report from
  the README.

## 0.3.0 — 2026-08-11

- Correct the interpretation of the calibration tilt and distinguish full-population
  inverse-probability weights from inverse odds, which target nonparticipants.
- Replace the misleading `oracle_odds` simulation baseline with `oracle_full`;
  the removed name now raises an explanatory error.
- Use Bernoulli-logistic participation in the simulator, with expected sample size and
  jointly calibrated expected outcome margins; calibrate outcome intercepts over realized
  covariates.
- Use the penalized influence adjustment for ridge calibration and stabilize calibration
  when base weights contain zeros or extreme finite values.
- Pin the LASSO solver random state so fixed simulation seeds reproduce validation results.
- Add the statistical-genetics LaTeX report, corrected documentation, citation metadata,
  frozen local validation results, and distribution-content checks.

Breaking changes: `SimConfig.sample_size` is now an expected Bernoulli count rather than
a fixed sample size; `weighting="oracle_odds"` has been removed; and
`calibration_ipw(base_scheme="odds")` now raises because that scheme targets
nonparticipants rather than the full population.

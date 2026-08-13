# Changelog

## Unreleased

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

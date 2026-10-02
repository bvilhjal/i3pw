# Changelog

## Unreleased

## 0.3.3 — 2026-10-02

- Fix the four mypy errors that had kept CI red — and pytest from running at all —
  since 2026-08-13: an `Optional` field `__post_init__` fills (`dgm.py`), and three
  branch-dependent `np.ndarray` bindings under shape-typed numpy stubs
  (`dgm.py`, `calibration.py` ×2). Type-checks clean against both the freeze numpy
  (2.4.6) and current (2.5.3) stubs. Cap the dev-extra linters by major
  (`mypy>=1.8,<3`, `ruff>=0.5,<2`): an unpinned mypy 2.x resolving from `>=1.8` is
  what silently broke the build.
- Correct the estimator labels the B2 and B8 writers record: B2's `none` rung is
  pure covariate IPW (`ipw`), its top rung is a stratified solve (`ipw+cal/s`), the
  oracle row is `oracle`, and each B8 row names its scenario's anchor
  (`ipw+cal` / `ipw+cal/s` / `ipw+cal/v`) — the frozen artifact labelled all of
  them `ipw+cal`, so slicing it by its estimator column misgrouped rows.
  `make_figures.py` now resolves each row's label from the artifact itself, so the
  displays build under either labeling. The committed freeze was deliberately NOT
  regenerated for this: it remains the 0.3.2 evidence of record that the report
  and docs cite, and the next deliberate re-freeze picks the labels up.
- Document, and measure, a limit of the freeze's exact-reproduction claim: a
  regeneration attempt under macOS 27.0.1 with identical package versions
  (2026-10-02) moved 845 of 894 rows — including rows computed from pure seeded
  draws — because the simulator's `multivariate_normal` runs through the
  Accelerate LAPACK and the OS upgrade perturbs the factorizations enough to flip
  the discrete draws built on top of them (a B6 tail solve rate nearly doubled).
  Exact reproduction is bound to the recorded platform, OS build included:
  `harness.environment` now records the BLAS/LAPACK backend and says so, and
  `benchmarks/README.md` documents re-freezing as a deliberate act that must
  resync every quoted number.
- `benchmarks/estimators.fit_weighting` now refuses `targets=` for `ipw+cal/s`
  (the stratified arm never consulted it, so a caller-supplied target was silently
  ignored) and drops a warning-capture block that could never catch anything
  (every inner solve runs with `warn=False`).
- Resynchronize every benchmark-suite number quoted in `README.md`,
  `docs/studies.md`, `docs/guide.md`, `docs/theory.md` and the report's prose with
  the current freeze. The markdown docs — and the report's B6 narrative — had been
  quoting the first (pre-Python-3.14) freeze while citing the current one; three
  claims had changed qualitatively: the outcome-only base-model harm is 19× (not
  11×), the under-recruited solve begins failing at ~5 sampled cases (it no longer
  "succeeds in every replication"), and the discard rate reaches 7.9% there (not
  3.4%). Corrected in prose everywhere, plus the report's oracle bound
  ("within 0.006 SD", was 0.001), its B8 uncertainty sentence (the ±0.02 envelope
  is the Monte Carlo interval, not the ±1 SD envelope), and the stale
  "Seven benchmarks"/"~17 min" counts.
- Correct this file's 0.3.2 entry, whose "reproduce byte-exactly" claim the two
  committed freezes falsify (~740 of 743 rows moved, from the
  residual-certification change and the Python 3.14 random streams) and whose
  quoted findings predated the final freeze.
- Add the missing literature to `docs/theory.md` and the report bibliography:
  Salvatore et al. 2024 (JAMIA, EHR-biobank weighting benchmarked against
  registry estimates), Kundu et al. 2024 (JRSS-A) and 2026 (Biostatistics,
  doubly robust multi-cohort), Pirastu et al. 2021 (Nature Genetics,
  sex-differential participation bias), Weissbrod et al. 2018 (AJHG, direct
  liability-scale estimation in case-control samples), Isaki & Fuller 1982
  (calibration asymptotics behind the linearization SE), and Elliott & Valliant
  2017 to the report. Candidate citations that could not be verified against
  Crossref/PubMed (a "PCGC-regression" paper, Chatterjee et al. 2005,
  Little & Vartivarian 2005) were deliberately left out rather than cited from
  memory.
- Tests: duplicate/rank-deficient calibration constraints, the `interactions=True`
  base-model path, and a doc-numbers guard that fails when a number quoted in the
  markdown docs stops matching the frozen artifact — the check whose absence let
  the stale-quote drift above ship.
- Annotate the historical R-prototype references in `dgm.py` (the scripts predate
  and are not shipped with this package); refresh the citation file with a release
  date; tag releases so cited versions are pinnable.

## 0.3.2 — 2026-09-04

- Single-source the version from `i3pw.__version__` (`dynamic = ["version"]` in
  `pyproject.toml`), removing the second copy that let the artifact's
  `i3pw_version` stamp drift from the changelog. Bump to 0.3.2.
- Recreate the evidence environment as `.condaenv` on Python 3.14.7, keeping the
  pinned freeze dependencies (NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0) and
  every seed unchanged. The rows still moved — Python 3.14's random streams differ
  from 3.11's, and the residual-certification change below alters which bootstrap
  replicates are kept — so the artifact was re-frozen: every value shifted within
  Monte Carlo error and no qualitative finding changed. (This entry originally
  claimed the rows would reproduce byte-exactly; the committed freezes falsify
  that, and 0.3.3 corrects the claim.) Extend the classifiers and CI matrix to
  3.13 and 3.14.
- Add `target_scale=` to `benchmarks/estimators.fit_weighting`: a common relative
  error on every prevalence-type target, leaving stratum shares and other
  demographic margins alone. It and `targets=` are mutually exclusive.
- Add benchmark B8 (`b8_k_error.py`): register error crossed with the anchor
  scenarios of B2/B4, sweeping a common relative error of ±20% through seven
  anchor-by-law combinations with the across-replication SD and Monte Carlo SE
  reported for every cell. The figure draws the 95% Monte Carlo interval of the
  mean and a ±1 SD envelope, so the anchor recommendations are read as
  differences with uncertainty, not point wins.
- Add benchmark B9 (`b9_wall_clock.py`): measured wall-clock seconds for one
  `calibration_ipw` fit, one B=200 bootstrap with the base held fixed, and one
  LASSO base fit, across N = 5k/20k/80k. It writes its own artifact,
  `report/timing_results.tsv`, because timings are not seed-reproducible, and it
  refuses to run on battery or in macOS Low Power Mode. The measured profile:
  the entropy dual is effectively free and nearly the entire cost of a fit is
  the LASSO base; 200 bootstrap re-solves add seconds, not minutes.
- `run_all.py` now runs B8 and B9 in every full and quick pass; B9 writes its own
  quick artifact. `harness.environment` names its artifact and records the
  reproducibility claim appropriate to it.
- `make_figures.py` gains `fig-k-error`, `tab-k-error`, and `tab-wall-clock`;
  the report's validation matrix and limits reflect B8 and B9.


- `liability_threshold` computes `t` as `-ppf(K)` rather than `ppf(1 - K)`. The
  identity is exact, so no realistic prevalence changes by even one bit; the old
  spelling lost digits below `K ~ 1e-9` and returned `+inf` for `K <= 1.1e-16`,
  which would have propagated through `z = 0` to an infinite factor in
  `lee_transform` with nothing in between checking finiteness.
- `lee_transform` documents that it is the Lee et al. (2011) leading factor and
  not the (2012) ascertainment-corrected form, gives the size of the gap
  (about 15% at `K=0.01, P=0.5, R2=0.2`), and points at
  `multipgs.metrics.liability_r2` for the 2012 transform. Behaviour is
  unchanged: the published benchmark comparison was run under the 2011 form.
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
  outcome-only recruitment a covariate base is 19× worse than a uniform one.
  Interval coverage is nominal only when the tilt family contains the truth and
  falls to 0.62–0.64 under ordinary applied misspecification, with no change in
  width. Stratification must follow the axis recruitment acts on; demographic
  strata do not repair severity-dependent recruitment. And the bootstrap starts
  discarding replicates before the solve itself begins to fail — discards begin
  around ten sampled cases and are substantial by five. README,
  `docs/studies.md` and the report are updated accordingly (in 0.3.3, to the
  final freeze these numbers come from), including a sixth recommendation on how
  to read an interval.

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

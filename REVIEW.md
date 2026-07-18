# i3pw Review — Idea, Theory, Documentation, Code

**Date:** 2026-07-18
**Scope:** full review at commit `e8bf2f8` — the scientific idea (prevalence-
calibrated IPW / density-ratio estimation), the theory in the README (the only
doc; 769 lines), the code (`src/i3pw/`, ~2500 lines), tests (68), and all nine
example scripts. Core modules (`calibration.py`, `aipw.py`, `liability.py`,
`methods.py`, `uncertainty.py`) were verified line-by-line firsthand; the rest
plus tests/examples via a full independent pass.
**Test suite:** 68 passed, zero warnings (Python 3.12, fresh conda env `i3pw`).
**Examples:** all nine re-run; every number quoted in the README reproduces to
the quoted digit.

## Verdict

The idea is sound and correctly circumscribed: known population prevalences
supply the outcome-driven term `θ·Y` of the selection log-odds that a
covariate-only participation model cannot learn, injected as calibration
constraints (entropy balancing) rather than into a propensity model. The
documentation is exceptionally honest about identification — the density-ratio /
minimum-divergence framing, the conditional coincidence with true IPW, the
"not doubly robust" disclaimer, the case-mix caution, and the collider boundary
are all precisely stated and correct. **No critical issues.** One real
user-facing defect (an example crashes on a default Windows console) and a
handful of polish-level nits.

**Findings: 1 major · 5 nit.**

---

## Major

### M1. `examples/ukb_participation.py` crashes on a default Windows console

`examples/ukb_participation.py:92,98,100` prints `β` and `δ`, which are not
encodable in cp1252 — the stdout encoding of a default (non-UTF-8-mode) Python
on a Western Windows install, under exactly the documented invocation
`python examples/ukb_participation.py`:

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u03b4'
```

The computation and numbers are correct (with `PYTHONIOENCODING=utf-8` the
README table reproduces exactly). Every other example was scanned — this is the
only script with non-cp1252 characters on a print path (`±`, `—` are
cp1252-encodable; Greek letters elsewhere appear only in docstrings).

**Fix:** print ASCII (`beta`, `delta_E`), or add
`sys.stdout.reconfigure(encoding="utf-8")` under a `hasattr` guard in `main()`.

---

## Nits

| # | Location | Issue | Suggested fix |
|---|----------|-------|---------------|
| n1 | `src/i3pw/metrics.py:12` | `weighted_prevalence` silently broadcasts mismatched shapes: `weights (n,)` × `y (n,1)` → `(n,n)` and a wrong number instead of raising. Internal callers pass 1-D slices, so no live bug | shape-check or `ravel` both inputs |
| n2 | `src/i3pw/dgm.py:134` | `Dataset.split("trian")` (typo) silently returns the test fold — any `which != "train"` is treated as test | `if which not in ("train", "test"): raise ValueError` |
| n3 | `src/i3pw/dgm.py:204` | realized sample prevalence drifts from `target_sample_prevalence` for multi-outcome draws (product of per-outcome odds weights + without-replacement sampling; mean relative deviation +4–11% on middle outcomes over 10 seeds). The name "target" is honest and no README number depends on exactness | docstring note, or a test asserting closeness |
| n4 | `src/i3pw/evaluation.py:92` | `format_summary({})` raises bare `StopIteration` | raise a clean `ValueError` |
| n5 | `src/i3pw/dgm.py:28` | `nearest_pd_correlation` docstring says "the nearest PD correlation matrix"; eigenvalue-clipping + rescale yields *a* nearby PD matrix, not the Frobenius-nearest | wording only |

---

## Verified sound

### Idea and theory (verified firsthand, line-by-line)

- **The core claim** (README "The problem, and the idea"): under selection
  log-odds `a(X) + θ·Y`, a covariate-only model learns `a(X)` but not `θ·Y` —
  correct, and the calibration constraint supplies exactly the missing term.
- **Identification framing** ("What is identified?"): calibration returns the
  minimum-divergence weights `d(X)·exp(λ·g(Y))` matching the supplied moments;
  they equal true IPW only when the population-to-sample density ratio lies in
  the tilt family + positivity. Precise and correct, including the
  weights ∝ `1/π` relationship to the density ratio.
- **Inverse-odds vs inverse-probability separability**: under logistic
  participation, `(1−π)/π = exp(−a(X)−θY)` is exactly log-linear and composes
  with the tilt; `1/π = 1 + exp(−a−θY)` is not — verified algebraically; the
  rare-inclusion limit statement is correct, and the "selection on the outcome
  alone ⇒ per-class constant weights ⇒ K/P exact" claim is correct.
- **Entropy balancing** (`calibration.py`): the dual
  `min_λ log Σ d_i·exp(λ·(f_i − t)) + (ridge/2)||λ||²` with weights
  `w ∝ d·exp(λ·(f − t))` is the standard convex form; constraints, centering,
  and normalization are correct; infeasibility/convergence warnings are
  appropriate; the stratified constraint set (`E_w[1(A=a)] = P(A=a)` with the
  last stratum dropped against singularity, plus `E_w[Y_q·1(A=a)] =
  P(Y_q=1, A=a)`) is exactly right; co-occurrence constraints via products are
  right; trim-broke-calibration is detected and warned.
- **AIPW** (`aipw.py`): `estimate = mean_X m(X) + Σw(V − m(X))` with
  self-normalized `w` is the standard Hájek-AIPW for a population mean; the
  doubly-robust and cross-fit caveats in the docstrings match the
  implementation (out-of-fold residuals + fold-averaged plug-in).
- **Liability model** (`liability.py`): Lee et al. (2011) transform
  `r2_obs·[K(1−K)/z²]·[K(1−K)/(P(1−P))]` and the population
  observed→liability factor are the standard forms; the Haseman–Elston-type
  moment slope (pairwise products on similarities, diagonal excluded, pair
  weights `w_i w_j`) is correct; the IPW route (reweight to `K`, then
  population transform only) is the correct design-based counterpart; the
  δ=0/delta>0 boundary between Lee and IPW is correctly characterized.
- **Uncertainty** (`uncertainty.py`): `weighted_mean_se` is the correct
  design-based linearization variance for a Hájek mean, honestly documented as
  a fixed-weights lower bound; the bootstrap re-solves calibration per
  replicate (and refits the base with `refit_base=True`); the "anchored SE ≈ 0
  by construction" read is honest.
- **Prentice & Pyke usage**: outcome-only ascertainment leaves logistic slopes
  unbiased — correctly cited in both the effect-size section and the AIPW demo
  writeup; the collider analysis (marginal calibration cannot fix joint
  exposure–outcome selection) is correct and demonstrated.
- **Joint vs marginal calibration**: `log π(y)` linear in `y` ⇒ `Q` marginals
  identify the selection (marginal calibration = oracle even on joint
  targets); an interaction needs the co-occurrence moment — correct, and
  demonstrated in `multi_outcome_calibration.py`.
- **LASSO propensity** (`methods.py`): a faithful `cv.glmnet` analogue
  (standardize, L1 logistic, CV on neg-log-loss); the `inverse` vs
  `oracle_odds` weighting distinction is correctly drawn (deployable vs
  simulation-only oracle).
- **DGM** (`dgm.py`): the biased-sampling construction is exactly
  target-achieving in expectation for one outcome; train/test partition is
  disjoint and exhaustive; RNG handling is reproducible.

### Documentation

- **Every quoted number reproduces**: benchmark (48.25/91.32…), monte_carlo
  (46.81±5.70…, ESS 155±32), doubly_robust (−0.096/0.101…), schoeler_plus
  (0.512/2.478/3.747/0.514/0.508), ukb_participation (1.096; 1.094/1.274;
  1.107/1.313), multi_outcome (every cell), probit lee_vs_ipw (12.21/0.483/
  0.494…), complex_selection (δ table), and all four
  selection_inference_extensive studies (A–D), including the qualitative
  monotonicity claims.
- **Claims hygiene**: "anchored margins are exact by construction ... not
  evidence the method works", the case-mix ("right count, wrong mix") caution,
  the ESS/variance cost, feasibility limits, and the effect-size boundary are
  all stated where they belong and match the code's behavior.
- **README API**: every public name used in the README exists in
  `i3pw.__init__.__all__`; both code blocks run verbatim.

### Tests and packaging

- **68 passed, zero warnings** (also clean under `-W error::Warning`).
- Tests are property tests, not smoke: exact calibration to targets, infeasible
  warnings, ridge shrinkage direction, stratified and co-occurrence constraints,
  AIPW double robustness in both directions + cross-fit determinism, bootstrap
  anchored-SE≈0, sensitivity tracking, Lee↔population identity at `P=K`,
  Lee/IPW divergence at `δ>0`, joint-vs-marginal vs oracle at `g=1`/`g=2.5`.
- `pyproject.toml` matches the README (Python ≥3.10; numpy/scipy/scikit-learn;
  src-layout; version synced with `__init__`).
- Coverage gaps (minor): `format_summary` rendering, the ESS 155±32 claim,
  AIPW on `calibration_ipw` weights end-to-end, DGM sample-prevalence closeness
  (n3) — none hides a defect.

---

## Recommended actions (priority order)

1. **M1** — ASCII prints or stdout reconfigure in `examples/ukb_participation.py`.
2. n1, n2 — one-line shape/argument guards in `metrics.py` and `dgm.py`.
3. n3 — docstring note on the multi-outcome sample-prevalence drift.
4. n4, n5 — tidy at leisure.

---

## Resolution (2026-07-18)

All findings fixed in the working tree. Test suite after the changes: **72
passed** (68 + 4 new), zero warnings.

- **M1** — `examples/ukb_participation.py` prints ASCII (`beta`, `delta_E`);
  verified to run cleanly on a default (cp1252) console and still reproduce the
  README table exactly.
- **n1** — `weighted_prevalence` validates equal length after raveling (column
  vectors accepted, genuine mismatches raise), with a regression test.
- **n2** — `Dataset.split` rejects any `which` outside `{"train", "test"}`,
  with a regression test.
- **n3** — `_induce_selection` docstring now notes the multi-outcome
  sample-prevalence drift (~4-10% relative) and the "target" naming.
- **n4** — `format_summary({})` raises a clean `ValueError`, with tests
  (new `tests/test_evaluation.py`, also covering table rendering).
- **n5** — `nearest_pd_correlation` docstring no longer claims the
  Frobenius-nearest projection.

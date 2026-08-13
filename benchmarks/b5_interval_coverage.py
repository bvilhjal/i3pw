"""B5 --- do the intervals cover?

The report lists calibrated interval coverage among the things its evidence base
does *not* establish. This benchmark establishes it for the simulated case, which is
the only place it can be established at all: coverage is a statement about repeated
sampling from a known truth.

Three interval procedures for the population mean of a trait observed only in
participants, all at nominal 95%:

``fixed-weight``       ``i3pw.weighted_mean_se`` -- conditions on the weights as if
                       they had been handed down rather than estimated.
``calibration-aware``  ``i3pw.calibration_mean_se`` -- the GREG linearization, which
                       subtracts the part of the trait the constrained moments
                       explain.
``bootstrap``          resample the participants and re-solve the calibration each
                       replicate, percentile interval. The base weights are carried
                       along with the resampled rows but the participation model is
                       not refit, so this covers the calibration's variability and
                       not the base model's.

and three specification regimes, because coverage is a property of the pair
(procedure, specification) and not of the procedure alone:

``correct``       outcome-only recruitment, uniform base, calibrated on the outcome.
                  For a binary outcome the tilt family contains the truth exactly, so
                  the estimator is consistent and coverage is a clean test of the
                  variance formula.
``base + margin`` additive covariate and outcome channels with a fitted base. The
                  usual applied situation, and mildly misspecified: the log density
                  ratio of a logistic participation model is not additive in the
                  covariate index and the outcome even when the logit is.
``misspecified``  a covariate-by-outcome interaction in recruitment, which
                  base-plus-marginal cannot represent at all.

An interval covers the estimator's own limit, not the truth. Where the two differ,
the honest reading of a coverage number is a measure of the bias, and this benchmark
reports the bias next to the coverage so the two cannot be separated.
"""

from __future__ import annotations

import warnings

import numpy as np

from benchmarks import estimators as E
from benchmarks.harness import Progress, Row, proportion, summarize
from benchmarks.simulate import SELECTION_LAWS, Design, simulate
from i3pw import calibration_mean_se, entropy_balance, weighted_mean_se

BENCHMARK = "B5_interval_coverage"

REGIMES = {
    "correct": dict(law="Y only", method="cal"),
    "base + margin": dict(law="X + Y", method="ipw+cal"),
    "misspecified": dict(law="X x Y", method="ipw+cal"),
}
PROCEDURES = ("fixed-weight", "calibration-aware", "bootstrap")


def _bootstrap_interval(
    values: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    base: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile interval and discard rate from re-solving on resampled participants."""
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    estimates, failures = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w, diag = entropy_balance(
                features[idx], targets, base_weights=base[idx],
                warn=False, return_diagnostics=True,
            )
        if not diag.converged or diag.max_abs_residual > 1e-6:
            failures += 1
            continue
        estimates.append(float(np.sum(w * values[idx]) / np.sum(w)))
    if len(estimates) < 2:
        return float("nan"), float("nan"), failures / n_boot
    alpha = 1.0 - level
    lo, hi = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi), failures / n_boot


def run(n_reps: int = 300, n_boot: int = 400) -> list[Row]:
    rows: list[Row] = []
    progress = Progress(BENCHMARK, len(REGIMES) * n_reps)
    for regime, spec in REGIMES.items():
        law, method = spec["law"], spec["method"]
        covered = {p: [] for p in PROCEDURES}
        width = {p: [] for p in PROCEDURES}
        bias, discard = [], []
        for seed in range(n_reps):
            pop = simulate(Design(seed=seed, **SELECTION_LAWS[law]))
            mask = pop.mask
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_hat = E.propensity(pop) if method.startswith("ipw") else None
                wt = E.fit_weighting(pop, method, p_hat=p_hat)
            z = pop.Z[mask]
            # The estimand is on the raw trait scale; the SD divides only the report.
            truth, scale = pop.trait_mean, pop.trait_sd
            bias.append((E.weighted_mean(z, wt.weights) - truth) / scale)

            fixed = weighted_mean_se(z, wt.weights)
            aware = calibration_mean_se(z, wt.weights, wt.features)
            base = np.ones(mask.sum()) if p_hat is None else 1.0 / p_hat
            lo, hi, failed = _bootstrap_interval(
                z, wt.features, wt.targets, base, n_boot=n_boot, seed=10_000 + seed,
            )
            discard.append(failed)

            for name, (low, high) in (
                ("fixed-weight", (fixed.ci_low, fixed.ci_high)),
                ("calibration-aware", (aware.ci_low, aware.ci_high)),
                ("bootstrap", (lo, hi)),
            ):
                if not np.isfinite(low) or not np.isfinite(high):
                    continue
                covered[name].append(bool(low <= truth <= high))
                width[name].append((high - low) / scale)
            progress.step()

        for name in PROCEDURES:
            rows.append(proportion(BENCHMARK, regime, name, "coverage_95", covered[name],
                                   notes="nominal 0.95; Monte Carlo error is binomial"))
            rows.append(summarize(BENCHMARK, regime, name, "interval_width_sd",
                                  width[name], notes="width in population SD units"))
        rows.append(summarize(BENCHMARK, regime, "estimator", "trait_bias_sd", bias,
                              notes="the bias the intervals are being asked to cover"))
        rows.append(summarize(BENCHMARK, regime, "bootstrap", "discard_rate", discard,
                              notes="replicates whose calibration did not solve"))
    progress.close()
    return rows


def main() -> None:
    for row in run(n_reps=20, n_boot=100):
        print(row.as_tsv())


if __name__ == "__main__":
    main()

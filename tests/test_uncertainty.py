import warnings

import numpy as np
import pytest

import i3pw
from i3pw import (
    bootstrap_calibration_ipw,
    make_dataset,
    prevalence_sensitivity,
    weighted_mean_se,
)


@pytest.fixture(scope="module")
def dataset():
    return make_dataset(
        seed=11, population_size=6000, n_features=12, n_outcomes=2,
        predictors_per_outcome=6,
        target_population_prevalence=(0.4, 0.08),
        target_sample_prevalence=(0.2, 0.02), sample_size=1500,
    )


def test_weighted_mean_se_matches_closed_form_for_uniform_weights():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=500) < 0.3).astype(float)
    w = np.ones(500)
    est = weighted_mean_se(y, w)
    mu = y.mean()
    se_expected = np.sqrt(np.sum((y - mu) ** 2)) / len(y)
    assert est.value == pytest.approx(mu)
    assert est.se == pytest.approx(se_expected)
    assert est.ci_low < est.value < est.ci_high


def test_weighted_mean_se_validates():
    with pytest.raises(ValueError):
        weighted_mean_se(np.zeros(3), np.ones(4))
    with pytest.raises(ValueError):
        weighted_mean_se(np.zeros(3), -np.ones(3))


def test_bootstrap_anchored_outcome_has_near_zero_se(dataset):
    r = bootstrap_calibration_ipw(dataset, anchor_outcomes=[0], base="uniform",
                                  n_boot=60, seed=1)
    # Outcome 0 is common, so no resample should fail to calibrate — but do not
    # hard-code zero discards: optimizer stop-flag reporting varies across scipy
    # versions, and a rare discard is the module's documented behavior, not a bug.
    assert r.failure_rate <= 0.05
    assert r.replicates.shape[1] == 2
    # Outcome 0 is calibrated to its target in every replicate -> essentially no spread.
    assert r.se[0] < 1e-4
    # Outcome 1 is left free -> genuine sampling variability.
    assert r.se[1] > r.se[0]
    assert r.ci_low[1] <= r.estimate[1] <= r.ci_high[1]


def test_bootstrap_refit_base_runs(dataset):
    # The heavier path (refits the LASSO base each replicate) must run and be finite.
    r = bootstrap_calibration_ipw(dataset, anchor_outcomes=[0], base="lasso",
                                  n_boot=8, refit_base=True, seed=2)
    assert np.all(np.isfinite(r.se))
    assert r.se[0] < 1e-3


def test_prevalence_sensitivity_tracks_anchored_target(dataset):
    deltas = (-0.1, 0.0, 0.1)
    r = prevalence_sensitivity(dataset, anchor_outcomes=[0], base="uniform", rel_deltas=deltas)
    pop0 = dataset.population_prevalence[0]
    for i, d in enumerate(deltas):
        assert r.estimates[i, 0] == pytest.approx(pop0 * (1.0 + d), abs=1e-3)
    # The anchored outcome's spread is ~ pop0 * (range of deltas).
    assert r.spread[0] == pytest.approx(pop0 * 0.2, abs=2e-3)
    assert np.all(r.ess > 0)
    assert "sensitivity" in r.summary()


def test_population_uncertainty_wrappers_reject_inverse_odds(dataset):
    with pytest.raises(ValueError, match="nonparticipants"):
        bootstrap_calibration_ipw(dataset, base_scheme="odds", n_boot=2)
    with pytest.raises(ValueError, match="nonparticipants"):
        prevalence_sensitivity(dataset, base_scheme="odds")


def test_bootstrap_discards_uncalibrated_replicates():
    """Replicates whose calibration cannot meet the targets must not enter the interval.

    A resample containing no cases of a rare anchored outcome puts the target outside
    its convex hull; the tilt then runs to a corner. Those replicates used to be
    recorded unconditionally, widening (and skewing) the interval — including for
    *other*, well-supported outcomes sharing the solve.
    """
    ds = i3pw.make_dataset(
        seed=9, population_size=6000, n_features=10, n_outcomes=3,
        predictors_per_outcome=5,
        target_population_prevalence=(0.40, 0.15, 0.06),
        # This seed leaves one rare-outcome case in the selected test fold: the full
        # estimator is defined, while many bootstrap resamples still omit that case.
        target_sample_prevalence=(0.20, 0.04, 0.010), sample_size=1200,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = i3pw.bootstrap_calibration_ipw(ds, base="uniform", n_boot=60, seed=1)

    # Accounting is exposed rather than hidden.
    assert res.n_boot == 60
    assert res.n_failed == res.n_boot - res.replicates.shape[0]
    assert 0.0 <= res.failure_rate <= 1.0
    # Every surviving replicate actually hit each anchored target.
    for a in res.anchor_outcomes:
        target = ds.population_prevalence[a]
        assert np.allclose(res.replicates[:, a], target, atol=1e-5)
    # No surviving replicate is a degenerate corner solution.
    assert not np.any((res.replicates <= 1e-9) | (res.replicates >= 1.0 - 1e-9))


def test_bootstrap_reports_failures_in_summary_and_warns():
    """A discarded replicate is warned about and shown in the summary, never silent."""
    rng = np.random.default_rng(0)
    n = 400
    Y = np.zeros((n, 1))
    Y[:3, 0] = 1.0  # 3 cases: many resamples will contain none

    class _Stub:  # minimal duck-typed stand-in for a Dataset
        population_prevalence = np.array([0.10])

        def split(self, which):
            X = rng.standard_normal((n, 2))
            return X, Y, np.ones(n, dtype=int)

    with pytest.warns(i3pw.CalibrationWarning, match="discarded"):
        res = i3pw.bootstrap_calibration_ipw(_Stub(), base="uniform", n_boot=40, seed=3)
    assert res.n_failed > 0
    assert "discarded" in res.summary()
    # The count alone reads as bookkeeping. Discarding is selective — it drops the
    # resamples poorest in rare cases, i.e. the tail — so the summary has to say that
    # the interval is narrower than the truth, not merely that replicates went missing.
    assert "too NARROW" in res.summary()
    assert res.failure_rate > 0

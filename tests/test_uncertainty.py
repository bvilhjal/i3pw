import numpy as np
import pytest

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
    assert r.replicates.shape == (60, 2)
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

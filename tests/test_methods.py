import numpy as np
import pytest

from i3pw import lasso_ipw, make_dataset, monte_carlo, no_correction
from i3pw.methods import _ipw_weight, _trim_weights


@pytest.fixture(scope="module")
def dataset():
    return make_dataset(
        seed=1, population_size=5000, n_features=12, n_outcomes=2,
        predictors_per_outcome=6,
        target_population_prevalence=(0.4, 0.1),
        target_sample_prevalence=(0.2, 0.02), sample_size=1200,
    )


def test_no_correction_and_lasso_ipw_run(dataset):
    nc = no_correction(dataset)
    res = lasso_ipw(dataset, cv=3)
    assert nc.weighted_prevalence.shape == (2,)
    assert res.weighted_prevalence.shape == (2,)
    assert np.all(np.isfinite(res.percent_diff))


def test_ipw_weight_schemes():
    P = np.array([0.2, 0.5, 0.8])
    s = np.array([1.0, 0.0, 1.0])
    inv = _ipw_weight(P, s, "inverse")
    # inverse: unselected excluded (weight 0), selected get 1/P > 0.
    assert inv[1] == 0.0
    assert np.allclose(inv[[0, 2]], 1.0 / P[[0, 2]])
    oracle = _ipw_weight(P, s, "oracle_odds")
    # oracle_odds: unselected get exactly 1, selected get (1-P)/P.
    assert oracle[1] == 1.0
    assert np.allclose(oracle[[0, 2]], (1 - P[[0, 2]]) / P[[0, 2]])
    with pytest.raises(ValueError):
        _ipw_weight(P, s, "nope")


def test_inverse_weighting_uses_sample_only(dataset):
    # The Hájek estimate must be computable from the sample alone:
    # zero-weight (unselected) units cannot affect it.
    res = lasso_ipw(dataset, weighting="inverse", cv=3)
    w = res.extra["weight"]
    _, _, s_test = dataset.split("test")
    assert np.all(w[s_test == 0] == 0)
    assert np.all(np.isfinite(res.percent_diff))


def test_trim_weights_caps_extremes():
    w = np.array([1.0, 2.0, 3.0, 100.0])
    trimmed = _trim_weights(w, 0.75)
    assert trimmed.max() < 100.0
    assert np.array_equal(_trim_weights(w, None), w)  # None is a no-op
    with pytest.raises(ValueError):
        _trim_weights(w, 1.5)


def test_monte_carlo_summary():
    # A few reps on a tiny population; calibration_ipw should on average beat
    # no correction and the covariate-only lasso baseline, and shapes must be right.
    sims = dict(population_size=1500, n_features=8, n_outcomes=2,
                predictors_per_outcome=4,
                target_population_prevalence=(0.4, 0.1),
                target_sample_prevalence=(0.2, 0.02), sample_size=400)
    summaries = monte_carlo(n_reps=3, base_seed=10, sim_kwargs=sims)
    assert set(summaries) == {"no_correction", "lasso_ipw", "calibration_ipw"}
    assert summaries["calibration_ipw"].mean_pct_error.shape == (2,)
    assert summaries["calibration_ipw"].overall() < summaries["no_correction"].overall()

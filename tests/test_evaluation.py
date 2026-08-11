"""Tests for the Monte Carlo evaluation harness."""

import numpy as np
import pytest

import i3pw.methods
from i3pw.evaluation import MonteCarloSummary, format_summary, monte_carlo


def test_format_summary_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        format_summary({})


def test_format_summary_renders_table():
    s = MonteCarloSummary(method="m1", mean_pct_error=np.array([1.0, 2.0]),
                          sd_pct_error=np.array([0.1, 0.2]), n_reps=3)
    out = format_summary({"m1": s})
    assert "m1" in out
    assert "Y1 %err" in out
    assert "1.00" in out


def test_monte_carlo_nan_errors_are_excluded_not_propagated(monkeypatch):
    # A replication whose realised population prevalence is 0 yields a NaN percent
    # difference; one such value must not turn the whole column NaN.
    real = i3pw.methods.percent_difference
    calls = {"n": 0}

    def nan_once(estimate, truth):
        calls["n"] += 1
        return np.nan if calls["n"] == 1 else real(estimate, truth)

    monkeypatch.setattr(i3pw.methods, "percent_difference", nan_once)
    sims = dict(population_size=1500, n_features=8, n_outcomes=2,
                predictors_per_outcome=4,
                target_population_prevalence=(0.4, 0.1),
                target_sample_prevalence=(0.2, 0.02), sample_size=400)
    with pytest.warns(UserWarning, match="NaN"):
        summaries = monte_carlo(n_reps=2, base_seed=10, sim_kwargs=sims,
                                include_lasso=False, include_calibration=False)
    s = summaries["no_correction"]
    assert s.n_nan == 1
    assert np.all(np.isfinite(s.mean_pct_error))

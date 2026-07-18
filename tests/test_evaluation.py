"""Tests for the Monte Carlo evaluation harness."""

import numpy as np
import pytest

from i3pw.evaluation import MonteCarloSummary, format_summary


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

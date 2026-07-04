import numpy as np
import pytest

from i3pw.weights import combine_weights


def test_mean_and_product():
    w = np.array([[2.0, 4.0], [1.0, 3.0]])
    assert np.allclose(combine_weights(w, "mean"), [3.0, 2.0])
    assert np.allclose(combine_weights(w, "product"), [8.0, 3.0])


def test_harmonic_matches_two_outcome_formula():
    # For Q=2 the harmonic mean must equal 2 w1 w2 / (w1 + w2), as in the R code.
    w = np.array([[2.0, 6.0], [1.0, 4.0]])
    expected = 2 * w[:, 0] * w[:, 1] / (w[:, 0] + w[:, 1])
    assert np.allclose(combine_weights(w, "harmonic"), expected)


def test_absdiff_weights_toward_better_calibrated_outcome():
    w = np.array([[10.0, 1.0]])
    pop = np.array([0.4, 0.1])
    sample = np.array([0.39, 0.0])  # outcome 0 nearly unbiased -> higher affinity
    out = combine_weights(w, "absdiff", pop_prevalence=pop, sample_prevalence=sample)
    # Affinities: 1/0.01=100 vs 1/0.1=10 -> heavily favors column 0's weight (10).
    assert out[0] > 9.0


def test_absdiff_requires_prevalences():
    with pytest.raises(ValueError):
        combine_weights(np.ones((2, 2)), "absdiff")


def test_unknown_method():
    with pytest.raises(ValueError):
        combine_weights(np.ones((2, 2)), "nope")

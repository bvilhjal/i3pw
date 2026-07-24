import numpy as np
import pytest

from i3pw import logit, sigmoid
from i3pw._links import EPS, clip_prob
from i3pw.metrics import percent_difference, weighted_prevalence


def test_clip_prob_bounds_and_passthrough():
    # 0/1 are clamped into the open interval; interior values pass through.
    out = clip_prob(np.array([0.0, 0.5, 1.0]))
    assert out[0] == pytest.approx(EPS)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0 - EPS)


def test_logit_uses_the_shared_clamp():
    # logit(0)/logit(1) are finite because they share clip_prob's epsilon clamp.
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()


def test_sigmoid_logit_inverse():
    p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    assert np.allclose(sigmoid(logit(p)), p, atol=1e-9)


def test_sigmoid_stable_extremes():
    # No overflow warnings / infs for large magnitudes.
    z = np.array([-1000.0, 1000.0])
    out = sigmoid(z)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[1] == pytest.approx(1.0, abs=1e-12)


def test_weighted_prevalence_equal_weights_is_mean():
    y = np.array([0, 1, 1, 0, 1])
    w = np.ones(5)
    assert weighted_prevalence(w, y) == pytest.approx(y.mean())


def test_weighted_prevalence_reweights():
    y = np.array([1, 0])
    w = np.array([3.0, 1.0])
    assert weighted_prevalence(w, y) == pytest.approx(0.75)


def test_percent_difference():
    assert percent_difference(0.2, 0.4) == pytest.approx(50.0)
    assert np.isnan(percent_difference(0.1, 0.0))


def test_zero_weight_sum_raises():
    with pytest.raises(ValueError):
        weighted_prevalence(np.zeros(3), np.ones(3))


def test_weighted_prevalence_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        weighted_prevalence(np.ones(3), np.ones(4))
    # (n, 1) column vectors are accepted gracefully
    col = np.array([[1.0], [0.0], [1.0]])
    assert weighted_prevalence(np.ones(3), col) == pytest.approx(2 / 3)

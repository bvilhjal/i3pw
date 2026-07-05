import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from i3pw import aipw_mean


def _mar_scenario(seed, n=6000, p=5):
    """Missing-at-random data: V observed when S=1, S depends on X only. E[V]=0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    b = rng.normal(size=p)
    V = X @ b + rng.normal(size=n)          # truth E[V] = 0
    c = rng.normal(size=p)                   # strong, covariate-driven selection
    Ps = 1.0 / (1.0 + np.exp(-(X @ c)))
    S = rng.uniform(size=n) < Ps
    return X, S, V[S], 1.0 / Ps[S]           # X_all, mask, V_sample, true IPW weights


def test_double_robustness_over_seeds():
    naive, both, wrong_w, wrong_m = [], [], [], []
    for seed in range(5):
        X, S, Vs, w_true = _mar_scenario(seed)
        w_unif = np.ones(S.sum())
        naive.append(abs(Vs.mean()))
        both.append(aipw_mean(X, S, Vs, w_true, outcome_model=Ridge(1.0), truth=0.0).error)
        # correct model, WRONG (uniform) weights -> still consistent
        wrong_w.append(aipw_mean(X, S, Vs, w_unif, outcome_model=Ridge(1.0), truth=0.0).error)
        # correct weights, WRONG (near-constant) model -> still consistent
        wrong_m.append(aipw_mean(X, S, Vs, w_true, outcome_model=Ridge(1e12), truth=0.0).error)

    assert np.mean(naive) > 0.2                 # naive is badly biased
    assert np.mean(both) < 0.12
    assert np.mean(wrong_w) < 0.12              # robust to wrong weights
    assert np.mean(wrong_m) < 0.15              # robust to wrong outcome model


def test_constant_model_reduces_to_ipw():
    # A constant outcome model must make AIPW collapse to the Hájek IPW estimate.
    X, S, Vs, w = _mar_scenario(0)
    res = aipw_mean(X, S, Vs, w, outcome_model=DummyRegressor(strategy="mean"))
    assert res.estimate == pytest.approx(res.ipw_only, abs=1e-9)


def test_error_and_result_fields():
    X, S, Vs, w = _mar_scenario(1)
    res = aipw_mean(X, S, Vs, w, truth=0.0)
    assert res.error == pytest.approx(abs(res.estimate))
    assert aipw_mean(X, S, Vs, w).error is None


def test_input_validation():
    X, S, Vs, w = _mar_scenario(2)
    with pytest.raises(ValueError):
        aipw_mean(X, S, Vs[:-1], w)            # length mismatch
    with pytest.raises(ValueError):
        aipw_mean(X, S, Vs, -w)                # negative weights

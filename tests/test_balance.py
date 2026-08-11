import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from i3pw import balance_report, entropy_balance


def _selection_population(seed, n=40000):
    """Selection driven by covariate X0 *and* the outcome Y — the realistic mix."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    Y = (rng.uniform(size=n) < expit(-1.0 + 0.8 * X[:, 1])).astype(float)
    pi = expit(-2.5 + 1.6 * X[:, 0] + 1.5 * Y)
    S = rng.uniform(size=n) < pi
    return X, Y, S


def test_constrained_moments_are_balanced_but_carry_no_information():
    rng = np.random.default_rng(0)
    Y = (rng.uniform(size=(500, 1)) < 0.3).astype(float)
    w = entropy_balance(Y, [0.45], warn=False)
    rep = balance_report(Y, w, [0.45], constrained=[True], names=["Y"])
    assert abs(rep.smd_after[0]) < 1e-6          # matched by construction
    assert abs(rep.smd_before[0]) > 0.1          # ...and it was off before
    # Nothing was held out, so the report cannot refute anything.
    assert np.isnan(rep.worst_held_out)
    assert rep.passed()                           # vacuously true
    assert "cannot refute" in rep.summary()


def test_held_out_covariate_detects_a_base_model_that_misses_a_driver():
    """Held-out balance can expose a discrepancy that solve diagnostics cannot.

    Selection depends on X0, but the "broken" run calibrates to the known prevalence
    from a uniform base, so X0 is never corrected. Both runs hit the prevalence target
    to machine precision — only the held-out covariate separates them.
    """
    X, Y, S = _selection_population(0)
    Xs, Ys, K = X[S], Y[S], float(Y.mean())

    clf = LogisticRegression(max_iter=400).fit(X[:, :1], S.astype(int))
    base_good = 1.0 / np.clip(clf.predict_proba(Xs[:, :1])[:, 1], 1e-4, 1 - 1e-4)
    w_good, d_good = entropy_balance(Ys.reshape(-1, 1), [K], base_weights=base_good,
                                     return_diagnostics=True, warn=False)
    w_bad, d_bad = entropy_balance(Ys.reshape(-1, 1), [K],
                                   return_diagnostics=True, warn=False)

    # Both solves look healthy; the broken one even reports a *larger* ESS.
    assert d_good.converged and d_bad.converged
    assert d_good.max_abs_residual < 1e-6 and d_bad.max_abs_residual < 1e-6
    assert d_bad.ess > d_good.ess

    F = np.column_stack([Ys, Xs[:, 0]])
    targets = np.array([K, X[:, 0].mean()])
    kw = dict(constrained=[True, False], names=["Y", "X0"])
    rep_good = balance_report(F, w_good, targets, **kw)
    rep_bad = balance_report(F, w_bad, targets, **kw)

    # The held-out covariate is what exposes the broken weighting.
    assert rep_bad.worst_held_out > 3 * rep_good.worst_held_out
    assert not rep_bad.passed()
    assert "FAILS" in rep_bad.summary()


def test_balance_report_validation():
    Y = np.zeros((10, 2))
    with pytest.raises(ValueError, match="one entry per feature column"):
        balance_report(Y, np.ones(10), [0.1])
    with pytest.raises(ValueError, match="one entry per row"):
        balance_report(Y, np.ones(9), [0.1, 0.2])
    with pytest.raises(ValueError, match="one flag per feature column"):
        balance_report(Y, np.ones(10), [0.1, 0.2], constrained=[True])
    with pytest.raises(ValueError, match="one label per feature column"):
        balance_report(Y, np.ones(10), [0.1, 0.2], names=["a"])
    with pytest.raises(ValueError, match="NaN or infinite"):
        balance_report(np.full((10, 1), np.nan), np.ones(10), [0.1])


def test_zero_variance_column_does_not_divide_by_zero():
    const = np.ones((20, 1))
    rep = balance_report(const, np.ones(20), [0.5])
    assert np.isfinite(rep.smd_after[0])

import numpy as np
import pytest

from i3pw.liability import (
    estimate_liability_r2,
    lee_transform,
    liability_r2_from_weights,
    liability_threshold,
    moment_slope,
    observed_to_liability,
    similarity_matrix,
    simulate_case_control,
    simulate_liability_selection,
)


def test_liability_threshold_known_values():
    t, z = liability_threshold(0.5)
    assert t == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(0.3989, abs=1e-4)
    t, z = liability_threshold(0.1)
    assert t == pytest.approx(1.2816, abs=1e-3)
    assert z == pytest.approx(0.1755, abs=1e-3)


def test_lee_reduces_to_population_transform_when_P_equals_K():
    # No ascertainment (P == K): the extra Lee factor is 1.
    assert lee_transform(0.3, 0.1, 0.1) == pytest.approx(observed_to_liability(0.3, 0.1))


def test_moment_slope_recovers_variance_on_continuous_liability():
    # With the latent liability observed directly, the moment estimator returns R2.
    rng = np.random.default_rng(0)
    n, M, r2 = 1500, 300, 0.5
    X = rng.standard_normal((n, M))
    beta = rng.normal(0, np.sqrt(r2 / M), M)
    L = X @ beta + rng.normal(0, np.sqrt(1 - r2), n)
    A = similarity_matrix(X)
    Ls = (L - L.mean()) / L.std()
    assert moment_slope(A, Ls) == pytest.approx(r2, abs=0.08)


def test_simulate_case_control_shapes_and_fraction():
    smp = simulate_case_control(300, 700, 100, 0.5, 0.05, np.random.default_rng(1))
    assert smp.X.shape == (1000, 100)
    assert smp.y.sum() == 300
    assert smp.case_fraction == pytest.approx(0.3)
    assert 0.0 < smp.true_r2 < 1.0


def test_ascertainment_corrections_beat_naive():
    # Under strong ascertainment (balanced sample of a rare outcome), the naive
    # estimate is badly biased while Lee and IPW recover the truth.
    naive, lee, ipw, truth = [], [], [], []
    for s in range(6):
        smp = simulate_case_control(800, 800, 250, 0.5, 0.05, np.random.default_rng(50 + s))
        truth.append(smp.true_r2)
        naive.append(estimate_liability_r2(smp, 0.05, "naive"))
        lee.append(estimate_liability_r2(smp, 0.05, "lee"))
        ipw.append(estimate_liability_r2(smp, 0.05, "ipw"))
    truth = np.mean(truth)
    assert abs(np.mean(lee) - truth) < 0.06
    assert abs(np.mean(ipw) - truth) < 0.06
    # naive ignores ascertainment and is off by a large factor.
    assert np.mean(naive) > truth + 0.5


def test_moment_slope_raises_on_degenerate_matrix():
    # A single unit has no off-diagonal pairs, so the denominator is zero and the
    # slope is undefined -> a clear error rather than a silent nan/inf.
    A = np.array([[1.0]])
    y = np.array([0.5])
    with pytest.raises(ValueError, match="off-diagonal"):
        moment_slope(A, y)


def test_estimate_invalid_method():
    smp = simulate_case_control(100, 100, 50, 0.5, 0.1, np.random.default_rng(2))
    with pytest.raises(ValueError):
        estimate_liability_r2(smp, 0.1, "nope")


def test_selection_population_shapes():
    pop = simulate_liability_selection(5000, 60, 0.5, 0.1, np.random.default_rng(0),
                                       liability_slope=0.5)
    assert pop.X.shape == (5000, 60)
    assert pop.inclusion_prob.min() > 0 and pop.inclusion_prob.max() < 1
    Xs, ys, pis = pop.sample()
    assert Xs.shape[0] == ys.shape[0] == pis.shape[0] == int(pop.selected.sum())
    assert 0.0 < pop.true_r2 < 1.0


def test_oracle_ipw_beats_simple_under_liability_selection():
    # With selection that depends on the latent liability, the simple K/P weights
    # are biased while oracle weights (true inclusion prob) recover the truth.
    simple_err, oracle_err = [], []
    for s in range(4):
        pop = simulate_liability_selection(
            9000, 100, 0.6, 0.1, np.random.default_rng(30 + s), liability_slope=1.0
        )
        Xs, ys, pis = pop.sample()
        K, P = pop.K, float(ys.mean())
        w_simple = np.where(ys == 1, K / P, (1 - K) / (1 - P))
        simple = liability_r2_from_weights(Xs, ys, K, w_simple)
        oracle = liability_r2_from_weights(Xs, ys, K, 1.0 / pis)
        simple_err.append(abs(simple - pop.true_r2))
        oracle_err.append(abs(oracle - pop.true_r2))
    assert np.mean(oracle_err) < 0.05                      # oracle recovers truth
    assert np.mean(oracle_err) < np.mean(simple_err)       # and beats simple K/P IPW


def test_modified_schoeler_recovers_liability_r2():
    # Selection on covariates (Schoeler-capturable) AND disease status (not). A
    # covariate-only participation model leaves the ascertainment uncorrected;
    # raking its weights to the known prevalence recovers the liability-scale R2.
    from scipy.special import expit
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression

    from i3pw import entropy_balance

    sch_err, mod_err = [], []
    for s in range(3):
        rng = np.random.default_rng(200 + s)
        N, pg, ncov, r2, K = 30000, 80, 5, 0.5, 0.1
        G = rng.standard_normal((N, pg))
        b = rng.normal(0, np.sqrt(r2 / pg), pg)
        sig = G @ b
        L = sig + rng.normal(0, np.sqrt(1 - r2), N)
        truth = sig.var() / L.var()
        Y = (L > norm.ppf(1 - K)).astype(float)
        Xs = rng.standard_normal((N, ncov))
        pi = expit(-3.5 + Xs @ rng.normal(0, 0.6, ncov) + 1.5 * Y)
        S = rng.uniform(size=N) < pi
        Gs, ys, Kp = G[S], Y[S], float(Y.mean())
        clf = LogisticRegression(max_iter=300).fit(Xs, S.astype(int))
        w_sch = 1.0 / np.clip(clf.predict_proba(Xs[S])[:, 1], 1e-4, 1 - 1e-4)
        w_mod = entropy_balance(ys.reshape(-1, 1), [Kp], base_weights=w_sch)
        sch_err.append(abs(liability_r2_from_weights(Gs, ys, Kp, w_sch) - truth))
        mod_err.append(abs(liability_r2_from_weights(Gs, ys, Kp, w_mod) - truth))
    assert np.mean(mod_err) < 0.15                 # modified recovers the truth
    assert np.mean(mod_err) < np.mean(sch_err)     # and beats covariate-only IPW

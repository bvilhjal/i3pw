import numpy as np
import pytest
from scipy.stats import norm

from i3pw import (
    calibration_ipw,
    effective_sample_size,
    entropy_balance,
    make_dataset,
    outcome_calibration_weights,
)


@pytest.fixture(scope="module")
def dataset():
    return make_dataset(
        seed=3, population_size=6000, n_features=12, n_outcomes=2,
        predictors_per_outcome=6,
        target_population_prevalence=(0.4, 0.08),
        target_sample_prevalence=(0.2, 0.01), sample_size=1500,
    )


def test_entropy_balance_hits_targets_exactly():
    rng = np.random.default_rng(0)
    Y = (rng.uniform(size=(400, 2)) < [0.3, 0.1]).astype(float)
    target = np.array([0.45, 0.2])
    w = entropy_balance(Y, target)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0)
    assert np.allclose((w[:, None] * Y).sum(axis=0), target, atol=1e-6)


def test_entropy_balance_single_target():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=300) < 0.25).astype(float)
    w = entropy_balance(y, np.array([0.5]))
    assert (w * y).sum() == pytest.approx(0.5, abs=1e-6)


def test_entropy_balance_uniform_base_recovers_uniform_when_already_calibrated():
    # If the sample already matches the target, no tilt is needed -> uniform weights.
    y = np.array([1.0, 1.0, 0.0, 0.0])
    w = entropy_balance(y, np.array([0.5]))
    assert np.allclose(w, 0.25)


def test_ridge_shrinks_toward_base():
    rng = np.random.default_rng(2)
    Y = (rng.uniform(size=(300, 1)) < 0.2).astype(float)
    exact = entropy_balance(Y, np.array([0.5]), ridge=0.0)
    shrunk = entropy_balance(Y, np.array([0.5]), ridge=5.0)
    # Heavy ridge -> weights closer to uniform (smaller deviation from 1/n).
    n = len(Y)
    assert np.std(shrunk) < np.std(exact)
    assert np.all(shrunk > 0)
    assert abs(shrunk.sum() - 1) < 1e-9 and abs(1 / n - shrunk.mean()) < 1e-9


def test_effective_sample_size():
    assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)
    # One dominating weight -> tiny ESS.
    w = np.array([100.0, 1.0, 1.0, 1.0])
    assert effective_sample_size(w) < 2.0


def test_calibration_matches_anchored_prevalence(dataset):
    # Calibrating on the common outcome must reproduce its prevalence exactly
    # (anchoring the common outcome is always feasible; a very rare outcome can
    # have too few cases in the ascertained sample to reach the target).
    r = calibration_ipw(dataset, anchor_outcomes=[0], base="uniform")
    pop = dataset.population_prevalence
    assert r.achieved_prevalence[0] == pytest.approx(pop[0], abs=1e-6)
    assert r.weighted_prevalence[0] == pytest.approx(pop[0], abs=1e-6)
    assert r.ess > 0


def test_calibration_sample_only(dataset):
    # Deployable: unselected units carry zero weight.
    r = calibration_ipw(dataset, base="uniform")
    _, _, s_test = dataset.split("test")
    w = r.extra["weight"]
    assert np.all(w[s_test == 0] == 0)


def test_calibration_beats_lasso_on_anchored_outcome(dataset):
    # The mechanism: calibrating on a known prevalence reproduces it (error ~0),
    # while the covariate-only LASSO participation model leaves substantial error
    # because participation here is outcome-driven, not covariate-driven.
    from i3pw import lasso_ipw

    cal = calibration_ipw(dataset, anchor_outcomes=[0], base="lasso")
    las = lasso_ipw(dataset, weighting="inverse")
    assert cal.percent_diff[0] < 1e-3
    assert cal.percent_diff[0] < las.percent_diff[0]


def test_partial_anchor_leaves_other_outcome_free(dataset):
    # Anchoring only outcome 0 fixes it exactly; outcome 1 is not constrained.
    r = calibration_ipw(dataset, anchor_outcomes=[0], base="uniform")
    assert r.anchor_outcomes == (0,)
    assert r.percent_diff[0] < 1e-3


def test_calibration_delegates_methodresult_api(dataset):
    r = calibration_ipw(dataset, base="uniform")
    assert "calibration_ipw" in r.summary()
    assert r.weighted_prevalence.shape == (2,)


def test_outcome_calibration_hits_marginals_and_cooccurrence():
    rng = np.random.default_rng(0)
    Y = (rng.uniform(size=(2000, 2)) < [0.3, 0.2]).astype(float)
    targets = [0.45, 0.35]
    w = outcome_calibration_weights(Y, targets)
    assert np.allclose((w[:, None] * Y).sum(axis=0), targets, atol=1e-6)
    # With a co-occurrence constraint, that joint moment is matched too.
    k12 = 0.18
    wj = outcome_calibration_weights(Y, targets, joint_prevalences={(0, 1): k12})
    assert np.allclose((wj[:, None] * Y).sum(axis=0), targets, atol=1e-6)
    assert (wj * Y[:, 0] * Y[:, 1]).sum() == pytest.approx(k12, abs=1e-6)


def _multi_outcome_sample(g, seed, n_pop=200000, rho=0.5, k1=0.15, k2=0.08):
    rng = np.random.default_rng(seed)
    L = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n_pop)
    t1, t2 = norm.ppf(1 - k1), norm.ppf(1 - k2)
    Y1 = (L[:, 0] > t1).astype(float)
    Y2 = (L[:, 1] > t2).astype(float)
    zj = L[:, 0] * L[:, 1]
    pi = np.clip(0.006 * 7.0**Y1 * 8.0**Y2 * g ** (Y1 * Y2), 1e-9, 1.0)
    s = rng.uniform(size=n_pop) < pi
    moments = (Y1.mean(), Y2.mean(), (Y1 * Y2).mean())
    return Y1[s], Y2[s], zj[s], pi[s], moments


def _joint_estimate(y1, y2, zj, moments, use_joint):
    k1, k2, k12 = moments
    feats = np.column_stack([y1, y2, y1 * y2]) if use_joint else np.column_stack([y1, y2])
    tgt = [k1, k2, k12] if use_joint else [k1, k2]
    w = entropy_balance(feats, tgt)
    return float(np.sum(w * zj) / np.sum(w))


def test_marginal_calibration_matches_oracle_without_coupling():
    # g = 1: selection factorizes, so marginal calibration tracks the oracle (true
    # inverse-probability weights) even on the joint target E[L1*L2]. Compare on the
    # same sample to cancel the estimand's variance.
    diffs = []
    for s in range(5):
        y1, y2, zj, pis, m = _multi_outcome_sample(1.0, s)
        marg = _joint_estimate(y1, y2, zj, m, use_joint=False)
        oracle = float(np.sum((1.0 / pis) * zj) / np.sum(1.0 / pis))
        diffs.append(abs(marg - oracle))
    assert np.mean(diffs) < 0.02  # measured ~0.007; the g>1 gap is ~5x larger


def test_interaction_needs_cooccurrence_constraint():
    # g > 1: comorbid over-recruitment couples the outcomes. Marginal calibration
    # drifts from the oracle on the joint target; adding the co-occurrence constraint
    # brings it back.
    marg_d, joint_d = [], []
    for s in range(5):
        y1, y2, zj, pis, m = _multi_outcome_sample(2.5, s)
        oracle = float(np.sum((1.0 / pis) * zj) / np.sum(1.0 / pis))
        marg_d.append(abs(_joint_estimate(y1, y2, zj, m, use_joint=False) - oracle))
        joint_d.append(abs(_joint_estimate(y1, y2, zj, m, use_joint=True) - oracle))
    assert np.mean(joint_d) < 0.015               # joint calibration ~ oracle
    assert np.mean(joint_d) * 2 < np.mean(marg_d)  # and clearly closer than marginal

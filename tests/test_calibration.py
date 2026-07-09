import warnings

import numpy as np
import pytest
from scipy.stats import norm

from i3pw import (
    CalibrationDiagnostics,
    CalibrationWarning,
    calibration_ipw,
    effective_sample_size,
    entropy_balance,
    make_dataset,
    outcome_calibration_weights,
    stratified_calibration_weights,
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


def test_entropy_balance_returns_diagnostics_on_success():
    rng = np.random.default_rng(0)
    Y = (rng.uniform(size=(400, 2)) < [0.3, 0.1]).astype(float)
    w, diag = entropy_balance(Y, np.array([0.45, 0.2]), return_diagnostics=True)
    assert isinstance(diag, CalibrationDiagnostics)
    assert diag.converged
    assert diag.max_abs_residual < 1e-6
    assert diag.ess == pytest.approx(effective_sample_size(w))
    assert 0.0 < diag.top1pct_weight_mass < 1.0
    assert "converged" in diag.summary()


def test_entropy_balance_warns_and_flags_infeasible_target():
    # A target of 0.2 when no unit is a case: no exponential tilt can reach it.
    y = np.zeros((50, 1))
    with pytest.warns(CalibrationWarning):
        w, diag = entropy_balance(y, np.array([0.2]), return_diagnostics=True)
    assert not diag.converged or diag.max_abs_residual > 1e-6
    assert diag.max_abs_residual == pytest.approx(0.2, abs=1e-6)


def test_entropy_balance_warn_false_is_silent():
    y = np.zeros((50, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        entropy_balance(y, np.array([0.2]), warn=False)


def test_ridge_does_not_warn_on_expected_residual():
    # With ridge > 0 the constraints are intentionally not met; that is not a warning.
    rng = np.random.default_rng(2)
    Y = (rng.uniform(size=(300, 1)) < 0.2).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        entropy_balance(Y, np.array([0.5]), ridge=5.0)


def test_calibration_ipw_validates_base_and_scheme(dataset):
    with pytest.raises(ValueError):
        calibration_ipw(dataset, base="nonsense")
    with pytest.raises(ValueError):
        calibration_ipw(dataset, base_scheme="nonsense")


def test_calibration_ipw_reports_support_and_diagnostics(dataset):
    r = calibration_ipw(dataset, anchor_outcomes=[0], base="uniform")
    assert r.diagnostics is not None and r.diagnostics.converged
    n_case, n_ctrl = r.support[0]
    assert n_case > 0 and n_ctrl > 0
    assert "support" in r.diagnostics_summary()
    assert r.pre_trim_residual < 1e-6
    assert r.post_trim_residual == pytest.approx(r.pre_trim_residual)


def test_calibration_ipw_trim_breaks_calibration_and_is_reported(dataset):
    # Anchoring the rare outcome too forces extreme case weights; trimming them
    # clips real mass and pulls the achieved prevalence off its exact target.
    with pytest.warns(CalibrationWarning):
        r = calibration_ipw(dataset, base="uniform", trim=0.9)
    assert r.post_trim_residual > r.pre_trim_residual


def test_outcome_calibration_warns_on_unreachable_cooccurrence():
    # Two mutually exclusive outcomes: their co-occurrence is never observed.
    y1 = np.array([1.0, 0.0] * 100)
    y2 = 1.0 - y1
    Y = np.column_stack([y1, y2])
    with pytest.warns(CalibrationWarning):
        outcome_calibration_weights(Y, [0.5, 0.5], joint_prevalences={(0, 1): 0.1})


def _stratified_pop(seed):
    # Two strata (share 0.6 / 0.4) with different disease prevalence (0.10 / 0.30);
    # selection over-samples stratum 1 AND cases; a held-out trait Z depends on the
    # stratum only, so its population mean is P(A=1)=0.4.
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    n = 200000
    A = (rng.uniform(size=n) < 0.4).astype(int)
    prev = np.where(A == 1, 0.30, 0.10)
    Y = (rng.uniform(size=n) < prev).astype(float)
    Z = A.astype(float) + rng.standard_normal(n)
    pi = expit(-1.5 + 1.2 * A + 1.0 * Y)
    S = rng.uniform(size=n) < pi
    return A[S], Y[S][:, None], Z[S], float(Z.mean())


def test_stratified_calibration_recovers_within_stratum_prevalence():
    A_s, Y_s, _, _ = _stratified_pop(0)
    within = np.array([[0.10], [0.30]])
    share = np.array([0.6, 0.4])
    w = stratified_calibration_weights(Y_s, A_s, within, share)
    for a in (0, 1):
        m = A_s == a
        assert (w[m] * Y_s[m, 0]).sum() / w[m].sum() == pytest.approx(within[a, 0], abs=1e-6)
        assert w[m].sum() == pytest.approx(share[a], abs=1e-6)  # stratum shares restored too


def test_stratified_beats_pooled_on_stratum_dependent_estimand():
    within = np.array([[0.10], [0.30]])
    share = np.array([0.6, 0.4])
    strat_err, pooled_err, naive_err = [], [], []
    for s in range(3):
        A_s, Y_s, Z_s, z_truth = _stratified_pop(10 + s)
        w_strat = stratified_calibration_weights(Y_s, A_s, within, share)
        w_pool = outcome_calibration_weights(Y_s, [float(share @ within[:, 0])])
        strat_err.append(abs(float(np.sum(w_strat * Z_s)) - z_truth))
        pooled_err.append(abs(float(np.sum(w_pool * Z_s)) - z_truth))
        naive_err.append(abs(Z_s.mean() - z_truth))
    assert np.mean(strat_err) < np.mean(pooled_err) < np.mean(naive_err)
    assert np.mean(strat_err) < 0.01  # stratum shares pinned -> Z recovered


def test_stratified_calibration_validates_shapes():
    Y = np.zeros((10, 2))
    share = np.array([0.5, 0.5])
    with pytest.raises(ValueError):  # within_stratum_prevalence must be (A, Q)
        stratified_calibration_weights(Y, np.zeros(10, int), np.zeros((2, 3)), share)
    with pytest.raises(ValueError):  # stratum label 5 out of range for A=2
        stratified_calibration_weights(Y, np.array([0, 5] * 5), np.zeros((2, 2)), share)


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


def test_outcome_calibration_composes_with_base_weights():
    # The recommended UK Biobank pipeline: take covariate-participation weights as a
    # base, then calibrate them to known disease prevalences. Calibration must still
    # hit the margins exactly, and must actually use the base (differ from uniform).
    rng = np.random.default_rng(5)
    Y = (rng.uniform(size=(1500, 2)) < [0.3, 0.15]).astype(float)
    base = rng.uniform(0.5, 5.0, 1500)  # e.g. 1 / P̂(S | X)
    targets = [0.5, 0.3]
    w = outcome_calibration_weights(Y, targets, base_weights=base)
    assert np.allclose((w[:, None] * Y).sum(axis=0), targets, atol=1e-6)
    assert not np.allclose(w, outcome_calibration_weights(Y, targets))


def test_selection_inference_combined_beats_registry():
    # Latent selection variable U; N outcomes proxy it, k observed frame-wide.
    # Combining a registry model with calibration to all N known means recovers a
    # held-out U-correlated trait better than the registry model or naive.
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression

    naive_z, reg_z, comb_z = [], [], []
    for s in range(3):
        rng = np.random.default_rng(300 + s)
        n_pop, n_out, k = 40000, 12, 3
        U = rng.standard_normal(n_pop)
        lam = rng.uniform(0.3, 0.5, n_out)
        lam[:k] = rng.uniform(0.7, 0.85, k)
        Kj = rng.uniform(0.05, 0.30, n_out)
        Y = (lam * U[:, None] + np.sqrt(1 - lam**2) * rng.standard_normal((n_pop, n_out))
             > norm.ppf(1 - Kj)).astype(float)
        Kpop = Y.mean(axis=0)
        Z = 0.8 * U + rng.standard_normal(n_pop)
        z_truth = Z.mean()
        S = rng.uniform(size=n_pop) < expit(-1.3 + U)
        Ys, Zs = Y[S], Z[S]
        clf = LogisticRegression(max_iter=300).fit(Y[:, :k], S.astype(int))
        w_reg = 1.0 / np.clip(clf.predict_proba(Ys[:, :k])[:, 1], 1e-4, 1 - 1e-4)
        w_comb = entropy_balance(Ys, Kpop, base_weights=w_reg)

        def zbias(w, zs=Zs, zt=z_truth):
            return abs(float(np.sum(w * zs) / np.sum(w)) - zt)

        naive_z.append(abs(Zs.mean() - z_truth))
        reg_z.append(zbias(w_reg))
        comb_z.append(zbias(w_comb))
    assert np.mean(comb_z) < np.mean(reg_z) < np.mean(naive_z)


def _lee_cc_weights(Ys, Kpop):
    # Product of per-outcome case-control ratios K_j/P_j (case), (1-K)/(1-P) (control).
    P = np.clip(Ys.mean(axis=0), 1e-3, 1 - 1e-3)
    K = np.clip(Kpop, 1e-3, 1 - 1e-3)
    logw = (Ys * np.log(K / P) + (1 - Ys) * np.log((1 - K) / (1 - P))).sum(axis=1)
    return np.exp(logw - logw.max())


def _sel_inference_pop(seed, scenario):
    # Shared latent U; N correlated outcomes; held-out U-correlated trait Z.
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    n_pop, n_out = 40000, 12
    U = rng.standard_normal(n_pop)
    lam = rng.uniform(0.3, 0.5, n_out)
    lam[:3] = rng.uniform(0.7, 0.85, 3)
    Kj = rng.uniform(0.05, 0.30, n_out)
    Y = (lam * U[:, None] + np.sqrt(1 - lam**2) * rng.standard_normal((n_pop, n_out))
         > norm.ppf(1 - Kj)).astype(float)
    Z = 0.8 * U + rng.standard_normal(n_pop)
    if scenario == "latent":
        pi = expit(-1.3 + 1.2 * U)
    else:  # case_control: selection driven by a few correlated outcomes, not U directly
        pi = expit(-1.4 + 1.3 * Y[:, :3].sum(axis=1))
    S = rng.uniform(size=n_pop) < pi
    return Y[S], Y.mean(axis=0), Z[S], float(Z.mean())


def _zbias(w, zs, zt):
    return abs(float(np.sum(w * zs) / np.sum(w)) - zt)


def test_lee_style_weights_beat_naive_under_latent_selection():
    # When every outcome proxies one latent driver, averaging N analytic case-control
    # corrections (lee_cc) reconstructs it and removes most of the held-out bias.
    lee_z, naive_z = [], []
    for s in range(3):
        Ys, Kpop, Zs, zt = _sel_inference_pop(400 + s, "latent")
        lee_z.append(_zbias(_lee_cc_weights(Ys, Kpop), Zs, zt))
        naive_z.append(abs(Zs.mean() - zt))
    assert np.mean(lee_z) < 0.4 * np.mean(naive_z)


def test_calibration_beats_lee_style_weights_under_case_control_selection():
    # When only a few correlated outcomes drive selection, the analytic lee_cc weights
    # over-correct (an independent correction per outcome) and barely beat naive, while
    # exact calibration to the same known means cannot push past the true margins.
    lee_z, cal_z, naive_z = [], [], []
    for s in range(3):
        Ys, Kpop, Zs, zt = _sel_inference_pop(420 + s, "case_control")
        lee_z.append(_zbias(_lee_cc_weights(Ys, Kpop), Zs, zt))
        cal_z.append(_zbias(entropy_balance(Ys, Kpop), Zs, zt))
        naive_z.append(abs(Zs.mean() - zt))
    assert np.mean(cal_z) < np.mean(lee_z)
    assert np.mean(cal_z) < 0.2 * np.mean(naive_z)


def _schoeler_pop(seed, c_x, c_u):
    # Selection on socioeconomic covariates X (independent of the disease latent U)
    # and on U; held-out Z loads on both channels. Returns sample pieces + weights.
    from scipy.special import expit
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    n_pop, n_out, p = 40000, 12, 8
    X = rng.standard_normal((n_pop, p))
    b = np.zeros(p)
    b[:3] = rng.uniform(0.4, 0.9, 3) * rng.choice([-1.0, 1.0], 3)
    xb = X @ b
    xb = xb / xb.std()
    U = rng.standard_normal(n_pop)
    lam = rng.uniform(0.3, 0.5, n_out)
    lam[:3] = rng.uniform(0.7, 0.85, 3)
    Kj = rng.uniform(0.05, 0.30, n_out)
    Y = (lam * U[:, None] + np.sqrt(1 - lam**2) * rng.standard_normal((n_pop, n_out))
         > norm.ppf(1 - Kj)).astype(float)
    Kpop = Y.mean(axis=0)
    Z = 0.7 * xb + 0.7 * U + rng.standard_normal(n_pop)
    S = rng.uniform(size=n_pop) < expit(-1.4 + c_x * xb + c_u * U)
    clf = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.5, max_iter=500)
    clf.fit(X, S.astype(int))
    w_sch = 1.0 / np.clip(clf.predict_proba(X[S])[:, 1], 1e-4, 1 - 1e-4)
    return Y[S], Kpop, Z[S], float(Z.mean()), w_sch


def test_schoeler_and_calibration_are_complementary():
    # X-driven selection: the Schoeler covariate model helps, prevalence calibration
    # barely does. Disease-driven selection: the reverse. Combining is best in both.
    for c_x, c_u, sch_wins in [(1.5, 0.4, True), (0.4, 1.5, False)]:
        sch_z, cal_z, both_z, naive_z = [], [], [], []
        for s in range(3):
            Ys, Kpop, Zs, zt, w_sch = _schoeler_pop(440 + s, c_x, c_u)
            sch_z.append(_zbias(w_sch, Zs, zt))
            cal_z.append(_zbias(entropy_balance(Ys, Kpop), Zs, zt))
            both_z.append(_zbias(entropy_balance(Ys, Kpop, base_weights=w_sch), Zs, zt))
            naive_z.append(abs(Zs.mean() - zt))
        m_sch, m_cal, m_both = np.mean(sch_z), np.mean(cal_z), np.mean(both_z)
        if sch_wins:
            assert m_sch < m_cal            # covariate model wins the socioeconomic channel
        else:
            assert m_cal < m_sch            # calibration wins the disease channel
        assert m_both < m_sch and m_both < m_cal        # combining beats either alone
        assert m_both < np.mean(naive_z)


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

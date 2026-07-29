"""Tests for the array-level entry point (:func:`i3pw.calibrate`)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

import i3pw
from i3pw import CalibrationWarning, calibrate, inverse_probability_weights
from i3pw.dgm import make_dataset


@pytest.fixture(scope="module")
def dataset():
    return make_dataset(
        seed=3, population_size=6000, n_features=12, n_outcomes=2,
        predictors_per_outcome=6,
        target_population_prevalence=(0.4, 0.08),
        target_sample_prevalence=(0.2, 0.01), sample_size=1500,
    )


def _cohort(seed=0, n=60_000):
    """A cohort shaped like the real thing: both selection channels, and a held-out margin.

    Selection depends on the disease (1.8 on the log-odds) *and* on age, so neither half
    of the correction can do the job alone. ``base_weights`` are inverted from a
    participation model fitted on the frame — the covariates of participants and
    non-participants alike, which is what a real cohort needs to fit one at all.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(45.0, 12.0, n)
    female = (rng.uniform(size=n) < 0.51).astype(float)
    Y = (rng.uniform(size=n) < expit(-3.0 + 0.03 * (age - 45.0) + 0.4 * female)).astype(float)
    trait = 25.0 + 0.05 * (age - 45.0) + 2.0 * Y + rng.normal(0.0, 3.0, n)
    S = rng.uniform(size=n) < expit(-2.0 + 1.8 * Y + 0.05 * (age - 45.0))

    frame = np.column_stack([age, female])
    p_hat = LogisticRegression().fit(frame, S.astype(int)).predict_proba(frame[S])[:, 1]
    return {
        "Y": Y[S], "age": age[S], "female": female[S], "trait": trait[S],
        "base_weights": inverse_probability_weights(p_hat),
        "K": float(Y.mean()), "age_mean": float(age.mean()),
        "female_mean": float(female.mean()), "trait_mean": float(trait.mean()),
    }


def test_calibrate_reproduces_the_known_prevalence():
    c = _cohort()
    fit = calibrate(c["Y"], [c["K"]])
    assert fit.weights.sum() == pytest.approx(1.0)
    assert float(fit.weights @ c["Y"]) == pytest.approx(c["K"], abs=1e-8)
    assert fit.diagnostics.converged
    assert fit.ess > 0


def test_calibrate_accepts_1d_outcomes_and_names_them():
    c = _cohort()
    flat = calibrate(c["Y"], [c["K"]], outcome_names=["scz"])
    wide = calibrate(c["Y"][:, None], [c["K"]], outcome_names=["scz"])
    assert np.allclose(flat.weights, wide.weights)
    assert flat.constraint_names == ["scz"]


def test_holdout_is_a_real_test_that_the_constraints_are_not():
    # The point of the whole exercise: the anchored margin matches by construction, and
    # the held-out register margins did not have to. Both ingredients are supplied here
    # — participation model *and* calibration — which is the recipe the docs recommend.
    c = _cohort()
    fit = calibrate(
        c["Y"], [c["K"]], base_weights=c["base_weights"],
        holdout={"age": (c["age"], c["age_mean"]), "female": (c["female"], c["female_mean"])},
    )
    rep = fit.balance
    assert rep is not None
    assert list(rep.constrained) == [True, False, False]
    assert abs(rep.smd_after[0]) < 1e-9              # constrained: exact by construction
    assert not np.isnan(rep.worst_held_out)          # ...and excluded from the verdict
    # Selection was on age, so the raw sample is unbalanced on it and weighting fixes it.
    assert abs(rep.smd_before[1]) > 0.3
    assert abs(rep.smd_after[1]) < 0.1
    assert rep.passed()


def test_calibration_alone_leaves_the_covariate_channel_uncorrected():
    # The package's central claim, run in reverse: without base weights the disease
    # margin is still matched exactly, and the age margin selection acted on is not.
    # A holdout is what makes the difference visible; the solve looks identical.
    c = _cohort()
    pure = calibrate(c["Y"], [c["K"]], holdout={"age": (c["age"], c["age_mean"])})
    both = calibrate(c["Y"], [c["K"]], base_weights=c["base_weights"],
                     holdout={"age": (c["age"], c["age_mean"])})
    assert pure.diagnostics.max_abs_residual < 1e-8
    assert both.diagnostics.max_abs_residual < 1e-8
    assert abs(pure.balance.worst_held_out) > 5 * abs(both.balance.worst_held_out)
    assert not pure.balance.passed()
    assert both.balance.passed()


def test_no_holdout_says_so_instead_of_implying_a_pass():
    c = _cohort()
    fit = calibrate(c["Y"], [c["K"]])
    assert fit.balance is None
    assert "no holdout supplied" in fit.summary()


def test_a_broken_weighting_is_refuted_by_the_holdout():
    # Calibrating the disease margin while ignoring the covariate channel leaves the
    # age selection uncorrected. Nothing in the solve notices; the holdout does.
    c = _cohort()
    fit = calibrate(
        c["Y"], [c["K"]],
        base_weights=np.exp(-0.35 * (c["age"] - 45.0)),  # a badly wrong participation model
        holdout={"age": (c["age"], c["age_mean"])},
    )
    assert fit.diagnostics.converged                      # the solve is perfectly happy
    assert fit.diagnostics.max_abs_residual < 1e-8        # ...and hits its target exactly
    assert not fit.balance.passed()                       # only the held-out margin objects
    assert "FAIL" in fit.balance.summary()


def test_calibration_corrects_a_downstream_mean():
    c = _cohort()
    fit = calibrate(c["Y"], [c["K"]], base_weights=c["base_weights"])
    naive_err = abs(float(c["trait"].mean()) - c["trait_mean"])
    calib_err = abs(fit.mean(c["trait"]).value - c["trait_mean"])
    assert calib_err < naive_err / 2


def test_mean_uses_the_constraints_for_its_standard_error():
    c = _cohort()
    fit = calibrate(c["Y"], [c["K"]])
    # An anchored margin has no sampling variability left: the GREG residual is zero.
    anchored = fit.mean(c["Y"])
    assert anchored.value == pytest.approx(c["K"], abs=1e-8)
    assert anchored.se == pytest.approx(0.0, abs=1e-12)
    # The fixed-weight formula cannot see that and reports a positive SE.
    assert i3pw.weighted_mean_se(c["Y"], fit.weights).se > 1e-4
    # An estimand orthogonal to the constraint keeps an ordinary-sized SE.
    assert fit.mean(c["female"]).se > 1e-4


def test_apply_to_reproduces_the_fitted_weights_on_the_same_rows():
    c = _cohort()
    bw = inverse_probability_weights(np.full(c["Y"].shape[0], 0.3))
    fit = calibrate(c["Y"], [c["K"]], base_weights=bw)
    again = fit.apply_to(fit.constraint_features, base_weights=bw)
    assert np.allclose(again, fit.weights)


def test_stratified_path_matches_within_stratum_prevalence():
    c = _cohort()
    strata = (c["age"] > 45.0).astype(int)          # two strata: younger / older
    within = np.array([[0.05], [0.12]])
    share = np.array([0.5, 0.5])
    fit = calibrate(c["Y"], within, strata=strata, stratum_share=share)
    for a in (0, 1):
        m = strata == a
        w = fit.weights[m]
        assert w.sum() == pytest.approx(share[a], abs=1e-6)
        assert float(w @ c["Y"][m]) / w.sum() == pytest.approx(within[a, 0], abs=1e-6)


def test_stratified_arguments_are_validated_not_silently_reinterpreted():
    # A (A, Q) vs (Q,) mix-up silently changes the estimand, so each pairing errors out.
    c = _cohort(n=5_000)
    strata = (c["age"] > 45.0).astype(int)
    share = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="within-stratum"):
        calibrate(c["Y"], [c["K"]], strata=strata, stratum_share=share)
    with pytest.raises(ValueError, match="stratum_share"):
        calibrate(c["Y"], np.array([[0.05], [0.12]]), strata=strata)
    with pytest.raises(ValueError, match="targets must be 1-D"):
        calibrate(c["Y"], np.array([[0.05], [0.12]]))
    with pytest.raises(ValueError, match="pass both or neither"):
        calibrate(c["Y"], [c["K"]], stratum_share=share)
    with pytest.raises(ValueError, match="joint_prevalences is not supported"):
        calibrate(c["Y"], np.array([[0.05], [0.12]]), strata=strata,
                  stratum_share=share, joint_prevalences={(0, 0): 0.01})


def test_length_mismatches_are_caught_with_the_offending_name():
    c = _cohort(n=5_000)
    with pytest.raises(ValueError, match="base_weights must have one entry"):
        calibrate(c["Y"], [c["K"]], base_weights=np.ones(7))
    with pytest.raises(ValueError, match="holdout\\['age'\\]"):
        calibrate(c["Y"], [c["K"]], holdout={"age": (np.ones(7), 45.0)})


def test_unreachable_target_warns():
    with pytest.warns(CalibrationWarning, match="no support"):
        calibrate(np.zeros(500), [0.2])


def test_joint_prevalences_reach_a_coupled_target():
    rng = np.random.default_rng(5)
    n = 20_000
    y1 = (rng.uniform(size=n) < 0.3).astype(float)
    y2 = (rng.uniform(size=n) < 0.3).astype(float)
    fit = calibrate(np.column_stack([y1, y2]), [0.2, 0.25],
                    joint_prevalences={(0, 1): 0.08})
    assert float(fit.weights @ y1) == pytest.approx(0.2, abs=1e-7)
    assert float(fit.weights @ y2) == pytest.approx(0.25, abs=1e-7)
    assert float(fit.weights @ (y1 * y2)) == pytest.approx(0.08, abs=1e-7)
    assert fit.constraint_names == ["Y0", "Y1", "Y0&Y1"]


def test_inverse_probability_weights_schemes():
    p = np.array([0.2, 0.5, 0.8])
    assert np.allclose(inverse_probability_weights(p), 1.0 / p)
    assert np.allclose(inverse_probability_weights(p, scheme="odds"), (1.0 - p) / p)
    # A participation model predicting certainty gives a large weight, not an infinite one.
    assert np.all(np.isfinite(inverse_probability_weights(np.array([0.0, 1.0]))))
    with pytest.raises(ValueError, match="scheme must be"):
        inverse_probability_weights(p, scheme="nonsense")


def test_calibration_ipw_is_a_thin_wrapper_over_calibrate(dataset):
    # The simulation front door and the array front door must not drift apart: given the
    # same base weights and targets they are the same estimator.
    r = i3pw.calibration_ipw(dataset, anchor_outcomes=[0], base="uniform")
    _, Y_test, s_test = dataset.split("test")
    sel = s_test == 1
    fit = calibrate(Y_test[sel][:, [0]], dataset.population_prevalence[[0]])
    assert np.allclose(fit.weights, r.extra["weight"][sel])


def test_constraint_count_guard_rejects_an_unidentified_tilt():
    # More constraints than units: the tilt matches anything and identifies nothing.
    rng = np.random.default_rng(0)
    Y = (rng.uniform(size=(6, 8)) < 0.4).astype(float)
    with pytest.raises(ValueError, match="constraints for 6 units"):
        calibrate(Y, np.full(8, 0.4))


def test_constraint_count_guard_warns_on_a_thin_stratified_design():
    # The realistic version: strata x outcomes outruns the cells that support them.
    # 8 strata x 1 outcome + 7 share constraints = 15 constraints on 120 units.
    rng = np.random.default_rng(1)
    n, A = 120, 8
    Y = (rng.uniform(size=(n, 1)) < 0.5).astype(float)
    strata = np.arange(n) % A          # every stratum populated, so the only
    with pytest.warns(CalibrationWarning, match="units per constraint"):  # complaint is
        calibrate(Y, np.full((A, 1), 0.5), strata=strata,                 # the count
                  stratum_share=np.full(A, 1.0 / A))


def test_warn_false_silences_the_calibrate_path():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calibrate(np.zeros(500), [0.2], warn=False)

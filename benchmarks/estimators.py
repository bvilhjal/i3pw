"""The estimator zoo, and the metrics every benchmark scores it on.

One place decides what ``ipw+cal`` means, so a row labelled ``ipw+cal`` in the
selection-law sweep is the same estimator as the one in the coverage study. Each
entry returns weights over the *participants only* — nothing here may read an
unobserved outcome — except ``oracle``, which reads the true inclusion
probabilities and exists to bound what any weighting could achieve.

    naive       uniform weights: the participant mean, uncorrected.
    ipw         1/P-hat(S=1 | frame), LASSO logistic. The Schoeler-style covariate
                participation model, and the only ingredient that can see a
                covariate-driven channel.
    cal         entropy calibration to the register prevalence from a uniform base.
                The pure label-shift correction.
    ipw+cal     the recommended estimator: the covariate model supplies the base,
                the register prevalence supplies the tilt.
    ipw+cal/s   the same, calibrated *within* strata to per-stratum prevalence and
                stratum shares rather than to the pooled margin.
    ipw+cal/v   the same, calibrated to the register's mild- and severe-case
                prevalences separately rather than to their sum.
    aipw        ipw+cal weights augmented with an outcome regression for the
                estimand (cross-fitted). Handled separately: it is an estimator of
                one estimand, not a weighting.
    oracle      1/pi with the true inclusion probabilities. Simulation only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from benchmarks.simulate import Population
from i3pw import (
    CalibrationWarning,
    balance_report,
    effective_sample_size,
    entropy_balance,
    lasso_propensity,
    stratified_calibration_weights,
)

WEIGHTING_METHODS = ("naive", "ipw", "cal", "ipw+cal", "ipw+cal/s", "ipw+cal/v", "oracle")


@dataclass
class Weighting:
    """Normalized participant weights, plus what is needed to score them."""

    method: str
    weights: np.ndarray
    features: np.ndarray | None = None
    """The calibration functions actually constrained, for a calibration-aware SE."""
    targets: np.ndarray | None = None
    converged: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def ess(self) -> float:
        return effective_sample_size(self.weights)

    @property
    def max_share(self) -> float:
        """Largest single unit's share of the total weight, in units of ``1/n``.

        A value of 1 is perfect equality; 30 means one participant carries thirty
        times an equal share, which is the shape of a weighting about to fail.
        """
        w = self.weights
        return float(w.max() / w.mean()) if w.size else float("nan")


def propensity(pop: Population, *, cv: int = 5) -> np.ndarray:
    """LASSO participation probabilities for the participants.

    Fitted on the whole frame, which is the register-linked situation: covariates
    and participation status are known for everybody, outcomes only for
    participants. (``i3pw.calibration_ipw`` instead splits train/test, because its
    ``Dataset`` has no separate frame.)
    """
    frame = pop.frame()
    return lasso_propensity(frame, pop.selected, frame[pop.mask], cv=cv)


def fit_weighting(
    pop: Population,
    method: str,
    *,
    p_hat: np.ndarray | None = None,
    anchors: tuple[int, ...] = (0,),
    targets: np.ndarray | None = None,
    shrinkage: float = 0.0,
) -> Weighting:
    """Weights for one method on one population.

    ``p_hat`` lets the caller fit the participation model once and share it across
    the methods that need it, which is both faster and a fairer comparison: the
    covariate-only and the calibrated estimators then differ by the calibration
    alone rather than by two independent LASSO fits.

    ``targets`` overrides the register prevalences of the anchored outcomes, which is
    how the target-error benchmark perturbs them.
    """
    if method not in WEIGHTING_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {WEIGHTING_METHODS}")
    mask = pop.mask
    n = int(mask.sum())
    Y_sel = pop.Y[mask]

    if method == "naive":
        return Weighting(method, np.full(n, 1.0 / n))
    if method == "oracle":
        w = 1.0 / pop.pi[mask]
        return Weighting(method, w / w.sum())

    needs_base = method.startswith("ipw")
    if needs_base:
        p = propensity(pop) if p_hat is None else p_hat
        base = 1.0 / p
    else:
        base = np.ones(n)

    if method == "ipw":
        return Weighting(method, base / base.sum())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", CalibrationWarning)
        if method == "ipw+cal/s":
            anchored = Y_sel[:, list(anchors)]
            within = pop.within_stratum_prevalence()[:, list(anchors)]
            w, diag = stratified_calibration_weights(
                anchored, pop.stratum[mask], within, pop.stratum_share,
                base_weights=base, shrinkage=shrinkage, warn=False, return_diagnostics=True,
            )
            features = stratified_features(anchored, pop.stratum[mask],
                                            len(pop.stratum_share))
            cons_targets = None
        else:
            if method == "ipw+cal/v":
                columns = pop.severity_columns()
                features = columns[mask]
                cons_targets = columns.mean(axis=0)
            else:
                features = Y_sel[:, list(anchors)]
                cons_targets = pop.population_prevalence[list(anchors)]
            if targets is not None:
                cons_targets = np.asarray(targets, dtype=float)
            w, diag = entropy_balance(
                features, cons_targets, base_weights=base,
                ridge=shrinkage, warn=False, return_diagnostics=True,
            )
    converged = bool(diag.converged) and not any(
        issubclass(c.category, CalibrationWarning) for c in caught
    )
    return Weighting(
        method, w, features=features, targets=cons_targets, converged=converged,
        extra={"diagnostics": diag, "max_abs_residual": float(diag.max_abs_residual)},
    )


def stratified_features(Y: np.ndarray, strata: np.ndarray, n_strata: int) -> np.ndarray:
    """The constrained columns of a stratified solve, for the calibration-aware SE."""
    onehot = (strata[:, None] == np.arange(n_strata)[None, :]).astype(float)
    cols = [onehot[:, :-1]]
    cols += [(onehot[:, a] * Y[:, q])[:, None] for a in range(n_strata) for q in range(Y.shape[1])]
    return np.hstack(cols)


# --------------------------------------------------------------------------------
# Metrics. Each takes a weighting and returns one number that a table can hold.
# --------------------------------------------------------------------------------

def weighted_mean(values: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * values) / np.sum(w))


def trait_error(pop: Population, wt: Weighting) -> float:
    """Signed error of the weighted trait mean, in units of the trait's population SD.

    ``Z`` is observed on participants only and is never a calibration constraint, so
    this is a held-out estimand throughout: it is the number that can be wrong.
    """
    est = weighted_mean(pop.Z[pop.mask], wt.weights)
    return (est - pop.trait_mean) / pop.trait_sd


def unanchored_prevalence_error(pop: Population, wt: Weighting, outcome: int = 1) -> float:
    """Relative error (%) on a disease whose prevalence was never supplied."""
    truth = pop.population_prevalence[outcome]
    est = weighted_mean(pop.Y[pop.mask, outcome], wt.weights)
    return (est - truth) / truth * 100.0


def held_out_balance(pop: Population, wt: Weighting, *, n_covariates: int = 4):
    """Balance against register margins the calibration was not given.

    The anchored prevalence goes in flagged as constrained, so the report shows the
    tautology next to the diagnostics rather than mixing them.
    """
    mask = pop.mask
    cols = [pop.Y[mask, 0][:, None], pop.X[mask][:, :n_covariates]]
    targets = [pop.population_prevalence[0], *pop.X[:, :n_covariates].mean(axis=0)]
    names = ["Y1 (anchored)", *[f"X{j}" for j in range(n_covariates)]]
    constrained = [True, *[False] * n_covariates]
    return balance_report(
        np.hstack(cols), wt.weights, np.asarray(targets, dtype=float),
        constrained=np.asarray(constrained), names=names,
    )


def case_mix_error(pop: Population, wt: Weighting) -> float:
    """Error in mean liability *among cases*: the case mix a pooled margin cannot fix.

    Prevalence pins how many cases the reweighted sample carries. It says nothing
    about how ill they are. Under severity-dependent recruitment the sampled cases
    are the sicker ones, and a pooled calibration reproduces the count while leaving
    the severity distribution wrong.
    """
    mask = pop.mask
    case = pop.Y[mask, 0] == 1
    if not np.any(case):
        return float("nan")
    truth = float(pop.liability[pop.Y[:, 0] == 1].mean())
    est = weighted_mean(pop.liability[mask][case], wt.weights[case])
    return est - truth

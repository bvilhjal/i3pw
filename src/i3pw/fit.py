"""The entry point for a real cohort: calibrate weights from plain arrays.

Everything in :mod:`i3pw.dgm`, :mod:`i3pw.methods` and :func:`i3pw.calibration_ipw`
takes a :class:`i3pw.Dataset` — a *simulated* population that carries the ground-truth
coefficients and computes its own population prevalence from outcomes observed on
everybody. A real biobank has none of that. It has outcomes on the people who
participated, prevalences from a register, and possibly a fitted participation model.
:func:`calibrate` is the same estimator over exactly those inputs::

    fit = i3pw.calibrate(Y_sample, targets=[0.012], base_weights=1 / p_hat)
    fit.weights                      # per-participant weights, summing to 1
    fit.mean(bmi).summary()          # a weighted mean with a calibration-aware SE

The two things the package argues you must do are arguments here rather than a later
assembly job, because that is the difference between advice and a default:

- ``strata=`` calibrates prevalence *within* sex / birth year / ancestry / severity
  rather than only the pooled margin. A known prevalence fixes the number of cases,
  not their type (``docs/theory.md#prevalence-sets-the-scale-not-the-case-mix``).
- ``holdout=`` names population quantities the calibration is *not* given and checks
  the reweighted sample against them. Constrained moments match by construction and
  are not diagnostics. A held-out mismatch reveals misspecification, but agreement is
  only a necessary check and does not validate the weighting model
  (``docs/theory.md#what-makes-this-falsifiable``).

Getting base weights in. The base weights are the covariate-driven half of the
correction and come from your own participation model — any model, fitted however you
like, on whatever frame you have. Pass ``1/P̂`` directly, or hand the predicted
probabilities to :func:`inverse_probability_weights` to get the scheme right. Omit them
and the calibration starts from uniform weights, which is the pure label-shift
correction: right when selection acts only through the outcome, and silent about the
covariate-driven part when it does not.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ._links import clip_prob
from .balance import BalanceReport, balance_report
from .calibration import (
    CalibrationDiagnostics,
    CalibrationWarning,
    _marginal_design,
    _stratified_design,
    apply_tilt,
    calibration_mean_se,
    effective_sample_size,
    entropy_balance,
)


def inverse_probability_weights(
    propensity: npt.ArrayLike,
    *,
    scheme: str = "inverse",
) -> np.ndarray:
    """Turn fitted participation probabilities ``P(S = 1 | X)`` into base weights.

    A one-liner, provided so the ``scheme`` choice is made explicitly rather than
    open-coded as ``1 / p`` at every call site:

    - ``"inverse"`` — ``1/P``, the Horvitz-Thompson weight that transports participants
      to the full target population.
    - ``"odds"`` — ``(1-P)/P``, the inverse-*odds* weight that transports participants
      to the *nonparticipant* population. It approximates full-population IPW only when
      participation is rare; it is not an exact substitute for ``1/P`` merely because a
      logistic model was used. See "Inverse vs odds base weights" in
      ``docs/theory.md#what-is-identified``.

    Probabilities are clipped away from 0 and 1, so a participation model that predicts
    certainty for some unit yields a large weight rather than an infinite one.
    """
    if scheme not in ("inverse", "odds"):
        raise ValueError("scheme must be 'inverse' or 'odds'.")
    p = clip_prob(np.asarray(propensity, dtype=float).ravel())
    return 1.0 / p if scheme == "inverse" else (1.0 - p) / p


@dataclass
class CalibrationFit:
    """Weights from :func:`calibrate`, with the constraints that produced them.

    Carrying the constraint matrix rather than just the weights is what lets
    :meth:`mean` compute a calibration-aware standard error and :meth:`apply_to`
    transfer the fitted tilt to new rows, neither of which the weights alone can
    support.
    """

    weights: np.ndarray
    """Length-``n`` calibration weights for the sampled units, summing to 1."""

    diagnostics: CalibrationDiagnostics
    """Convergence, residual and weight-concentration report for the solve."""

    constraint_features: np.ndarray
    """``(n, k)`` calibration functions that were actually constrained."""

    constraint_targets: np.ndarray
    """Length-``k`` population values those columns were matched to."""

    constraint_names: list[str] = field(default_factory=list)
    """Labels for the constrained columns, in column order."""

    balance: BalanceReport | None = None
    """Held-out balance check, when ``holdout=`` was supplied — otherwise ``None``.

    ``None`` is not a pass. It means nothing was tested.
    """

    unreachable: list[str] = field(default_factory=list)
    """Targets with no support in the sample, which no reweighting can meet."""

    shrinkage: float = 0.0
    """Ridge used to fit the tilt; needed for the penalized influence function."""

    @property
    def ess(self) -> float:
        """Kish effective sample size of the weights."""
        return effective_sample_size(self.weights)

    @property
    def tilt(self) -> np.ndarray | None:
        """Coefficients of the fitted log population-to-participant density ratio."""
        return self.diagnostics.tilt

    def mean(self, values: npt.ArrayLike, *, level: float = 0.95):
        """Weighted mean of ``values`` with a standard error that accounts for the tilt.

        Delegates to :func:`i3pw.calibration_mean_se` with the constraints this fit
        actually used, including the ridge stored in :attr:`shrinkage`. An exactly
        constrained margin (zero shrinkage) gets an SE of approximately zero. A ridge-
        shrunken margin retains uncertainty and uses the penalized influence coefficient
        ``(Cov(g) + shrinkage I)^-1 Cov(g, y)``.
        """
        return calibration_mean_se(
            values, self.weights, self.constraint_features,
            ridge=self.shrinkage, level=level,
        )

    def apply_to(
        self,
        features: npt.ArrayLike,
        *,
        base_weights: np.ndarray | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Weight fresh rows under this already-fitted tilt (see :func:`i3pw.apply_tilt`).

        ``features`` must carry the same calibration functions, in the same column order
        — :attr:`constraint_names` records what those were.
        """
        if self.diagnostics.tilt is None:
            raise ValueError("this fit carries no tilt coefficients to apply.")
        return apply_tilt(
            features, self.diagnostics.tilt, self.constraint_targets,
            base_weights=base_weights, normalize=normalize,
        )

    def summary(self) -> str:
        """Solve diagnostics followed by the held-out specification diagnostic."""
        lines = [
            f"calibrated on {len(self.constraint_names)} constraint(s): "
            + ", ".join(self.constraint_names),
            self.diagnostics.summary(),
        ]
        if self.unreachable:
            lines.append(f"  UNREACHABLE targets (no sample support): {self.unreachable}")
        lines.append("")
        if self.balance is None:
            lines.append(
                "no holdout supplied: every known quantity was used as a constraint, so "
                "the weights reproduce their targets by construction and nothing here "
                "checks the density-ratio specification. Pass "
                "holdout={name: (values, population_mean)} with a register margin you "
                "did NOT calibrate on to obtain a specification diagnostic."
            )
        else:
            lines.append(self.balance.summary())
        return "\n".join(lines)


def calibrate(
    Y: npt.ArrayLike,
    targets: npt.ArrayLike,
    *,
    base_weights: npt.ArrayLike | None = None,
    strata: npt.ArrayLike | None = None,
    stratum_share: npt.ArrayLike | None = None,
    joint_prevalences: dict | None = None,
    shrinkage: float = 0.0,
    holdout: dict | None = None,
    outcome_names: list[str] | None = None,
    warn: bool = True,
) -> CalibrationFit:
    """Calibrate weights for a real sample to known population prevalences.

    The array-level counterpart of :func:`i3pw.calibration_ipw`, which needs a simulated
    :class:`i3pw.Dataset`. Everything here is something a cohort actually has.

    Parameters
    ----------
    Y:
        ``(n, Q)`` 0/1 outcomes for the **sampled** units only. A 1-D array is read as a
        single outcome.
    targets:
        The known population prevalences. Length ``Q`` normally; ``(A, Q)`` when
        ``strata`` is given, holding ``P(Y_q = 1 | stratum = a)``.
    base_weights:
        Length-``n`` weights from a participation model — ``1/P̂``, or the output of
        :func:`inverse_probability_weights`. Use its default inverse scheme for a full-
        population target; the odds scheme instead targets nonparticipants and is only a
        rare-participation approximation to population IPW. Defaults to uniform, i.e.
        pure calibration with no covariate-driven correction.
    strata:
        Length-``n`` integer labels in ``0..A-1``. Supplying them calibrates prevalence
        *within* each stratum and pins the stratum shares, instead of matching only the
        pooled margin. Requires ``stratum_share`` and a ``(A, Q)`` ``targets``.
    stratum_share:
        Length-``A`` known population share of each stratum, ``P(stratum = a)``
        (normalized internally).
    joint_prevalences:
        Optional ``{(q, q'): P(Y_q = 1, Y_q' = 1)}`` co-occurrence targets, for when
        selection couples the outcomes rather than acting on each margin separately.
        Unstratified designs only.
    shrinkage:
        Ridge on the tilt. ``0`` calibrates exactly; larger values trade exactness for
        less extreme weights.
    holdout:
        ``{name: (values, population_mean)}`` — quantities whose population value you
        know but are deliberately *not* calibrating on. ``values`` is length ``n``. These
        become :attr:`CalibrationFit.balance`. A mismatch diagnoses a failure, while
        agreement is not proof that the weighting model is correct.
    outcome_names:
        Optional labels for the columns of ``Y``, used in the constraint names and the
        balance report.
    warn:
        Emit :class:`i3pw.CalibrationWarning` on unreachable targets, non-convergence,
        or too few units per constraint.

    Returns
    -------
    CalibrationFit
        Weights, diagnostics, the constraints used, and the held-out balance report.

    Examples
    --------
    A cohort that over-recruited cases, anchored to a register prevalence and tested
    against a register age margin it was not given::

        fit = i3pw.calibrate(
            Y=case_status[:, None],
            targets=[0.012],
            base_weights=i3pw.inverse_probability_weights(p_hat),
            holdout={"mean age": (age, 41.7), "% female": (female, 0.508)},
        )
        print(fit.summary())          # solve + held-out specification diagnostics
        print(fit.mean(bmi).summary())
    """
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr[:, None]
    if Y_arr.ndim != 2:
        raise ValueError("Y must be 1-D (one outcome) or 2-D (n, Q).")
    n = Y_arr.shape[0]
    t_arr = np.asarray(targets, dtype=float)

    if strata is None:
        if stratum_share is not None:
            raise ValueError("stratum_share was given without strata; pass both or neither.")
        if t_arr.ndim > 1:
            raise ValueError(
                "targets must be 1-D (one entry per outcome) when no strata are given; "
                "a 2-D targets array is only meaningful with strata=."
            )
        features, cons_targets, names, unreachable = _marginal_design(
            Y_arr, np.atleast_1d(t_arr), joint_prevalences, labels=outcome_names
        )
    else:
        if stratum_share is None:
            raise ValueError("strata require stratum_share (the population share of each).")
        if joint_prevalences:
            raise ValueError(
                "joint_prevalences is not supported with strata; encode the co-occurrence "
                "as an outcome column, or stratify on it."
            )
        if t_arr.ndim != 2:
            raise ValueError(
                "with strata, targets must be the (A, Q) within-stratum prevalences "
                "P(Y_q = 1 | stratum = a), not the pooled margin."
            )
        features, cons_targets, names, unreachable = _stratified_design(
            Y_arr, strata, t_arr, stratum_share, labels=outcome_names
        )

    if warn and unreachable:
        warnings.warn(
            f"calibrate: target(s) {unreachable} have no support in the sample; no "
            "reweighting can meet them. Drop them, or coarsen until they have cases.",
            CalibrationWarning, stacklevel=2,
        )

    bw = None if base_weights is None else np.asarray(base_weights, dtype=float).ravel()
    if bw is not None and bw.shape[0] != n:
        raise ValueError("base_weights must have one entry per row of Y.")

    w, diag = entropy_balance(
        features, cons_targets, base_weights=bw, ridge=shrinkage,
        warn=warn, return_diagnostics=True,
    )

    report = None
    if holdout:
        held_names = list(holdout)
        held_cols, held_targets = [], []
        for name in held_names:
            values, target = holdout[name]
            col = np.asarray(values, dtype=float).ravel()
            if col.shape[0] != n:
                raise ValueError(
                    f"holdout[{name!r}] has {col.shape[0]} values for {n} rows of Y."
                )
            held_cols.append(col[:, None])
            held_targets.append(float(target))
        # Constrained columns go in too, flagged: the report then shows what was pinned
        # next to the held-out diagnostics; balance_report's summary already excludes
        # the former from its check. Seeing both makes "matched by construction" plain.
        report = balance_report(
            np.hstack([features, *held_cols]),
            w,
            np.concatenate([cons_targets, np.asarray(held_targets, dtype=float)]),
            constrained=np.concatenate([
                np.ones(features.shape[1], dtype=bool),
                np.zeros(len(held_names), dtype=bool),
            ]),
            names=names + held_names,
        )

    return CalibrationFit(
        weights=w,
        diagnostics=diag,
        constraint_features=features,
        constraint_targets=cons_targets,
        shrinkage=float(shrinkage),
        constraint_names=names,
        balance=report,
        unreachable=unreachable,
    )

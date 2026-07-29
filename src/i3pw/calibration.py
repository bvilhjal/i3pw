"""Prevalence-calibrated inverse-probability weighting — the core i3pw idea.

The project's motivating observation: the standard approach of predicting
participation probabilities ``P(selected | X)`` from socioeconomic covariates
(e.g. LASSO, :func:`i3pw.methods.lasso_propensity`) works poorly for many
*disease* outcomes, because who participates depends on *having the disease* —
a signal largely orthogonal to the covariates. Decompose the selection log-odds
as ``a(X) + theta . Y``: the covariate model can learn ``a(X)`` but not the
disease term ``theta . Y``.

If the population prevalences ``Pr(Y_q)`` are known a priori (from a registry or
census), they supply exactly that missing term. This module injects them the
principled way — as **calibration constraints on the weighted outcome**, so the
reweighted sample reproduces the known prevalences exactly:

    find weights w minimizing KL(w || base)
    s.t.  sum_i w_i Y_iq / sum_i w_i = Pr(Y_q)   for each anchored outcome q

This splits selection correction into two separable tasks the package keeps apart:
(1) *predict* individual selection with a participation model ``P(S | X)`` (the base
weights; the predictors may be demographic, clinical, or genetic), and (2) *anchor*
the weighted sample to the target population by calibrating to known register
quantities — the overall prevalence and, via :func:`stratified_calibration_weights`,
prevalence within strata.

The solution is exponential tilting, ``w_i ∝ base_i * exp(lambda . (Y_i - Pr))``,
with ``lambda`` from a small convex dual (entropy balancing; Hainmueller 2012,
Deville & Sarndal 1992).

What this identifies, in one paragraph
--------------------------------------
The result is a **density-ratio** model, not a recovered per-unit inclusion
probability: calibration returns the minimum-divergence weights
``base(X) * exp(lambda . g(Y))`` matching the supplied moments ``g(Y)``, which
coincide with the true inverse-probability weights only when the population-to-sample
density ratio genuinely lies in that tilt family. Two consequences drive the API: an
anchored margin is reproduced *by construction* and is therefore not evidence, and the
tilt-family assumption is testable only against population quantities the calibration
did not use (:func:`i3pw.balance_report`, wired in as ``holdout=`` on
:func:`i3pw.calibrate`).

``docs/theory.md`` is the canonical statement of all of this and is the file to edit
if it changes — the derivation, the placement among density-ratio / I-projection /
label-shift results, why anchored margins get a zero standard error, and which
robustness claim is and is not being made:

- what the weights identify and when they equal true IPW:
  ``docs/theory.md#what-is-identified``
- why ``base_scheme="odds"`` composes exactly with the tilt and ``"inverse"`` only
  approximately: same section, "Inverse vs odds base weights"
- the GREG equivalence behind :func:`calibration_mean_se`, and the Zhao & Percival
  double-robustness claim i3pw's caveat is *not* about:
  ``docs/theory.md#calibration-is-a-regression-estimator-and-why-the-ses-look-the-way-they-do``
- why held-out moments are the only falsification available:
  ``docs/theory.md#what-makes-this-falsifiable``
- why a known prevalence fixes the number of cases but not their type:
  ``docs/theory.md#prevalence-sets-the-scale-not-the-case-mix``
- the bibliography for every result named above: ``docs/theory.md#references``
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .dgm import Dataset
from .methods import MethodResult, _test_arrays, _trim_weights, lasso_propensity
from .metrics import percent_difference, weighted_prevalence


class CalibrationWarning(UserWarning):
    """Warned when calibration weights are unreliable (non-convergence, infeasible target)."""


_MIN_UNITS_PER_CONSTRAINT = 10
"""Below this many units per calibration constraint, :func:`entropy_balance` warns.

A rule of thumb, not a threshold with theory behind it: the tilt is a ``k``-parameter
fit to ``n`` units, and the survey-calibration convention is to want many units backing
each constraint. Ten is the round number that flags a stratified design whose cells have
outrun its sample without firing on ordinary use.
"""


def _require_finite(a: np.ndarray, name: str) -> None:
    """Raise if ``a`` holds NaN or infinity.

    Without this, a single missing value silently propagates through the exponential
    tilt and yields an all-NaN weight vector whose diagnostics report ``ESS 0.0`` and
    blame the optimizer — a confusing failure a long way from its cause.
    """
    if not np.all(np.isfinite(a)):
        raise ValueError(
            f"{name} contains NaN or infinite values; drop or impute them before "
            "calibrating (missing values cannot be reweighted)."
        )


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` (ignoring zero weights)."""
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    denom = np.sum(w**2)
    return float(np.sum(w) ** 2 / denom) if denom > 0 else 0.0


def _weight_concentration(weights: np.ndarray) -> tuple[float, float, float, float]:
    """``(ess, max_weight, min_weight, top1pct_mass)`` for a weight vector.

    ``top1pct_mass`` is the fraction of total weight carried by the largest 1% of
    units — a blunt read on how badly the weighting leans on a handful of rows.
    """
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    ess = effective_sample_size(w)
    n = w.size
    k = max(1, int(np.ceil(0.01 * n)))
    top = float(np.sort(w)[::-1][:k].sum())
    return ess, float(w.max()), float(w.min()), (top / total if total > 0 else 0.0)


@dataclass
class CalibrationDiagnostics:
    """Convergence and stability diagnostics for a set of calibration weights.

    ``converged`` and ``max_abs_residual`` describe whether the calibration targets
    were actually met (``max_abs_residual`` is the largest ``|weighted mean - target|``
    over the constraints — with ``ridge > 0`` it is expected to be non-zero). The
    remaining fields describe how concentrated the weights are: a low ``ess`` or a
    large ``top1pct_weight_mass`` flags a fragile, high-variance weighting.
    """

    converged: bool
    n_iter: int
    max_abs_residual: float
    ess: float
    max_weight: float
    min_weight: float
    top1pct_weight_mass: float
    message: str = ""
    tilt: np.ndarray | None = None
    """The fitted dual coefficients ``lambda``, one per calibration constraint.

    For the simplest case these have a closed form worth carrying in your head. With one
    binary outcome, a uniform base, sample prevalence ``P`` and target ``K``, the single
    coefficient is the log odds-ratio between the two::

        lambda = log[ (K/(1-K)) / (P/(1-P)) ]

    which yields weights proportional to ``K/P`` for cases and ``(1-K)/(1-P)`` for
    controls — the classical choice-based-sample weights (Manski & Lerman 1977). So
    ``lambda`` measures how hard the sample had to be pushed to reach the register, in
    log-odds. With several constraints or a non-uniform base there is no closed form, but
    the reading is the same.

    These *are* the estimated outcome-driven part of the selection log-odds: the weights
    are ``base_i * exp(lambda . (g_i - target))``, so ``lambda`` is the ``theta`` of the
    package's ``a(X) + theta . g(Y)`` decomposition, on the log-odds scale and identified
    up to the constraints supplied. Large components mean the sample needed a hard push
    on that moment. Pass them to :func:`apply_tilt` to weight a fresh set of rows under
    the same fitted calibration.
    """

    def summary(self) -> str:
        ok = "converged" if self.converged else "DID NOT CONVERGE"
        lines = [
            f"calibration diagnostics: {ok}"
            + (f" ({self.message})" if self.message else ""),
            f"  max abs calibration residual : {self.max_abs_residual:.3e}",
            f"  effective sample size (Kish) : {self.ess:.1f}",
            f"  weight range [min, max]      : [{self.min_weight:.3e}, {self.max_weight:.3e}]",
            f"  top-1% units carry           : {100 * self.top1pct_weight_mass:.1f}% of weight",
        ]
        return "\n".join(lines)


def _diagnostics(weights: np.ndarray, converged: bool, n_iter: int,
                 max_abs_residual: float, message: str = "",
                 tilt: np.ndarray | None = None) -> CalibrationDiagnostics:
    ess, wmax, wmin, top = _weight_concentration(weights)
    return CalibrationDiagnostics(
        converged=converged, n_iter=n_iter, max_abs_residual=max_abs_residual,
        ess=ess, max_weight=wmax, min_weight=wmin, top1pct_weight_mass=top, message=message,
        tilt=tilt,
    )


def base_weights(
    base: str,
    base_scheme: str,
    X_train: np.ndarray,
    s_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    interactions: bool = False,
    cv: int = 5,
) -> np.ndarray:
    """Base weights for the sampled units: uniform, or a covariate participation model.

    ``base="uniform"`` returns ones (pure calibration). ``base="lasso"`` fits
    :func:`i3pw.methods.lasso_propensity` ``P(selected | X)`` on the training frame and
    inverts it, using ``base_scheme="inverse"`` (``1/P``) or ``"odds"`` (``(1-P)/P``,
    which composes exactly with the exponential-tilt calibration; see "What is
    identified?" in ``docs/theory.md``). Shared by :func:`calibration_ipw` and the bootstrap so the
    two cannot drift.
    """
    if base not in ("lasso", "uniform"):
        raise ValueError("base must be 'lasso' or 'uniform'.")
    if base_scheme not in ("inverse", "odds"):
        raise ValueError("base_scheme must be 'inverse' or 'odds'.")
    if base == "uniform":
        return np.ones(X_eval.shape[0])
    P = lasso_propensity(X_train, s_train, X_eval, interactions=interactions, cv=cv)
    return 1.0 / P if base_scheme == "inverse" else (1.0 - P) / P


def entropy_balance(
    features: npt.ArrayLike,
    targets: npt.ArrayLike,
    *,
    base_weights: np.ndarray | None = None,
    ridge: float = 0.0,
    max_iter: int = 500,
    tol: float = 1e-6,
    warn: bool = True,
    return_diagnostics: bool = False,
):
    """Entropy-balancing weights matching each feature column's weighted mean to a target.

    Solves ``min_w sum_i base_i KL(w_i / base_i)`` subject to
    ``sum_i w_i (features_i - targets) = 0`` and ``sum_i w_i = 1`` via the convex
    dual ``min_lambda log sum_i base_i exp(lambda . (features_i - targets))`` (plus
    an optional ridge ``(ridge/2)||lambda||^2`` that shrinks toward the base
    weights, trading exact calibration for lower variance).

    Parameters
    ----------
    features:
        ``(n, k)`` array; each column is a quantity to calibrate.
    targets:
        Length-``k`` desired weighted means.
    base_weights:
        Length-``n`` non-negative base weights (e.g. covariate-model IPW weights);
        defaults to uniform.
    ridge:
        Non-negative shrinkage. ``0`` calibrates exactly; larger values pull the
        weights back toward ``base_weights``.
    tol:
        Residual tolerance for the feasibility warning. With ``ridge == 0`` a final
        ``max_abs_residual > tol`` means the target was not reached — usually because
        it lies outside the sample's convex hull (an exponential tilt cannot get
        there). Ignored when ``ridge > 0`` (the residual is non-zero by design).
    warn:
        Emit a :class:`CalibrationWarning` on non-convergence or an unmet target.
    return_diagnostics:
        If ``True`` return ``(weights, diagnostics)`` where ``diagnostics`` is a
        :class:`CalibrationDiagnostics`; otherwise return the weights only.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, CalibrationDiagnostics]
        Weights of length ``n`` summing to 1 (and diagnostics if requested).
    """
    F = np.atleast_2d(np.asarray(features, dtype=float))
    if F.shape[0] == 1 and F.shape[1] != len(np.atleast_1d(targets)):
        F = F.T
    t = np.atleast_1d(np.asarray(targets, dtype=float))
    n, k = F.shape
    if k != t.shape[0]:
        raise ValueError("features must have one column per target.")
    _require_finite(F, "features")
    _require_finite(t, "targets")

    d = np.ones(n) if base_weights is None else np.asarray(base_weights, dtype=float)
    _require_finite(d, "base_weights")
    if np.any(d < 0):
        raise ValueError("base_weights must be non-negative.")
    if d.sum() == 0:
        raise ValueError("base_weights sum to zero; cannot form calibration weights.")
    d = d / d.sum()

    if k == 0:
        # No constraints: the base weights already solve the problem exactly. Skip the
        # dual solve, which would otherwise report a spurious "ERROR: N <= 0" failure.
        if return_diagnostics:
            return d, _diagnostics(d, True, 0, 0.0, "", tilt=np.zeros(0))
        return d

    # The tilt spends one free parameter per constraint. Past k == n it can reproduce
    # any target vector whatever the data, so the calibration carries no information;
    # well before that the weights start resting on a handful of units per constraint
    # and the effective sample size collapses. Neither is visible in the residual --
    # the solve reports success either way -- so it has to be said here. This is the
    # failure mode a stratified design walks into: A strata x Q outcomes plus A - 1
    # share constraints grows much faster than the cells that support it.
    if k >= n:
        raise ValueError(
            f"entropy_balance: {k} constraints for {n} units. The tilt has at least one "
            "free parameter per unit, so it reproduces any target exactly regardless of "
            "the data and identifies nothing. Coarsen the constraints (fewer strata, "
            "margins instead of cells) so that constraints are far fewer than units."
        )
    if warn and n < _MIN_UNITS_PER_CONSTRAINT * k:
        warnings.warn(
            f"entropy_balance: {k} constraints for {n} units "
            f"(< {_MIN_UNITS_PER_CONSTRAINT} units per constraint). The targets will be "
            "met, but on very little data per constraint — expect a small effective "
            "sample size and unstable weights. Coarsen the constraints or set a positive "
            "ridge/shrinkage.",
            CalibrationWarning, stacklevel=2,
        )

    H = F - t  # centered constraints; we want the weighted mean of H to be 0

    def objective(lam):
        z = H @ lam
        m = z.max()
        ew = d * np.exp(z - m)
        Z = ew.sum()
        p = ew / Z
        f = m + np.log(Z) + 0.5 * ridge * lam @ lam
        grad = H.T @ p + ridge * lam
        return f, grad

    res = minimize(
        objective, np.zeros(k), jac=True, method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
    )
    z = H @ res.x
    w = d * np.exp(z - z.max())
    w = w / w.sum()

    max_abs_residual = float(np.max(np.abs(w @ F - t))) if k else 0.0
    message = "" if res.success else str(res.message)
    if warn:
        # The unmet-target diagnosis is the *informative* one — an unreachable target
        # is exactly the case where the optimizer also fails to converge, so it must
        # not be hidden behind `res.success`.
        if ridge == 0.0 and max_abs_residual > tol:
            warnings.warn(
                f"entropy_balance: calibration targets not met (max residual "
                f"{max_abs_residual:.2e} > tol {tol:.1e}); the target likely lies outside "
                "the sample's convex hull (e.g. an anchored outcome with no cases sampled).",
                CalibrationWarning, stacklevel=2,
            )
        elif not res.success:
            warnings.warn(
                f"entropy_balance: optimizer did not converge ({message!r}); "
                "weights may be unreliable.",
                CalibrationWarning, stacklevel=2,
            )

    if return_diagnostics:
        return w, _diagnostics(w, res.success, int(res.nit), max_abs_residual, message,
                               tilt=np.asarray(res.x, dtype=float))
    return w


def apply_tilt(
    features: npt.ArrayLike,
    tilt: npt.ArrayLike,
    targets: npt.ArrayLike,
    *,
    base_weights: np.ndarray | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Apply an already-fitted calibration to new rows.

    :func:`entropy_balance` solves for the dual coefficients ``lambda`` on one sample.
    Those coefficients *are* the fitted density-ratio model, so they transfer: this
    evaluates ``base_i * exp(lambda . (g_i - target))`` on any rows carrying the same
    calibration functions ``g``.

    Use it to weight a held-out fold, to score newly recruited participants without
    re-solving, or to check a calibration fitted on one wave against another. Note that
    the transferred weights reproduce the targets only insofar as the new rows follow the
    same density ratio — they are *not* re-calibrated, so their achieved moments will
    differ from ``targets`` by ordinary sampling error (which is exactly what makes this
    a usable check rather than a tautology).

    Parameters
    ----------
    features:
        ``(m, k)`` calibration functions evaluated on the new rows, in the same column
        order used when the tilt was fitted.
    tilt:
        Length-``k`` dual coefficients, i.e. ``CalibrationDiagnostics.tilt``.
    targets:
        The length-``k`` targets the tilt was fitted against (the centring constants).
    base_weights:
        Optional length-``m`` base weights for the new rows; defaults to uniform.
    normalize:
        If ``True`` (default) the returned weights sum to 1.
    """
    F = np.atleast_2d(np.asarray(features, dtype=float))
    lam = np.atleast_1d(np.asarray(tilt, dtype=float))
    t = np.atleast_1d(np.asarray(targets, dtype=float))
    if F.shape[0] == 1 and F.shape[1] != lam.shape[0]:
        F = F.T
    if F.shape[1] != lam.shape[0]:
        raise ValueError("features must have one column per tilt coefficient.")
    if t.shape[0] != lam.shape[0]:
        raise ValueError("targets must have one entry per tilt coefficient.")
    _require_finite(F, "features")
    _require_finite(lam, "tilt")

    d = np.ones(F.shape[0]) if base_weights is None else np.asarray(base_weights, dtype=float)
    _require_finite(d, "base_weights")
    if np.any(d < 0):
        raise ValueError("base_weights must be non-negative.")
    z = (F - t) @ lam
    w = d * np.exp(z - z.max())  # shift is absorbed by the normalization
    if normalize:
        total = w.sum()
        if total == 0:
            raise ValueError("tilted weights sum to zero; the tilt underflowed.")
        w = w / total
    return w


def calibration_mean_se(
    values: npt.ArrayLike,
    weights: npt.ArrayLike,
    features: npt.ArrayLike,
    *,
    level: float = 0.95,
):
    """Standard error of a weighted mean that accounts for the *estimated* calibration.

    :func:`i3pw.weighted_mean_se` conditions on the weights, which is wrong for a
    calibration estimate in both directions — most starkly on an anchored margin, where
    it reports a positive SE for a quantity calibration reproduces exactly (true
    sampling variability zero).

    The calibration estimator is asymptotically a regression (GREG) estimator, so its
    influence function is the *residual* of the outcome on the calibration functions:

        e_i = y_i - mu - beta . (g_i - gbar),   beta = weighted LS of y on g
        Var(mu) = sum_i w_i^2 e_i^2 / (sum_i w_i)^2

    Constraining ``g`` removes exactly the part of ``y`` that ``g`` explains. An anchored
    outcome is itself a column of ``g``, so its residual is identically zero and the SE
    correctly collapses to ~0; an estimand orthogonal to the constraints reduces to the
    fixed-weight formula; anything in between is shrunk by the variance the calibration
    absorbed. This is the closed-form counterpart of
    :func:`i3pw.bootstrap_calibration_ipw`, without re-solving the dual per replicate.

    Parameters
    ----------
    values:
        Length-``n`` outcome whose weighted mean is being estimated.
    weights:
        Length-``n`` calibration weights.
    features:
        ``(n, k)`` calibration functions actually constrained when the weights were
        solved (for :func:`calibration_ipw`, the anchored outcome columns).

    Returns
    -------
    Estimate
        Point estimate, SE and normal-approximation interval.
    """
    from .uncertainty import Estimate  # local import: uncertainty imports this module

    y = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    G = np.atleast_2d(np.asarray(features, dtype=float))
    if G.shape[0] != y.shape[0] and G.shape[1] == y.shape[0]:
        G = G.T
    if y.shape != w.shape:
        raise ValueError("values and weights must have the same length.")
    if G.shape[0] != y.shape[0]:
        raise ValueError("features must have one row per value.")
    _require_finite(y, "values")
    _require_finite(w, "weights")
    _require_finite(G, "features")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    total = w.sum()
    if total == 0:
        raise ValueError("weights sum to zero.")

    p = w / total
    mu = float(p @ y)
    gbar = p @ G
    Gc = G - gbar
    # Weighted least squares of y on the centered constraints (lstsq handles collinear
    # or redundant constraint columns, which a normal-equation solve would not).
    sw = np.sqrt(p)
    beta, *_ = np.linalg.lstsq(Gc * sw[:, None], (y - mu) * sw, rcond=None)
    resid = (y - mu) - Gc @ beta
    var = float(np.sum(w**2 * resid**2) / total**2)
    se = float(np.sqrt(max(var, 0.0)))
    from scipy.stats import norm as _norm
    z = float(_norm.ppf(1.0 - (1.0 - level) / 2.0))
    return Estimate(value=mu, se=se, ci_low=mu - z * se, ci_high=mu + z * se, level=level)


def _marginal_design(
    Y: np.ndarray,
    prevalences,
    joint_prevalences=None,
    *,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build the ``(features, targets, names, unreachable)`` design for outcome margins.

    Shared by :func:`outcome_calibration_weights` and :func:`i3pw.calibrate` so the two
    front doors cannot drift into building different constraint matrices from the same
    arguments. ``unreachable`` names the targets no reweighting can meet (an outcome with
    no sampled cases, a co-occurrence never observed); the caller decides how to report
    them, since it knows what to call the columns.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    prev = np.asarray(list(prevalences), dtype=float)
    if Y.shape[0] == 1 and Y.shape[1] != prev.shape[0]:
        Y = Y.T
    if Y.shape[1] != prev.shape[0]:
        raise ValueError("prevalences must have one entry per outcome column of Y.")
    names = list(labels) if labels is not None else [f"Y{q}" for q in range(Y.shape[1])]
    if len(names) != Y.shape[1]:
        raise ValueError("labels must have one name per outcome column of Y.")

    counts = Y.sum(axis=0)
    unreachable = [
        names[q] for q in range(Y.shape[1])
        if (counts[q] == 0 and prev[q] > 0) or (counts[q] == Y.shape[0] and prev[q] < 1)
    ]

    cols = [Y]
    targets = list(prev)
    out_names = list(names)
    if joint_prevalences:
        for (q, qp), value in joint_prevalences.items():
            pattern = Y[:, q] * Y[:, qp]
            label = f"{names[q]}&{names[qp]}"
            if value > 0 and pattern.sum() == 0:
                unreachable.append(label)
            cols.append(pattern[:, None])
            targets.append(value)
            out_names.append(label)
    return np.hstack(cols), np.asarray(targets, dtype=float), out_names, unreachable


def _stratified_design(
    Y: np.ndarray,
    strata: np.ndarray,
    within_stratum_prevalence: np.ndarray,
    stratum_share: np.ndarray,
    *,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """``(features, targets, names, unreachable)`` for within-stratum calibration.

    Companion to :func:`_marginal_design`; see there for why these are shared.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    lab = np.asarray(strata).ravel()
    within = np.atleast_2d(np.asarray(within_stratum_prevalence, dtype=float))
    share = np.asarray(stratum_share, dtype=float).ravel()
    n, Q = Y.shape
    A = share.shape[0]
    if lab.shape[0] != n:
        raise ValueError("strata must have one label per row of Y.")
    if within.shape != (A, Q):
        raise ValueError("within_stratum_prevalence must have shape (A, Q).")
    if lab.min() < 0 or lab.max() >= A:
        raise ValueError("strata labels must lie in 0..A-1 (A = len(stratum_share)).")
    if np.any(share < 0):
        raise ValueError("stratum_share must be non-negative.")
    if share.sum() == 0:
        raise ValueError("stratum_share sums to zero; cannot normalize stratum shares.")
    share = share / share.sum()
    names = list(labels) if labels is not None else [f"Y{q}" for q in range(Q)]
    if len(names) != Q:
        raise ValueError("labels must have one name per outcome column of Y.")

    onehot = (lab[:, None] == np.arange(A)[None, :]).astype(float)  # (n, A)
    counts = onehot.sum(axis=0)
    joint = share[:, None] * within  # (A, Q) population joint P(Y_q = 1, A = a)

    unreachable = [f"stratum {a}" for a in range(A) if counts[a] == 0]
    unreachable += [
        f"{names[q]}|stratum {a}" for a in range(A) for q in range(Q)
        if joint[a, q] > 0 and float((onehot[:, a] * Y[:, q]).sum()) == 0.0
    ]

    # Drop the last stratum indicator: its share is implied by the others plus the
    # sum-to-one constraint, so keeping it would make the dual singular.
    cols = [onehot[:, :-1]]
    targets = list(share[:-1])
    out_names = [f"stratum {a}" for a in range(A - 1)]
    for a in range(A):
        for q in range(Q):
            cols.append((onehot[:, a] * Y[:, q])[:, None])
            targets.append(joint[a, q])
            out_names.append(f"{names[q]}|stratum {a}")
    return np.hstack(cols), np.asarray(targets, dtype=float), out_names, unreachable


def outcome_calibration_weights(
    Y: np.ndarray,
    prevalences,
    *,
    joint_prevalences=None,
    base_weights: np.ndarray | None = None,
    shrinkage: float = 0.0,
    warn: bool = True,
    return_diagnostics: bool = False,
):
    """Calibrate weights to known outcome margins — and optionally co-occurrences.

    For a sample ascertained on several outcomes with known population prevalences,
    the optimal weights come from **jointly** calibrating to all of them at once
    (not per-outcome weights combined heuristically). This solves for the unique
    exponential tilt reproducing every supplied moment.

    Parameters
    ----------
    Y:
        ``(n, Q)`` array of the sampled units' 0/1 outcomes.
    prevalences:
        Length-``Q`` known population marginal prevalences ``P(Y_q = 1)``.
    joint_prevalences:
        Optional dict ``{(q, q'): P(Y_q = 1, Y_q' = 1)}`` of known pairwise
        co-occurrence prevalences. Add these when the sampling *couples* the
        outcomes (interaction terms in the selection): marginals alone cannot
        represent an interaction, so calibrating on the co-occurrences is what
        restores exactness. Without coupling they are unnecessary.
    base_weights, shrinkage:
        Passed through to :func:`entropy_balance` (starting weights and ridge).
    warn, return_diagnostics:
        Passed through to :func:`entropy_balance`. Additionally, a
        :class:`CalibrationWarning` is raised for any marginal or co-occurrence
        target with no support in the sample (e.g. a co-occurrence ``(q, q')`` that
        is never observed), which no reweighting can reach.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, CalibrationDiagnostics]
        Calibration weights for the sampled units, summing to 1 (and diagnostics if
        requested).
    """
    features, targets, _, unreachable = _marginal_design(Y, prevalences, joint_prevalences)
    if warn and unreachable:
        warnings.warn(
            f"outcome_calibration_weights: target(s) {unreachable} have no support in the "
            "sample (an outcome with no cases or no controls, or a co-occurrence never "
            "observed); no reweighting can meet them.",
            CalibrationWarning, stacklevel=2,
        )
    return entropy_balance(
        features, targets, base_weights=base_weights,
        ridge=shrinkage, warn=warn, return_diagnostics=return_diagnostics,
    )


def stratified_calibration_weights(
    Y: np.ndarray,
    strata: np.ndarray,
    within_stratum_prevalence: np.ndarray,
    stratum_share: np.ndarray,
    *,
    base_weights: np.ndarray | None = None,
    shrinkage: float = 0.0,
    warn: bool = True,
    return_diagnostics: bool = False,
):
    """Calibrate to disease prevalence *within strata*, not just the pooled margin.

    A single pooled prevalence is often too crude: in registers and biobanks,
    prevalence varies strongly by sex, birth cohort, ancestry, region, or calendar
    time, and participation varies across those same strata. If prevalence is known
    *within* strata, calibrate to it directly. This matches, for every stratum ``a``
    and outcome ``q``, the joint moments

        E_w[1(A = a)]        = P(A = a)                  (stratum shares)
        E_w[Y_q · 1(A = a)]  = P(Y_q = 1, A = a)         (within-stratum prevalence)

    so the reweighted sample reproduces both the stratum sizes and the per-stratum
    disease prevalences. It reduces to :func:`outcome_calibration_weights` when there
    is a single stratum. Calibrating disease prevalence within covariate strata is
    also the natural way to reach past *marginal* selection toward the interaction
    structure that pure marginal calibration cannot represent.

    Parameters
    ----------
    Y:
        ``(n, Q)`` array of the sampled units' 0/1 outcomes.
    strata:
        Length-``n`` integer stratum labels in ``0..A-1`` (``A = len(stratum_share)``).
    within_stratum_prevalence:
        ``(A, Q)`` known within-stratum prevalences ``P(Y_q = 1 | A = a)``.
    stratum_share:
        Length-``A`` known population stratum shares ``P(A = a)`` (normalized internally).
    base_weights, shrinkage, warn, return_diagnostics:
        As in :func:`entropy_balance`. A :class:`CalibrationWarning` is raised for any
        stratum with no sampled units, or any ``(stratum, outcome)`` cell whose known
        prevalence is positive but which has no sampled case.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, CalibrationDiagnostics]
        Calibration weights for the sampled units, summing to 1 (and diagnostics if
        requested).
    """
    features, targets, _, unreachable = _stratified_design(
        Y, strata, within_stratum_prevalence, stratum_share
    )
    if warn and unreachable:
        warnings.warn(
            f"stratified_calibration_weights: target(s) {unreachable} have no support in "
            "the sample (an empty stratum, or a cell with positive known prevalence but no "
            "sampled case); no reweighting can meet them.",
            CalibrationWarning, stacklevel=2,
        )
    return entropy_balance(
        features, targets, base_weights=base_weights,
        ridge=shrinkage, warn=warn, return_diagnostics=return_diagnostics,
    )


@dataclass
class CalibrationResult:
    """Result of :func:`calibration_ipw`, extending :class:`MethodResult` semantics."""

    method_result: MethodResult
    anchor_outcomes: tuple[int, ...]
    ess: float
    achieved_prevalence: np.ndarray  # weighted prevalence of the anchored outcomes
    diagnostics: CalibrationDiagnostics | None = None
    support: dict[int, tuple[int, int]] = field(default_factory=dict)  # anchor -> (cases, controls)
    pre_trim_residual: float = 0.0   # max |achieved - target| before weight trimming
    post_trim_residual: float = 0.0  # ...and after (differs only when trim= is used)

    # The MethodResult surface is forwarded explicitly rather than through
    # ``__getattr__``. A catch-all delegator recursed infinitely whenever
    # ``method_result`` was not yet set on the instance — which is exactly what
    # ``copy.deepcopy`` and unpickling do — raising RecursionError on the package's
    # headline return type. Explicit properties also keep the surface type-checkable.

    @property
    def name(self) -> str:
        return self.method_result.name

    @property
    def weighted_prevalence(self) -> np.ndarray:
        return self.method_result.weighted_prevalence

    @property
    def percent_diff(self) -> np.ndarray:
        return self.method_result.percent_diff

    @property
    def population_prevalence(self) -> np.ndarray:
        return self.method_result.population_prevalence

    @property
    def extra(self) -> dict:
        return self.method_result.extra

    def summary(self) -> str:
        """Per-outcome estimate table (delegates to the underlying :class:`MethodResult`)."""
        return self.method_result.summary()

    def diagnostics_summary(self) -> str:
        """Human-readable convergence / support / stability report."""
        lines = [self.diagnostics.summary() if self.diagnostics else "calibration diagnostics: n/a"]
        if self.post_trim_residual > self.pre_trim_residual + 1e-9:
            lines.append(
                f"  trimming raised the residual {self.pre_trim_residual:.3e} -> "
                f"{self.post_trim_residual:.3e} (exact calibration no longer holds)"
            )
        for q, (n_case, n_ctrl) in self.support.items():
            lines.append(f"  anchor Y{q + 1} support: {n_case} cases / {n_ctrl} controls")
        return "\n".join(lines)


def calibration_ipw(
    dataset: Dataset,
    *,
    anchor_outcomes=None,
    base: str = "lasso",
    base_scheme: str = "inverse",
    shrinkage: float = 0.0,
    trim: float | None = None,
    interactions: bool = False,
    cv: int = 5,
) -> CalibrationResult:
    """Prevalence-calibrated IPW on a simulated :class:`~i3pw.Dataset`.

    Fits (optionally) a covariate participation model on the training fold to get
    base weights, then calibrates the sampled test units so their weighted
    prevalence matches the known population prevalence of each anchored outcome.
    The estimator is deployable — it uses the sampled units only.

    **This is the simulation front door.** It needs a :class:`~i3pw.Dataset`, which
    carries ground-truth coefficients and computes ``population_prevalence`` from
    outcomes observed on the whole population — things a real cohort does not have. It
    is a thin wrapper over :func:`i3pw.calibrate`, which does the same work from plain
    arrays and is what to call on real data.

    Parameters
    ----------
    anchor_outcomes:
        Indices of outcomes whose population prevalence is known and used as
        calibration targets. Defaults to all outcomes. Outcomes *not* listed are
        left free, so evaluating them measures how well calibrating on the known
        diseases transfers to an unknown one.
    base:
        ``"lasso"`` uses covariate-model IPW weights as the base (so the covariate-
        driven part of selection is handled by the base and the outcome-driven part
        by the calibration constraints); ``"uniform"`` starts from equal weights
        (pure calibration).
    base_scheme:
        ``"inverse"`` (``1/P``, inverse-probability) or ``"odds"`` (``(1-P)/P``) for
        the ``"lasso"`` base weights.
    shrinkage:
        Ridge on the tilt; ``0`` calibrates exactly, larger values shrink toward
        the base weights (bias-variance trade-off / stabilization).
    """
    if base not in ("lasso", "uniform"):
        raise ValueError("base must be 'lasso' or 'uniform'.")
    if base_scheme not in ("inverse", "odds"):
        raise ValueError("base_scheme must be 'inverse' or 'odds'.")

    X_train, _, s_train = dataset.split("train")
    X_test, Y_test, s_test = _test_arrays(dataset)
    pop = dataset.population_prevalence
    q = len(pop)
    anchors = tuple(range(q)) if anchor_outcomes is None else tuple(anchor_outcomes)

    from .fit import calibrate  # local import: fit builds on this module

    sel = s_test == 1
    Y_sel = Y_test[sel]

    # Positivity / support: each anchored binary outcome needs both classes present
    # in the sample, or its interior prevalence target is unreachable by reweighting.
    # Counted here because CalibrationResult reports it per anchor in the *dataset's*
    # outcome numbering; the warning itself comes from calibrate(), which is told the
    # same numbering via outcome_names so it can name the outcome rather than a column.
    support = {}
    for a in anchors:
        n_case = int(np.round(Y_sel[:, a].sum()))
        support[a] = (n_case, int(Y_sel.shape[0] - n_case))

    base_w = base_weights(
        base, base_scheme, X_train, s_train, X_test[sel], interactions=interactions, cv=cv
    )

    anchor_targets = pop[list(anchors)]
    fit = calibrate(
        Y_sel[:, anchors], anchor_targets, base_weights=base_w, shrinkage=shrinkage,
        outcome_names=[f"Y{a + 1}" for a in anchors],
    )
    w_sel, diag = fit.weights, fit.diagnostics
    pre = np.array([weighted_prevalence(w_sel, Y_sel[:, a]) for a in anchors])
    pre_trim_residual = float(np.max(np.abs(pre - anchor_targets))) if anchors else 0.0

    w_sel = _trim_weights(w_sel, trim)
    if trim is not None:
        post = np.array([weighted_prevalence(w_sel, Y_sel[:, a]) for a in anchors])
        post_trim_residual = float(np.max(np.abs(post - anchor_targets))) if anchors else 0.0
        if post_trim_residual > pre_trim_residual + 1e-9:
            warnings.warn(
                f"calibration_ipw: trim={trim} broke exact calibration "
                f"(max residual {pre_trim_residual:.2e} -> {post_trim_residual:.2e}).",
                CalibrationWarning, stacklevel=2,
            )
    else:
        post_trim_residual = pre_trim_residual

    # Scatter sample weights back over all test units (unselected get 0).
    w_full = np.zeros(Y_test.shape[0])
    w_full[sel] = w_sel

    est = np.array([weighted_prevalence(w_full, Y_test[:, j]) for j in range(q)])
    pdiff = np.array([percent_difference(est[j], pop[j]) for j in range(q)])
    result = MethodResult(
        "calibration_ipw", est, pdiff, pop,
        extra={"weight": w_full, "base": base, "shrinkage": shrinkage},
    )
    return CalibrationResult(
        method_result=result,
        anchor_outcomes=anchors,
        ess=effective_sample_size(w_sel),
        achieved_prevalence=est[list(anchors)],
        diagnostics=diag,
        support=support,
        pre_trim_residual=pre_trim_residual,
        post_trim_residual=post_trim_residual,
    )

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

The solution is exponential tilting, ``w_i ∝ base_i * exp(lambda . (Y_i - Pr))``,
with ``lambda`` from a small convex dual (entropy balancing; Hainmueller 2012,
Deville & Sarndal 1992). Because that tilt is log-linear in ``Y`` — the same form
as the selection mechanism — it recovers the disease-driven selection the
covariate model cannot. Using covariate-model weights as the ``base`` keeps the
part of selection that *is* covariate-driven, giving a doubly-robust flavour.

References
----------
- Deville & Särndal (1992), *JASA* 87, 376–382 — calibration estimators.
- Hainmueller (2012), *Political Analysis* 20, 25–46 — entropy balancing.
- Kott & Chang (2010), *JASA* 105, 1265–1275 — calibration for nonignorable
  nonresponse (the ``base_weights`` + known-prevalence construction).
- Horvitz & Thompson (1952), *JASA* 47, 663–685 — the underlying IPW estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .dgm import Dataset
from .methods import MethodResult, _test_arrays, _trim_weights, lasso_propensity
from .metrics import percent_difference, weighted_prevalence


def entropy_balance(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    base_weights: np.ndarray | None = None,
    ridge: float = 0.0,
    max_iter: int = 500,
) -> np.ndarray:
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

    Returns
    -------
    numpy.ndarray
        Weights of length ``n`` summing to 1.
    """
    F = np.atleast_2d(np.asarray(features, dtype=float))
    if F.shape[0] == 1 and F.shape[1] != len(np.atleast_1d(targets)):
        F = F.T
    t = np.atleast_1d(np.asarray(targets, dtype=float))
    n, k = F.shape
    if k != t.shape[0]:
        raise ValueError("features must have one column per target.")

    d = np.ones(n) if base_weights is None else np.asarray(base_weights, dtype=float)
    if np.any(d < 0):
        raise ValueError("base_weights must be non-negative.")
    d = d / d.sum()
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
    return w / w.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` (ignoring zero weights)."""
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    denom = np.sum(w**2)
    return float(np.sum(w) ** 2 / denom) if denom > 0 else 0.0


def outcome_calibration_weights(
    Y: np.ndarray,
    prevalences,
    *,
    joint_prevalences=None,
    base_weights: np.ndarray | None = None,
    shrinkage: float = 0.0,
) -> np.ndarray:
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

    Returns
    -------
    numpy.ndarray
        Calibration weights for the sampled units, summing to 1.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if Y.shape[0] == 1 and Y.shape[1] != len(list(prevalences)):
        Y = Y.T
    cols = [Y]
    targets = list(prevalences)
    if joint_prevalences:
        for (q, qp), value in joint_prevalences.items():
            cols.append((Y[:, q] * Y[:, qp])[:, None])
            targets.append(value)
    features = np.hstack(cols)
    return entropy_balance(
        features, np.asarray(targets, dtype=float), base_weights=base_weights, ridge=shrinkage
    )


@dataclass
class CalibrationResult:
    """Result of :func:`calibration_ipw`, extending :class:`MethodResult` semantics."""

    method_result: MethodResult
    anchor_outcomes: tuple[int, ...]
    ess: float
    achieved_prevalence: np.ndarray  # weighted prevalence of the anchored outcomes

    def __getattr__(self, item):  # delegate weighted_prevalence/percent_diff/summary/...
        return getattr(self.method_result, item)


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
    """Prevalence-calibrated IPW: reweight the sample to the known population prevalences.

    Fits (optionally) a covariate participation model on the training fold to get
    base weights, then calibrates the sampled test units so their weighted
    prevalence matches the known population prevalence of each anchored outcome.
    The estimator is deployable — it uses the sampled units only.

    Parameters
    ----------
    anchor_outcomes:
        Indices of outcomes whose population prevalence is known and used as
        calibration targets. Defaults to all outcomes. Outcomes *not* listed are
        left free, so evaluating them measures how well calibrating on the known
        diseases transfers to an unknown one.
    base:
        ``"lasso"`` uses covariate-model IPW weights as the base (doubly-robust
        flavour); ``"uniform"`` starts from equal weights (pure calibration).
    base_scheme:
        ``"inverse"`` (``1/P``, Horvitz-Thompson) or ``"odds"`` (``(1-P)/P``) for
        the ``"lasso"`` base weights.
    shrinkage:
        Ridge on the tilt; ``0`` calibrates exactly, larger values shrink toward
        the base weights (bias-variance trade-off / stabilization).
    """
    X_train, _, s_train = dataset.split("train")
    X_test, Y_test, s_test = _test_arrays(dataset)
    pop = dataset.population_prevalence
    q = len(pop)
    anchors = tuple(range(q)) if anchor_outcomes is None else tuple(anchor_outcomes)

    sel = s_test == 1
    Y_sel = Y_test[sel]

    if base == "uniform":
        base_w = np.ones(Y_sel.shape[0])
    elif base == "lasso":
        P_sel = lasso_propensity(
            X_train, s_train, X_test[sel], interactions=interactions, cv=cv
        )
        base_w = 1.0 / P_sel if base_scheme == "inverse" else (1.0 - P_sel) / P_sel
    else:
        raise ValueError("base must be 'lasso' or 'uniform'.")

    w_sel = entropy_balance(
        Y_sel[:, anchors], pop[list(anchors)], base_weights=base_w, ridge=shrinkage
    )
    w_sel = _trim_weights(w_sel, trim)

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
    )

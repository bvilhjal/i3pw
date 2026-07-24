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
prevalence within strata. A known marginal prevalence fixes the *number* of cases,
not their *type*: if the sampled cases are systematically milder/more severe than the
population's, matching the margin leaves that within-case selection uncorrected —
calibrate within severity/comorbidity strata for that.

The solution is exponential tilting, ``w_i ∝ base_i * exp(lambda . (Y_i - Pr))``,
with ``lambda`` from a small convex dual (entropy balancing; Hainmueller 2012,
Deville & Sarndal 1992).

What this identifies (be precise). The result is a **density-ratio** model, not a
recovered per-unit inclusion probability. Writing the population-to-sample density
ratio as ``log dP_pop/dP_sample = a(X) + theta . g(Y)``, calibration returns the
minimum-divergence weights ``base(X) * exp(lambda . g(Y))`` that match the supplied
moments ``g(Y)``. These *coincide* with the true inverse-probability weights
``1/pi(X, Y)`` only when that density ratio genuinely lies in the tilt family — the
base weights span the covariate-driven part and ``g(Y)`` the outcome-driven part —
and positivity holds; otherwise they are simply the closest reweighting (in KL) to
the base that reproduces the known moments. Using covariate-model weights as the
``base`` keeps the part of selection that *is* covariate-driven. (Under logistic
participation the *inverse-odds* weight ``(1-pi)/pi`` is exactly log-linear and so
composes exactly with the tilt, whereas ``1/pi`` does so only as inclusion becomes
rare — hence ``calibration_ipw``'s ``base_scheme`` choice.)

This is therefore **not** doubly robust in the AIPW sense (where either the outcome
model or the weights alone suffice). It is better stated as: the weights are
consistent if the base weights capture the ignorable, covariate-driven part of
selection *and* the supplied calibration functions ``g(Y)`` span the remaining
outcome-driven part. The two ingredients cover different pieces of the selection
mechanism; neither alone is enough.

References
----------
- Deville & Särndal (1992), *JASA* 87, 376–382 — calibration estimators.
- Hainmueller (2012), *Political Analysis* 20, 25–46 — entropy balancing.
- Kott & Chang (2010), *JASA* 105, 1265–1275 — calibration for nonignorable
  nonresponse (the ``base_weights`` + known-prevalence construction).
- Horvitz & Thompson (1952), *JASA* 47, 663–685 — the underlying IPW estimator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .dgm import Dataset
from .methods import MethodResult, _test_arrays, _trim_weights, lasso_propensity
from .metrics import percent_difference, weighted_prevalence


class CalibrationWarning(UserWarning):
    """Warned when calibration weights are unreliable (non-convergence, infeasible target)."""


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
                 max_abs_residual: float, message: str = "") -> CalibrationDiagnostics:
    ess, wmax, wmin, top = _weight_concentration(weights)
    return CalibrationDiagnostics(
        converged=converged, n_iter=n_iter, max_abs_residual=max_abs_residual,
        ess=ess, max_weight=wmax, min_weight=wmin, top1pct_weight_mass=top, message=message,
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
    which composes exactly with the exponential-tilt calibration; see the README's
    "What is identified?"). Shared by :func:`calibration_ipw` and the bootstrap so the
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
    features: np.ndarray,
    targets: np.ndarray,
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

    d = np.ones(n) if base_weights is None else np.asarray(base_weights, dtype=float)
    if np.any(d < 0):
        raise ValueError("base_weights must be non-negative.")
    if d.sum() == 0:
        raise ValueError("base_weights sum to zero; cannot form calibration weights.")
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
    w = w / w.sum()

    max_abs_residual = float(np.max(np.abs(w @ F - t))) if k else 0.0
    message = "" if res.success else str(res.message)
    if warn:
        if not res.success:
            warnings.warn(
                f"entropy_balance: optimizer did not converge ({message!r}); "
                "weights may be unreliable.",
                CalibrationWarning, stacklevel=2,
            )
        elif ridge == 0.0 and max_abs_residual > tol:
            warnings.warn(
                f"entropy_balance: calibration targets not met (max residual "
                f"{max_abs_residual:.2e} > tol {tol:.1e}); the target likely lies outside "
                "the sample's convex hull (e.g. an anchored outcome with no cases sampled).",
                CalibrationWarning, stacklevel=2,
            )

    if return_diagnostics:
        return w, _diagnostics(w, res.success, int(res.nit), max_abs_residual, message)
    return w


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
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    prev = np.asarray(list(prevalences), dtype=float)
    if Y.shape[0] == 1 and Y.shape[1] != prev.shape[0]:
        Y = Y.T
    if Y.shape[1] != prev.shape[0]:
        raise ValueError("prevalences must have one entry per outcome column of Y.")

    if warn:
        counts = Y.sum(axis=0)
        unreachable = [
            q for q in range(Y.shape[1])
            if (counts[q] == 0 and prev[q] > 0) or (counts[q] == Y.shape[0] and prev[q] < 1)
        ]
        if unreachable:
            warnings.warn(
                f"outcome_calibration_weights: outcome(s) {unreachable} have no "
                "cases (or no controls) in the sample; their marginal target cannot be met.",
                CalibrationWarning, stacklevel=2,
            )

    cols = [Y]
    targets = list(prev)
    if joint_prevalences:
        for (q, qp), value in joint_prevalences.items():
            pattern = Y[:, q] * Y[:, qp]
            if warn and value > 0 and pattern.sum() == 0:
                warnings.warn(
                    f"outcome_calibration_weights: co-occurrence {(q, qp)} is never "
                    "observed in the sample; its joint target cannot be met.",
                    CalibrationWarning, stacklevel=2,
                )
            cols.append(pattern[:, None])
            targets.append(value)
    features = np.hstack(cols)
    return entropy_balance(
        features, np.asarray(targets, dtype=float), base_weights=base_weights,
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
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    labels = np.asarray(strata).ravel()
    within = np.atleast_2d(np.asarray(within_stratum_prevalence, dtype=float))
    share = np.asarray(stratum_share, dtype=float).ravel()
    n, Q = Y.shape
    A = share.shape[0]
    if labels.shape[0] != n:
        raise ValueError("strata must have one label per row of Y.")
    if within.shape != (A, Q):
        raise ValueError("within_stratum_prevalence must have shape (A, Q).")
    if labels.min() < 0 or labels.max() >= A:
        raise ValueError("strata labels must lie in 0..A-1 (A = len(stratum_share)).")
    if np.any(share < 0):
        raise ValueError("stratum_share must be non-negative.")
    if share.sum() == 0:
        raise ValueError("stratum_share sums to zero; cannot normalize stratum shares.")
    share = share / share.sum()

    onehot = (labels[:, None] == np.arange(A)[None, :]).astype(float)  # (n, A)
    counts = onehot.sum(axis=0)
    joint = share[:, None] * within  # (A, Q) population joint P(Y_q = 1, A = a)

    if warn:
        empty = [a for a in range(A) if counts[a] == 0]
        if empty:
            warnings.warn(
                f"stratified_calibration_weights: stratum/strata {empty} have no sampled "
                "units; their shares and within-stratum prevalences cannot be matched.",
                CalibrationWarning, stacklevel=2,
            )
        unreachable = [
            (a, q) for a in range(A) for q in range(Q)
            if joint[a, q] > 0 and float((onehot[:, a] * Y[:, q]).sum()) == 0.0
        ]
        if unreachable:
            warnings.warn(
                f"stratified_calibration_weights: (stratum, outcome) cells {unreachable} "
                "have a positive known prevalence but no sampled case; unreachable.",
                CalibrationWarning, stacklevel=2,
            )

    # Drop the last stratum indicator: its share is implied by the others plus the
    # sum-to-one constraint, so keeping it would make the dual singular.
    cols = [onehot[:, :-1]]
    targets = list(share[:-1])
    for a in range(A):
        for q in range(Q):
            cols.append((onehot[:, a] * Y[:, q])[:, None])
            targets.append(joint[a, q])

    features = np.hstack(cols)
    return entropy_balance(
        features, np.asarray(targets, dtype=float), base_weights=base_weights,
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

    def __getattr__(self, item):  # delegate weighted_prevalence/percent_diff/summary/...
        return getattr(self.method_result, item)

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

    sel = s_test == 1
    Y_sel = Y_test[sel]

    # Positivity / support: each anchored binary outcome needs both classes present
    # in the sample, or its interior prevalence target is unreachable by reweighting.
    support = {}
    for a in anchors:
        n_case = int(np.round(Y_sel[:, a].sum()))
        support[a] = (n_case, int(Y_sel.shape[0] - n_case))
    infeasible = [
        a for a, (n_case, n_ctrl) in support.items()
        if (n_case == 0 and pop[a] > 0) or (n_ctrl == 0 and pop[a] < 1)
    ]
    if infeasible:
        warnings.warn(
            f"calibration_ipw: anchored outcome(s) {infeasible} lack both classes in the "
            "sample; their population prevalence cannot be matched by reweighting.",
            CalibrationWarning, stacklevel=2,
        )

    base_w = base_weights(
        base, base_scheme, X_train, s_train, X_test[sel], interactions=interactions, cv=cv
    )

    w_sel, diag = entropy_balance(
        Y_sel[:, anchors], pop[list(anchors)], base_weights=base_w, ridge=shrinkage,
        return_diagnostics=True,
    )
    anchor_targets = pop[list(anchors)]
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

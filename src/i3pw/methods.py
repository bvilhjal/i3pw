"""Baseline correction methods.

Each estimates the population prevalence of one or more binary outcomes from a
biased sample and reports the percentage difference from the truth:

- :func:`no_correction` – naive prevalence in the observed sample.
- :func:`lasso_propensity` / :func:`lasso_ipw` – the standard covariate-only
  participation model (``cv.glmnet`` analogue). This is the approach that the
  project found wanting for disease outcomes; the prevalence-informed remedy is
  :func:`i3pw.calibration_ipw`.

The covariate participation model follows Schoeler et al. (2023) / van Alten et al.
(2024); the LASSO fit mirrors ``glmnet`` (Friedman, Hastie & Tibshirani 2010).
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ._links import clip_prob
from .dgm import Dataset
from .metrics import percent_difference, weighted_prevalence


@dataclass
class MethodResult:
    """Outcome-level estimates from a correction method."""

    name: str
    weighted_prevalence: np.ndarray  # (Q,)
    percent_diff: np.ndarray  # (Q,)
    population_prevalence: np.ndarray  # (Q,)
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{self.name}:"]
        for q in range(len(self.weighted_prevalence)):
            lines.append(
                f"  Y{q + 1}: est={self.weighted_prevalence[q]:.4f} "
                f"true={self.population_prevalence[q]:.4f} "
                f"(% diff {self.percent_diff[q]:.2f})"
            )
        return "\n".join(lines)


def _test_arrays(dataset: Dataset):
    X_test, Y_test, s_test = dataset.split("test")
    return X_test, Y_test, s_test


def _ipw_weight(P: np.ndarray, s: np.ndarray, weighting: str) -> np.ndarray:
    """Per-unit IPW weights from inclusion probabilities ``P`` and indicator ``s``.

    ``"inverse"`` — the deployable Hájek weight: selected units get ``1/P``,
    unselected units get ``0`` (a weighted mean then uses the *sample only*).
    ``"oracle_odds"`` — a simulation-only diagnostic: selected units get ``(1-P)/P``
    and unselected units get ``1``, so a weighted mean over *all* units reads the
    outcomes of unselected units — only possible when everyone's outcome is known.
    """
    if weighting == "oracle_odds":
        return np.where(s == 1, (1.0 - P) / P, 1.0)
    if weighting == "inverse":
        return np.where(s == 1, 1.0 / P, 0.0)
    raise ValueError("weighting must be 'inverse' (deployable) or 'oracle_odds'.")


def _trim_weights(w: np.ndarray, trim: float | None) -> np.ndarray:
    """Clip weights at their ``trim`` upper quantile (no-op when ``trim`` is None)."""
    if trim is None:
        return w
    if not 0.0 < trim <= 1.0:
        raise ValueError("trim must be in (0, 1].")
    positive = w[w > 0]
    if positive.size == 0:
        return w
    cap = np.quantile(positive, trim)
    return np.minimum(w, cap)


def lasso_propensity(
    X_train: np.ndarray,
    s_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    interactions: bool = False,
    cv: int = 5,
    Cs=None,
    max_iter: int = 1000,
) -> np.ndarray:
    """Fit ``P(selected | X)`` by cross-validated L1 logistic regression.

    This is the *standard* participation model — the covariate-only propensity
    that the project's central finding shows is weak for many disease outcomes
    (see :func:`i3pw.calibration.calibration_ipw` for the prevalence-informed
    alternative). Returns inclusion probabilities on ``X_eval``, clipped to
    ``(1e-6, 1 - 1e-6)``. The scikit-learn analogue of
    ``cv.glmnet(..., family="binomial")``.

    ``Cs`` (the inverse-regularization grid) defaults to ``numpy.logspace(-3, 1, 8)``
    — moderate-to-mild L1, kept away from very large ``C`` where the near-separable
    inclusion problem makes liblinear iterate pathologically.
    """
    if interactions:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_eval = poly.transform(X_eval)

    if Cs is None:
        Cs = np.logspace(-3, 1, 8)

    lr_kwargs = dict(Cs=Cs, cv=cv, scoring="neg_log_loss", max_iter=max_iter)
    params = inspect.signature(LogisticRegressionCV).parameters
    if "penalty" in params:
        # liblinear runs glmnet-style coordinate descent: far faster than saga
        # on these problem sizes. `penalty` is deprecated (but present) in recent
        # scikit-learn; silence just that forward-compat notice.
        lr_kwargs.update(solver="liblinear", penalty="l1")
    else:
        # `penalty` removed in a future scikit-learn: use the elastic-net API.
        lr_kwargs.update(solver="saga", l1_ratios=(1.0,))
    if "use_legacy_attributes" in params:
        lr_kwargs["use_legacy_attributes"] = False

    # Standardize first, as glmnet does internally: it puts the shared L1 penalty
    # on a comparable scale across covariates and speeds convergence.
    model = make_pipeline(StandardScaler(), LogisticRegressionCV(**lr_kwargs))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        model.fit(X_train, s_train)
    return clip_prob(model.predict_proba(X_eval)[:, 1])


def no_correction(dataset: Dataset) -> MethodResult:
    """Naive prevalence: the outcome mean among sampled units in the test fold."""
    _, Y_test, s_test = _test_arrays(dataset)
    mask = s_test == 1
    est = Y_test[mask].mean(axis=0)
    pop = dataset.population_prevalence
    pdiff = np.array([percent_difference(est[q], pop[q]) for q in range(len(pop))])
    return MethodResult("no_correction", est, pdiff, pop)


def lasso_ipw(
    dataset: Dataset,
    *,
    interactions: bool = False,
    cv: int = 5,
    Cs=None,
    max_iter: int = 1000,
    weighting: str = "inverse",
    trim: float | None = None,
) -> MethodResult:
    """LASSO logistic IPW with a single inclusion model shared across outcomes.

    Fits ``P(selected | X)`` by L1-penalized logistic regression with the
    penalty strength chosen by cross-validation (the scikit-learn analogue of
    ``cv.glmnet(..., family="binomial")``). One weight per unit is applied to
    every outcome. With ``interactions=True`` all pairwise products of the
    covariates are added, mirroring the ``(X)^2`` design in the R code.

    ``Cs`` is the inverse-regularization grid; it defaults to
    ``numpy.logspace(-3, 1, 8)`` (moderate-to-mild L1). This is deliberately
    kept away from very large ``C`` (near-zero regularization): the inclusion
    problem is near-separable, so an under-regularized fit both defeats the
    purpose of the LASSO and makes liblinear iterate pathologically.
    """
    X_train, _, s_train = dataset.split("train")
    X_test, Y_test, s_test = _test_arrays(dataset)

    P_test = lasso_propensity(
        X_train, s_train, X_test, interactions=interactions, cv=cv, Cs=Cs, max_iter=max_iter
    )

    weight = _ipw_weight(P_test, s_test, weighting)
    weight = _trim_weights(weight, trim)
    pop = dataset.population_prevalence
    est = np.array([weighted_prevalence(weight, Y_test[:, q]) for q in range(len(pop))])
    pdiff = np.array([percent_difference(est[q], pop[q]) for q in range(len(pop))])
    return MethodResult("lasso_ipw", est, pdiff, pop, extra={"weight": weight})

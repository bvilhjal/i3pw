"""Doubly-robust (AIPW) estimation of downstream population means.

Prevalence calibration (:mod:`i3pw.calibration`) fixes the *ascertained outcome*
itself, which is not point-identified by covariate weighting. But most analyses
target a *downstream* quantity — the population mean of a trait, biomarker, or
polygenic score measured only on participants. When that quantity is missing at
random given the covariates (``S ⊥ V | X``), it can be recovered, and the
efficient, robust way to do so is the augmented IPW (AIPW) estimator.

For a variable ``V`` observed only on sampled units, with covariates ``X`` known
for the whole population and weights ``w`` (from a participation model or from
:func:`i3pw.calibration_ipw`):

    mu_AIPW = mean_i m(X_i)  +  sum_{i in sample} w_i (V_i - m(X_i))

where ``m(X) = E[V | X]`` is an outcome regression fit on the sample and the
weights are self-normalized (Hájek). This is **doubly robust**: consistent if
*either* the outcome model ``m`` *or* the weights ``w`` are correct
(Robins–Rotnitzky–Zhao 1994). The outcome model also cuts variance relative to
weighting alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge


def _predict(model, X: np.ndarray) -> np.ndarray:
    """Predict E[V|X] (or P(V=1|X)) from a fitted sklearn regressor/classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


@dataclass
class AIPWResult:
    estimate: float
    ipw_only: float       # Hájek weighted mean of V over the sample (no augmentation)
    outcome_only: float   # plug-in mean of m(X) over the population
    truth: float | None = None

    @property
    def error(self) -> float | None:
        return None if self.truth is None else abs(self.estimate - self.truth)


def aipw_mean(
    X_all: np.ndarray,
    sample_mask: np.ndarray,
    V_sample: np.ndarray,
    weights: np.ndarray,
    *,
    outcome_model=None,
    truth: float | None = None,
) -> AIPWResult:
    """Doubly-robust estimate of the population mean of ``V``.

    Parameters
    ----------
    X_all:
        ``(N, p)`` covariates for the whole population (known for everyone).
    sample_mask:
        Length-``N`` boolean; ``True`` where ``V`` is observed (the sample).
    V_sample:
        The observed values of ``V`` on the sampled units (length ``sample_mask.sum()``).
    weights:
        Non-negative weights for the sampled units (length ``sample_mask.sum()``);
        e.g. inverse-probability or calibration weights. Normalized internally.
    outcome_model:
        An unfitted sklearn-style estimator for ``E[V|X]``; cloned and fit on the
        sample. Defaults to ridge regression. Pass a classifier (with
        ``predict_proba``) for a binary ``V``.
    truth:
        Optional known population mean, stored for convenience.
    """
    X_all = np.asarray(X_all, dtype=float)
    mask = np.asarray(sample_mask, dtype=bool)
    V = np.asarray(V_sample, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if not (mask.sum() == V.shape[0] == w.shape[0]):
        raise ValueError("V_sample and weights must have length sample_mask.sum().")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    w = w / w.sum()

    Xs = X_all[mask]
    model = clone(Ridge(alpha=1.0) if outcome_model is None else outcome_model)
    model.fit(Xs, V)
    m_all = _predict(model, X_all)
    m_s = _predict(model, Xs)

    outcome_only = float(m_all.mean())
    ipw_only = float(np.sum(w * V))
    estimate = outcome_only + float(np.sum(w * (V - m_s)))
    return AIPWResult(estimate=estimate, ipw_only=ipw_only, outcome_only=outcome_only, truth=truth)

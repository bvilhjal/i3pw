"""Doubly-robust (AIPW) estimation of downstream population means.

Calibration (:mod:`i3pw.calibration`) fixes the *ascertained outcome* itself. But
the target is usually something else — the population mean of a trait or biomarker
measured only on participants. If that trait is missing at random given the
covariates (``S ⊥ V | X``), it is recoverable, and augmented IPW (AIPW) is the
robust way to recover it: it combines a prediction for everyone with a weighted
correction from the sample.

For a trait ``V`` seen only on sampled units, with covariates ``X`` known for the
whole population and weights ``w`` (from a participation model or from
:func:`i3pw.calibration_ipw`):

    mu_AIPW = mean_i m(X_i)  +  sum_{i in sample} w_i (V_i - m(X_i))
              \\_____________/   \\_____________________________________/
              predict for all     reweighted residual: fixes the prediction
                                   where the sample and model disagree

with ``m(X) = E[V | X]`` an outcome regression fit on the sample and the weights
self-normalized (Hájek 1971). It is **doubly robust** — consistent if *either* ``m``
or ``w`` is correct (Robins–Rotnitzky–Zhao 1994) — so you get two chances to be
right, and the prediction term also lowers variance versus weighting alone.

References
----------
- Robins, Rotnitzky & Zhao (1994), *JASA* 89, 846–866 — AIPW / doubly-robust estimation.
- Bang & Robins (2005), *Biometrics* 61, 962–973 — doubly-robust estimation for
  missing data.
- Chen, Li & Wu (2020), *JASA* 115, 2011–2021 — doubly-robust inference for
  nonprobability (e.g. volunteer) samples.
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

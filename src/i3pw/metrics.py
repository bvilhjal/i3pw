"""Evaluation metrics for weighted prevalence estimation.

These mirror the quantities the original R scripts report for every method:
a weighted prevalence estimate, its percentage difference from the known
population prevalence, and a (weighted) mean squared error of the fitted
inclusion probabilities.
"""

from __future__ import annotations

import numpy as np


def weighted_prevalence(weights: np.ndarray, y: np.ndarray) -> float:
    r"""Weighted mean of a binary outcome, ``sum(w * y) / sum(w)``.

    This is the Horvitz–Thompson style prevalence estimate used throughout the
    project: reweighting a biased sample so that it represents the target
    population.
    """
    weights = np.asarray(weights, dtype=float)
    y = np.asarray(y, dtype=float)
    total = weights.sum()
    if total == 0:
        raise ValueError("Weights sum to zero; cannot compute weighted prevalence.")
    return float((weights * y).sum() / total)


def percent_difference(estimate: float, truth: float) -> float:
    """Absolute percentage difference of ``estimate`` from ``truth``.

    Defined as ``100 * |estimate - truth| / |truth|`` (the ``perc_diff`` metric
    in the R code). Returns ``nan`` when ``truth`` is zero.
    """
    if truth == 0:
        return float("nan")
    return float(abs(estimate - truth) / abs(truth) * 100.0)


def weighted_mse(weights: np.ndarray, p_pred: np.ndarray, y: np.ndarray) -> float:
    """Weighted mean squared error ``sum(w (p - y)^2) / sum(w)``.

    With ``weights`` all equal to one this reduces to the ordinary MSE between
    predicted probabilities and observed outcomes.
    """
    weights = np.asarray(weights, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    total = np.nansum(weights)
    if total == 0:
        raise ValueError("Weights sum to zero; cannot compute weighted MSE.")
    return float(np.nansum(weights * (p_pred - y) ** 2) / total)

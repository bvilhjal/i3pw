"""Evaluation metrics for weighted prevalence estimation.

A weighted (Hájek) prevalence estimate and its percentage difference from the
known population prevalence.
"""

from __future__ import annotations

import numpy as np


def weighted_prevalence(weights: np.ndarray, y: np.ndarray) -> float:
    r"""Weighted mean of a binary outcome, ``sum(w * y) / sum(w)``.

    This is the **Hájek** (self-normalized ratio) prevalence estimate used
    throughout the project: reweighting a biased sample so that it represents the
    target population. It is *not* the Horvitz–Thompson estimator
    ``(1/N) sum_i y_i / P_i`` — dividing by ``sum(w)`` rather than ``N`` makes it
    self-normalizing, which is the standard, lower-variance choice when the
    population size (or the exact sampling fractions) is not pinned down.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if weights.shape != y.shape:
        raise ValueError(
            f"weights and y must have the same length; got "
            f"{weights.shape} and {y.shape}")
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

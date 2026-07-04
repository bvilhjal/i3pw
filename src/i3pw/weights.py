"""Combining per-outcome inverse-probability weights into a single weight.

When the selection model is fit separately for each of the ``Q`` outcomes, each
unit receives ``Q`` candidate weights. The R code experiments with four ways of
collapsing them into one weight per unit; all four are reproduced here.
"""

from __future__ import annotations

import numpy as np

COMBINE_METHODS = ("mean", "product", "harmonic", "absdiff")


def combine_weights(
    per_outcome_weights: np.ndarray,
    method: str = "mean",
    *,
    pop_prevalence: np.ndarray | None = None,
    sample_prevalence: np.ndarray | None = None,
) -> np.ndarray:
    """Collapse an ``(n, Q)`` array of per-outcome weights to shape ``(n,)``.

    Parameters
    ----------
    per_outcome_weights:
        Array of shape ``(n_units, n_outcomes)``; column ``q`` holds the IPW
        weight implied by the ``q``-th outcome's inclusion model.
    method:
        One of:

        - ``"mean"`` – arithmetic mean across outcomes (``rowMeans``).
        - ``"product"`` – product across outcomes.
        - ``"harmonic"`` – harmonic mean ``Q / sum(1 / w_q)``. For two outcomes
          this equals ``2 w1 w2 / (w1 + w2)``, matching the R ``combined_weight_3``.
        - ``"absdiff"`` – average weighted by each outcome's inverse absolute
          gap between population and sample prevalence,
          ``a_q = 1 / |pop_q - sample_q|``. Outcomes whose sample prevalence is
          already far from the population get *less* influence. Requires
          ``pop_prevalence`` and ``sample_prevalence``.

    Returns
    -------
    numpy.ndarray
        Combined weight for each unit, shape ``(n_units,)``.
    """
    w = np.asarray(per_outcome_weights, dtype=float)
    if w.ndim != 2:
        raise ValueError("per_outcome_weights must be 2-D with shape (n, Q).")

    if method == "mean":
        return w.mean(axis=1)
    if method == "product":
        return w.prod(axis=1)
    if method == "harmonic":
        q = w.shape[1]
        with np.errstate(divide="ignore"):
            inv = np.where(w != 0, 1.0 / w, np.inf)
        denom = inv.sum(axis=1)
        return np.where(denom != 0, q / denom, 0.0)
    if method == "absdiff":
        if pop_prevalence is None or sample_prevalence is None:
            raise ValueError(
                "method='absdiff' requires pop_prevalence and sample_prevalence."
            )
        pop = np.asarray(pop_prevalence, dtype=float)
        smp = np.asarray(sample_prevalence, dtype=float)
        gap = np.abs(pop - smp)
        # Guard against a zero gap producing an infinite affinity.
        affinity = 1.0 / np.where(gap == 0, np.finfo(float).eps, gap)
        return (w * affinity).sum(axis=1) / affinity.sum()

    raise ValueError(f"Unknown combine method {method!r}; choose from {COMBINE_METHODS}.")

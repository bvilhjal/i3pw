"""i3pw — Informed Inference of Inverse Probability Weights.

Correcting outcome-dependent selection (ascertainment) bias by inverse-probability
weighting, when the population prevalences of the outcomes are known a priori.

The motivating problem: the standard approach predicts participation probabilities
``P(selected | X)`` from covariates (e.g. socioeconomic features via LASSO) — but
that works poorly for many *disease* outcomes, because participation depends on
having the disease, a signal the covariates barely capture. i3pw instead
**leverages the known population prevalences** to inform the weights.

Two prevalence-informed estimators are provided:

- :func:`calibration_ipw` — the principled version: calibrate the weights so the
  reweighted sample reproduces the known prevalences exactly (entropy balancing),
  optionally on top of a covariate participation model.
- :func:`penalized_ipw` — the softer precursor from the original R project: a
  logistic inclusion model with a quadratic prevalence penalty, numba-JIT compiled.

Baselines for comparison: :func:`no_correction` and :func:`lasso_ipw` (the covariate
propensity model that motivated the whole exercise).
"""

from __future__ import annotations

from ._links import logit, sigmoid
from .aipw import AIPWResult, aipw_mean
from .calibration import (
    CalibrationResult,
    calibration_ipw,
    effective_sample_size,
    entropy_balance,
)
from .dgm import Dataset, SimConfig, make_dataset, nearest_pd_correlation, random_correlation
from .evaluation import MonteCarloSummary, format_summary, monte_carlo
from .methods import (
    MethodResult,
    cross_validate,
    lasso_ipw,
    lasso_propensity,
    no_correction,
    penalized_ipw,
)
from .metrics import percent_difference, weighted_mse, weighted_prevalence
from .penalized import PenalizedIPW, warmup
from .weights import combine_weights

__version__ = "0.1.0"

__all__ = [
    "SimConfig",
    "Dataset",
    "make_dataset",
    "random_correlation",
    "nearest_pd_correlation",
    "PenalizedIPW",
    "warmup",
    "no_correction",
    "lasso_ipw",
    "lasso_propensity",
    "penalized_ipw",
    "calibration_ipw",
    "CalibrationResult",
    "entropy_balance",
    "effective_sample_size",
    "aipw_mean",
    "AIPWResult",
    "cross_validate",
    "MethodResult",
    "monte_carlo",
    "MonteCarloSummary",
    "format_summary",
    "combine_weights",
    "weighted_prevalence",
    "percent_difference",
    "weighted_mse",
    "sigmoid",
    "logit",
    "__version__",
]

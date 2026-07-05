"""i3pw — Informed Inference of Inverse Probability Weights.

Correcting outcome-dependent selection (ascertainment) bias by inverse-probability
weighting, when the population prevalences of the outcomes are known a priori.

The motivating problem: the standard approach predicts participation probabilities
``P(selected | X)`` from covariates (e.g. socioeconomic features via LASSO) — but
that works poorly for many *disease* outcomes, because participation depends on
having the disease, a signal the covariates barely capture. i3pw instead
**leverages the known population prevalences** to inform the weights.

The estimator: :func:`calibration_ipw` — calibrate the weights so the reweighted
sample reproduces the known prevalences exactly (entropy balancing), optionally on
top of a covariate participation model. Baselines for comparison:
:func:`no_correction` and :func:`lasso_ipw` (the covariate propensity model that
motivated the whole exercise).
"""

from __future__ import annotations

from ._links import logit, sigmoid
from .aipw import AIPWResult, aipw_mean
from .calibration import (
    CalibrationResult,
    calibration_ipw,
    effective_sample_size,
    entropy_balance,
    outcome_calibration_weights,
)
from .dgm import Dataset, SimConfig, make_dataset, nearest_pd_correlation, random_correlation
from .evaluation import MonteCarloSummary, format_summary, monte_carlo
from .liability import (
    AscertainedSample,
    SelectionPopulation,
    estimate_liability_r2,
    lee_transform,
    liability_r2_from_weights,
    liability_threshold,
    moment_slope,
    observed_to_liability,
    similarity_matrix,
    simulate_case_control,
    simulate_liability_selection,
)
from .methods import (
    MethodResult,
    lasso_ipw,
    lasso_propensity,
    no_correction,
)
from .metrics import percent_difference, weighted_prevalence

__version__ = "0.1.0"

__all__ = [
    "SimConfig",
    "Dataset",
    "make_dataset",
    "random_correlation",
    "nearest_pd_correlation",
    "no_correction",
    "lasso_ipw",
    "lasso_propensity",
    "calibration_ipw",
    "CalibrationResult",
    "entropy_balance",
    "outcome_calibration_weights",
    "effective_sample_size",
    "aipw_mean",
    "AIPWResult",
    "liability_threshold",
    "observed_to_liability",
    "lee_transform",
    "moment_slope",
    "similarity_matrix",
    "simulate_case_control",
    "estimate_liability_r2",
    "liability_r2_from_weights",
    "simulate_liability_selection",
    "SelectionPopulation",
    "AscertainedSample",
    "MethodResult",
    "monte_carlo",
    "MonteCarloSummary",
    "format_summary",
    "weighted_prevalence",
    "percent_difference",
    "sigmoid",
    "logit",
    "__version__",
]

"""i3pw — Informed Inference of Inverse Probability Weights.

Correcting outcome-dependent selection (ascertainment) bias by reweighting, when the
population prevalences of the outcomes are known a priori.

The motivating problem: the standard approach predicts participation probabilities
``P(selected | X)`` from covariates (e.g. socioeconomic features via LASSO) — but
that works poorly for many *disease* outcomes, because participation depends on
having the disease, a signal the covariates barely capture. i3pw instead
**leverages the known population prevalences** to inform the weights.

What it computes, stated exactly: not a recovered per-unit inclusion probability, but
the **minimum-divergence weights that reproduce the known population moments** — which
equal the true inverse-probability weights when the population-to-sample density ratio
is spanned by the base weights plus those moments, and are the closest reweighting to
the base otherwise. ``docs/theory.md#what-is-identified`` is the canonical statement.

Two front doors, and they are not interchangeable:

- :func:`calibrate` — **for a real cohort.** Plain arrays: outcomes on the participants,
  register prevalences, optional base weights from your own participation model, and a
  ``holdout=`` of register margins to check against.
- :func:`calibration_ipw` — **for the simulations.** Needs a :class:`Dataset`, which
  carries ground truth. Baselines to compare against it: :func:`no_correction` and
  :func:`lasso_ipw` (the covariate propensity model that motivated the exercise).
"""

from __future__ import annotations

from ._links import logit, sigmoid
from .aipw import AIPWResult, aipw_mean
from .balance import BalanceReport, balance_report
from .calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    CalibrationWarning,
    apply_tilt,
    calibration_ipw,
    calibration_mean_se,
    compute_base_weights,
    effective_sample_size,
    entropy_balance,
    outcome_calibration_weights,
    stratified_calibration_weights,
)
from .dgm import Dataset, SimConfig, make_dataset, nearest_pd_correlation, random_correlation
from .evaluation import MonteCarloSummary, format_summary, monte_carlo
from .fit import CalibrationFit, calibrate, inverse_probability_weights
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
from .uncertainty import (
    BootstrapResult,
    Estimate,
    SensitivityResult,
    bootstrap_calibration_ipw,
    prevalence_sensitivity,
    weighted_mean_se,
)

__version__ = "0.3.1"

__all__ = [
    "SimConfig",
    "Dataset",
    "make_dataset",
    "random_correlation",
    "nearest_pd_correlation",
    "calibrate",
    "CalibrationFit",
    "inverse_probability_weights",
    "no_correction",
    "lasso_ipw",
    "lasso_propensity",
    "calibration_ipw",
    "CalibrationResult",
    "CalibrationDiagnostics",
    "CalibrationWarning",
    "entropy_balance",
    "apply_tilt",
    "calibration_mean_se",
    "balance_report",
    "BalanceReport",
    "outcome_calibration_weights",
    "stratified_calibration_weights",
    "compute_base_weights",
    "effective_sample_size",
    "weighted_mean_se",
    "Estimate",
    "bootstrap_calibration_ipw",
    "BootstrapResult",
    "prevalence_sensitivity",
    "SensitivityResult",
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

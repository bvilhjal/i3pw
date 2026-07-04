"""i3pw — Informed Inference of Inverse Probability Weights.

A Python reimplementation of the key methods from the ``SelectionBias`` R
project: simulating outcome-dependent selection bias and correcting it with
inverse-probability weighting, including a novel *prevalence-penalized* IPW
estimator that uses known population prevalences to inform the weights.

Compute-heavy kernels (penalized objective, gradient, and gradient descent)
are JIT-compiled with numba.
"""

from __future__ import annotations

from ._links import logit, sigmoid
from .dgm import Dataset, SimConfig, make_dataset, nearest_pd_correlation, random_correlation
from .methods import (
    MethodResult,
    cross_validate,
    lasso_ipw,
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
    "penalized_ipw",
    "cross_validate",
    "MethodResult",
    "combine_weights",
    "weighted_prevalence",
    "percent_difference",
    "weighted_mse",
    "sigmoid",
    "logit",
    "__version__",
]

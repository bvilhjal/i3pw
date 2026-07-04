"""Prevalence-penalized inverse-probability-weighting estimator.

This is the core method of the project (``i3pw`` — *informed inference of
inverse probability weights*). For each outcome it fits a logistic model of
sample inclusion, regularized by a LASSO (L1) penalty and, crucially, an
*informed* prevalence penalty that pulls the model's average predicted
inclusion probability toward the outcome's known population prevalence.

Fitted inclusion probabilities become inverse-probability weights that reweight
a biased sample back toward the population.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from . import _kernels as K
from ._links import clip_prob, logit, sigmoid

OPTIMIZERS = ("gd", "bfgs", "lbfgs")


def warmup() -> None:
    """Trigger numba JIT compilation of the kernels on a tiny problem.

    numba compiles each kernel the first time it runs (a one-time cost, then
    cached to disk). Calling ``warmup()`` up front pays that cost explicitly —
    useful before timing anything or to avoid a puzzling pause on the first fit.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3))
    s = (rng.uniform(size=64) < 0.4).astype(float)
    for opt in OPTIMIZERS:
        PenalizedIPW(lam=0.01, gamma=0.1, optimizer=opt, max_iter=20).fit(X, s, [0.4])


class PenalizedIPW:
    """Fit outcome-specific inclusion models with an informed prevalence penalty.

    Parameters
    ----------
    lam:
        LASSO (L1) penalty strength.
    gamma:
        Prevalence-penalty strength. ``gamma=0`` recovers ordinary L1-penalized
        logistic IPW with no population information.
    optimizer:
        ``"gd"`` (numba gradient descent with learning-rate decay), ``"bfgs"``
        (SciPy BFGS), or ``"lbfgs"`` (SciPy L-BFGS-B, supports box ``bounds``).
    fit_intercept:
        Prepend an unpenalized intercept column. The original R gradient code
        omitted the intercept; keeping it makes the prevalence penalty able to
        shift the mean prediction directly, so it defaults to ``True``.
    learning_rate, max_iter, decay_interval, tol:
        Gradient-descent controls (``tol`` is also used as the SciPy tolerance).
    bounds:
        Optional ``(low, high)`` box constraint on coefficients for ``"lbfgs"``.
    """

    def __init__(
        self,
        lam: float = 0.0,
        gamma: float = 0.0,
        *,
        optimizer: str = "gd",
        fit_intercept: bool = True,
        learning_rate: float = 1e-3,
        max_iter: int = 5000,
        decay_interval: int = 1000,
        tol: float = 1e-6,
        bounds: tuple[float, float] | None = None,
        eps: float = 1e-6,
    ) -> None:
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}.")
        self.lam = lam
        self.gamma = gamma
        self.optimizer = optimizer
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.decay_interval = decay_interval
        self.tol = tol
        self.bounds = bounds
        self.eps = eps

    # -- internals ---------------------------------------------------------
    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=float)
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        return np.ascontiguousarray(X)

    def _l1_mask(self, d: int) -> np.ndarray:
        mask = np.ones(d)
        if self.fit_intercept:
            mask[0] = 0.0  # never L1-penalize the intercept
        return mask

    def _fit_one_scipy(self, X, s, logit_pi, l1_mask, beta0):
        args = (X, s, self.lam, self.gamma, logit_pi, l1_mask, self.eps)
        if self.optimizer == "bfgs":
            res = minimize(
                K.objective, beta0, args=args, jac=K.gradient,
                method="BFGS", options={"maxiter": self.max_iter, "gtol": self.tol},
            )
        else:  # lbfgs
            box = None
            if self.bounds is not None:
                box = [self.bounds] * beta0.shape[0]
                if self.fit_intercept:
                    box[0] = (None, None)  # leave the intercept free
            res = minimize(
                K.objective, beta0, args=args, jac=K.gradient,
                method="L-BFGS-B", bounds=box,
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )
        return res.x

    # -- public API --------------------------------------------------------
    def fit(self, X: np.ndarray, sample_indicator: np.ndarray, population_prevalence) -> "PenalizedIPW":
        """Fit one inclusion model per outcome.

        ``population_prevalence`` is an array of length ``Q`` (one known
        population prevalence per outcome). The same ``sample_indicator`` is the
        regression target for every outcome; only the prevalence penalty target
        differs across outcomes.
        """
        Xd = self._design(X)
        s = np.ascontiguousarray(sample_indicator, dtype=float)
        pop = np.atleast_1d(np.asarray(population_prevalence, dtype=float))
        logit_pis = logit(pop)
        q, d = pop.shape[0], Xd.shape[1]
        l1_mask = self._l1_mask(d)
        betas0 = np.zeros((q, d))

        if self.optimizer == "gd":
            coef = K.fit_all_gradient_descent(
                Xd, s, betas0,
                np.full(q, self.lam), np.full(q, self.gamma), logit_pis, l1_mask,
                self.learning_rate, self.max_iter, self.decay_interval, self.tol, self.eps,
            )
        else:
            coef = np.empty((q, d))
            for j in range(q):
                coef[j] = self._fit_one_scipy(Xd, s, logit_pis[j], l1_mask, betas0[j])

        self.coef_ = coef  # (Q, d), includes intercept in column 0 when fit_intercept
        self.population_prevalence_ = pop
        self.n_outcomes_ = q
        return self

    def score_objective(self, X: np.ndarray, sample_indicator: np.ndarray) -> float:
        """Mean penalized objective across outcomes (lower is better).

        Used as the cross-validation criterion. Evaluated at the fitted
        coefficients on a held-out fold.
        """
        Xd = self._design(X)
        s = np.ascontiguousarray(sample_indicator, dtype=float)
        logit_pis = logit(self.population_prevalence_)
        l1_mask = self._l1_mask(Xd.shape[1])
        vals = [
            K.objective(self.coef_[j], Xd, s, self.lam, self.gamma, logit_pis[j], l1_mask, self.eps)
            for j in range(self.n_outcomes_)
        ]
        return float(np.mean(vals))

    def predict_inclusion(self, X: np.ndarray) -> np.ndarray:
        """Predicted inclusion probability per unit and outcome, shape ``(n, Q)``."""
        Xd = self._design(X)
        return clip_prob(sigmoid(Xd @ self.coef_.T), self.eps)

    def weights(self, X: np.ndarray, sample_indicator: np.ndarray) -> np.ndarray:
        """Per-outcome IPW weights, shape ``(n, Q)``.

        Selected units (``sample_indicator == 1``) receive the inverse-odds
        weight ``(1 - P) / P``; unselected units receive weight 1, matching the
        construction used throughout the R scripts.
        """
        P = self.predict_inclusion(X)
        s = np.asarray(sample_indicator, dtype=float)[:, None]
        return np.where(s == 1, (1.0 - P) / P, 1.0)

"""Data-generating mechanism for the selection-bias simulations.

This reproduces the simulation used in the R scripts (``generalised_form.R`` and
``differing_dgms.R``):

1. Draw ``n`` correlated covariates ``X`` from a multivariate normal whose
   correlation matrix has low-to-moderate off-diagonal entries.
2. Generate ``Q`` binary outcomes from logistic models, each driven by its own
   block of predictors and calibrated to a target population prevalence.
3. Draw a *biased* sample from a Bernoulli-logistic participation model. Its
   coefficients are calibrated so that the expected sample count and expected
   outcome margins equal their configured targets.
4. Split the population into train / test folds for fitting and evaluation.

The result is a :class:`Dataset` that carries everything downstream methods
need, including the ground-truth coefficients and population prevalences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ._links import sigmoid


def nearest_pd_correlation(a: np.ndarray) -> np.ndarray:
    """Repair a symmetric matrix to a nearby positive-definite correlation matrix.

    The R code relies on ``Matrix::nearPD``. Here we clip the eigenvalues to a
    small positive floor and rescale the result to have a unit diagonal. This
    is not the Frobenius-nearest projection, but it yields a valid
    (positive-definite, unit-diagonal) correlation matrix.
    """
    a = np.asarray(a, dtype=float)
    a = (a + a.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(a)
    eigvals = np.clip(eigvals, 1e-6, None)
    a_pd = (eigvecs * eigvals) @ eigvecs.T
    # Rescale to a correlation matrix (unit diagonal).
    d = np.sqrt(np.diag(a_pd))
    a_pd = a_pd / np.outer(d, d)
    a_pd = (a_pd + a_pd.T) / 2.0
    np.fill_diagonal(a_pd, 1.0)
    return a_pd


def random_correlation(
    n_features: int,
    rng: np.random.Generator,
    low: float = 0.1,
    high: float = 0.5,
) -> np.ndarray:
    """Random correlation matrix with off-diagonals drawn uniformly from ``[low, high]``."""
    upper = np.zeros((n_features, n_features))
    iu = np.triu_indices(n_features, k=1)
    upper[iu] = rng.uniform(low, high, size=iu[0].size)
    corr = upper + upper.T
    np.fill_diagonal(corr, 1.0)
    return nearest_pd_correlation(corr)


_DEFAULT_POP_PREVALENCE = (0.4, 0.2, 0.15, 0.1, 0.05)
_DEFAULT_SAMPLE_PREVALENCE = (0.2, 0.1, 0.05, 0.01, 0.005)


def _default_prevalence(base: tuple[float, ...], q: int) -> tuple[float, ...]:
    """First ``q`` of ``base``, extended by halving the last entry if ``q`` is larger."""
    ext = list(base[:q])
    while len(ext) < q:
        ext.append(ext[-1] / 2.0)
    return tuple(ext)


@dataclass
class SimConfig:
    """Configuration for :func:`make_dataset`.

    The default prevalences follow the five-outcome scenario in ``generalised_form.R``
    and adapt to ``n_outcomes`` when it is overridden (so ``SimConfig(n_outcomes=2)``
    just uses the first two). Pass explicit tuples to control them. Population targets
    calibrate the mean outcome probabilities over the realized covariates; binary
    outcomes then fluctuate around those targets. Likewise, ``sample_size`` and
    ``target_sample_prevalence`` are expectations under independent Bernoulli
    participation, not fixed realized totals.
    """

    population_size: int = 11000
    n_features: int = 50
    n_outcomes: int = 5
    predictors_per_outcome: int = 10
    target_population_prevalence: tuple[float, ...] | None = None
    target_sample_prevalence: tuple[float, ...] | None = None
    sample_size: int = 1000
    """Expected number of participants under the Bernoulli selection model."""
    coef_low: float = -0.5
    coef_high: float = 0.5
    corr_low: float = 0.1
    corr_high: float = 0.5
    test_size: float = 0.25
    seed: int | None = 97
    selection_covariate_strength: float = 0.0
    """How strongly the covariates ``X`` drive participation, on the log-odds scale.

    The default ``0.0`` keeps the original no-covariate-channel setting, in which
    selection depends on the **outcomes only**. That default is worth understanding
    before trusting a benchmark run on it: with no covariate channel there is no direct
    ``a(X)`` for a
    participation model to learn. What little signal remains is second-hand — ``X``
    predicts ``Y`` and ``Y`` drives selection — so ``P(S | X)`` is only weakly
    learnable and carries little a covariate model can act on. Consequently
    :func:`i3pw.lasso_ipw` is close to guaranteed to be no better than no correction,
    and ``calibration_ipw(base="lasso")`` mostly adds estimation noise to a base with
    little to estimate. Comparisons on this setting cannot discriminate between the two
    selection channels the package is *about*.

    Set it positive (``0.5``–``1.5`` is a reasonable range) to add a covariate-driven
    component to selection, so that the covariate model has real signal and the
    contribution of the base weights becomes measurable.
    """
    n_selection_covariates: int = 5
    """How many covariates participate in the selection model when the strength is > 0."""

    def __post_init__(self) -> None:
        q = self.n_outcomes
        if self.target_population_prevalence is None:
            self.target_population_prevalence = _default_prevalence(_DEFAULT_POP_PREVALENCE, q)
        if self.target_sample_prevalence is None:
            self.target_sample_prevalence = _default_prevalence(_DEFAULT_SAMPLE_PREVALENCE, q)
        if len(self.target_population_prevalence) != q:
            raise ValueError("target_population_prevalence must have n_outcomes entries.")
        if len(self.target_sample_prevalence) != q:
            raise ValueError("target_sample_prevalence must have n_outcomes entries.")
        population_target = np.asarray(self.target_population_prevalence, dtype=float)
        sample_target = np.asarray(self.target_sample_prevalence, dtype=float)
        if np.any(~np.isfinite(population_target)) or np.any(
            (population_target <= 0.0) | (population_target >= 1.0)
        ):
            raise ValueError("target_population_prevalence entries must lie in (0, 1).")
        if np.any(~np.isfinite(sample_target)) or np.any(
            (sample_target <= 0.0) | (sample_target >= 1.0)
        ):
            raise ValueError("target_sample_prevalence entries must lie in (0, 1).")
        if not 0 < self.sample_size < self.population_size:
            raise ValueError("sample_size must lie strictly between 0 and population_size.")


@dataclass
class Dataset:
    """Container for a simulated population and its biased sample."""

    X: np.ndarray  # (N, p) covariates for the whole population
    Y: np.ndarray  # (N, Q) binary outcomes
    sample_indicator: np.ndarray  # (N,) Bernoulli participation indicator
    coefficients: np.ndarray  # (Q, p) ground-truth outcome coefficients
    intercepts: np.ndarray  # (Q,) ground-truth outcome intercepts
    population_prevalence: np.ndarray  # (Q,) realised population prevalence
    train_idx: np.ndarray
    test_idx: np.ndarray
    config: SimConfig = field(repr=False)

    @property
    def n_outcomes(self) -> int:
        return self.Y.shape[1]

    @property
    def sample_prevalence(self) -> np.ndarray:
        """Outcome prevalence among the biased sample (the naive, uncorrected estimate)."""
        mask = self.sample_indicator == 1
        return self.Y[mask].mean(axis=0)

    def split(self, which: str):
        """Return ``(X, Y, sample_indicator)`` for the ``"train"`` or ``"test"`` fold."""
        if which not in ("train", "test"):
            raise ValueError(f"which must be 'train' or 'test'; got {which!r}")
        idx = self.train_idx if which == "train" else self.test_idx
        return self.X[idx], self.Y[idx], self.sample_indicator[idx]


def make_dataset(config: SimConfig | None = None, **overrides) -> Dataset:
    """Simulate a population, induce selection bias, and split into train/test.

    Parameters
    ----------
    config:
        A :class:`SimConfig`. If omitted a default one is built and any keyword
        ``overrides`` are applied to it (e.g. ``make_dataset(seed=1, n_outcomes=2)``).
    """
    if config is None:
        config = SimConfig(**overrides)
    elif overrides:
        raise ValueError("Pass either a SimConfig or keyword overrides, not both.")

    rng = np.random.default_rng(config.seed)
    n, p, q = config.population_size, config.n_features, config.n_outcomes

    # 1. Correlated covariates.
    corr = random_correlation(p, rng, config.corr_low, config.corr_high)
    X = rng.multivariate_normal(np.zeros(p), corr, size=n)

    # 2. Outcome coefficients: each outcome owns a contiguous block of predictors.
    coefs = np.zeros((q, p))
    for i in range(q):
        start = (i * config.predictors_per_outcome) % p
        end = min(start + config.predictors_per_outcome, p)
        coefs[i, start:end] = rng.uniform(config.coef_low, config.coef_high, size=end - start)

    linear_predictors = X @ coefs.T
    intercepts = np.array(
        [
            _solve_logistic_intercept(linear_predictors[:, j], target)
            for j, target in enumerate(config.target_population_prevalence)
        ]
    )
    logits = intercepts[None, :] + linear_predictors  # (n, Q)
    probs = sigmoid(logits)
    Y = (rng.uniform(size=probs.shape) < probs).astype(int)

    population_prevalence = Y.mean(axis=0)

    # 3. Biased Bernoulli sampling. The participation-model coefficients are fitted
    #    jointly so its expected count and outcome margins equal the configured targets.
    #    The optional covariate score is a fixed offset in the participation logit.
    # Annotated because the rescale below produces a differently-shaped static type
    # under shape-typed numpy stubs, which a bare inferred binding would reject.
    selection_coef: np.ndarray = np.zeros(p)
    if config.selection_covariate_strength != 0.0:
        m = min(config.n_selection_covariates, p)
        idx = rng.choice(p, size=m, replace=False)
        selection_coef[idx] = rng.normal(0.0, 1.0, size=m)
        norm = np.linalg.norm(selection_coef)
        if norm > 0:  # unit-norm so `strength` alone sets the scale
            selection_coef = selection_coef / norm * config.selection_covariate_strength

    sample_indicator = _induce_selection(
        Y,
        np.asarray(config.target_sample_prevalence),
        config.sample_size,
        rng,
        covariate_score=X @ selection_coef,
    )

    # 4. Train / test split of the whole population.
    perm = rng.permutation(n)
    n_test = int(round(config.test_size * n))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])

    return Dataset(
        X=X,
        Y=Y,
        sample_indicator=sample_indicator,
        coefficients=coefs,
        intercepts=intercepts,
        population_prevalence=population_prevalence,
        train_idx=train_idx,
        test_idx=test_idx,
        config=config,
    )


def _solve_logistic_intercept(
    linear_predictor: np.ndarray,
    target_mean: float,
    *,
    atol: float = 1e-13,
    max_iter: int = 100,
) -> float:
    """Solve ``mean(expit(intercept + linear_predictor)) = target_mean``.

    Bisection is sufficient because the left-hand side is continuous and strictly
    increasing in the intercept. Bounds based on the extreme linear predictors
    bracket the root without relying on arbitrary constants.
    """
    eta = np.asarray(linear_predictor, dtype=float)
    if eta.ndim != 1 or eta.size == 0 or np.any(~np.isfinite(eta)):
        raise ValueError("linear_predictor must be a non-empty finite one-dimensional array.")
    if not np.isfinite(target_mean) or not 0.0 < target_mean < 1.0:
        raise ValueError("target_mean must lie in (0, 1).")

    target_logit = np.log(target_mean) - np.log1p(-target_mean)
    lower = target_logit - eta.max()
    upper = target_logit - eta.min()
    for _ in range(max_iter):
        midpoint = (lower + upper) / 2.0
        error = float(sigmoid(midpoint + eta).mean() - target_mean)
        if abs(error) <= atol:
            return midpoint
        if error < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _selection_probabilities(
    Y: np.ndarray,
    target_sample_prevalence: np.ndarray,
    sample_size: int,
    covariate_score: np.ndarray | None = None,
) -> np.ndarray:
    """Fit joint logistic participation probabilities by moment matching.

    For rows ``y_i`` and fixed covariate offsets ``z_i``, this solves

    ``P(S_i=1 | y_i, X_i) = expit(alpha + y_i @ theta + z_i)``

    so that ``sum_i P(S_i=1)`` equals ``sample_size`` and, for every outcome
    ``j``, ``sum_i P(S_i=1) y_ij / sample_size`` equals the requested sample
    prevalence. The convex dual objective is minimized jointly over ``alpha``
    and ``theta``. Correlated outcomes are therefore handled jointly rather than
    by multiplying incompatible marginal weights.

    A finite solution need not exist for arbitrary margins (for example, two
    identical outcome columns cannot have different requested prevalences). Such
    configurations raise ``ValueError`` rather than silently changing the model.
    """
    outcomes = np.asarray(Y, dtype=float)
    target = np.asarray(target_sample_prevalence, dtype=float)
    if outcomes.ndim != 2 or outcomes.shape[0] == 0:
        raise ValueError("Y must be a non-empty two-dimensional array.")
    if np.any(~np.isfinite(outcomes)) or np.any((outcomes != 0.0) & (outcomes != 1.0)):
        raise ValueError("Y must contain finite binary outcomes.")
    n, q = outcomes.shape
    if target.shape != (q,):
        raise ValueError("target_sample_prevalence must have one entry per outcome.")
    if np.any(~np.isfinite(target)) or np.any((target <= 0.0) | (target >= 1.0)):
        raise ValueError("target_sample_prevalence entries must lie in (0, 1).")
    if not 0 < sample_size < n:
        raise ValueError("sample_size must lie strictly between 0 and len(Y).")

    if covariate_score is None:
        offset = np.zeros(n)
    else:
        offset = np.asarray(covariate_score, dtype=float)
        if offset.shape != (n,) or np.any(~np.isfinite(offset)):
            raise ValueError("covariate_score must be a finite vector with len(Y) entries.")

    features = np.column_stack((np.ones(n), outcomes))
    target_totals = sample_size * np.r_[1.0, target]

    # Necessary marginal capacity checks give a much clearer error than an optimizer
    # failure for the common infeasible cases. Strict inequalities are required
    # because finite logistic probabilities lie strictly between zero and one.
    available_positive = outcomes.sum(axis=0)
    available_negative = n - available_positive
    desired_positive = sample_size * target
    desired_negative = sample_size * (1.0 - target)
    if np.any(desired_positive >= available_positive) or np.any(
        desired_negative >= available_negative
    ):
        raise ValueError(
            "target_sample_prevalence is infeasible for the realized outcomes and sample_size."
        )

    scale = float(n)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        score = offset + features @ beta
        probability = sigmoid(score)
        value = (np.logaddexp(0.0, score).sum() - target_totals @ beta) / scale
        gradient = (features.T @ probability - target_totals) / scale
        return float(value), gradient

    initial = np.zeros(q + 1)
    initial[0] = _solve_logistic_intercept(offset, sample_size / n)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 1000, "maxls": 50},
    )
    probability = sigmoid(offset + features @ result.x)
    residual = features.T @ probability - target_totals
    max_error = float(np.max(np.abs(residual)))
    tolerance = 1e-7 * sample_size
    if (
        np.any(~np.isfinite(probability))
        or np.any((probability <= 0.0) | (probability >= 1.0))
        or not np.isfinite(max_error)
        or max_error > tolerance
    ):
        raise ValueError(
            "No finite Bernoulli-logistic selection model matches the requested joint "
            f"margins (largest expected-count error {max_error:.3g})."
        )
    return probability


def _induce_selection(
    Y: np.ndarray,
    target_sample_prevalence: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    covariate_score: np.ndarray | None = None,
) -> np.ndarray:
    """Draw independent Bernoulli participation indicators.

    ``sample_size`` and ``target_sample_prevalence`` describe expectations under
    the calibrated model; the realized count and margins fluctuate because the
    indicators are Bernoulli draws.
    """
    probability = _selection_probabilities(
        Y,
        target_sample_prevalence,
        sample_size,
        covariate_score=covariate_score,
    )
    return (rng.uniform(size=Y.shape[0]) < probability).astype(int)

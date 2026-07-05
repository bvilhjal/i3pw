"""Data-generating mechanism for the selection-bias simulations.

This reproduces the simulation used in the R scripts (``generalised_form.R`` and
``differing_dgms.R``):

1. Draw ``n`` correlated covariates ``X`` from a multivariate normal whose
   correlation matrix has low-to-moderate off-diagonal entries.
2. Generate ``Q`` binary outcomes from logistic models, each driven by its own
   block of predictors and calibrated to a target population prevalence.
3. Draw a *biased* sample in which each outcome is over- or under-represented
   relative to the population, by sampling without replacement with selection
   probabilities that depend on the outcomes.
4. Split the population into train / test folds for fitting and evaluation.

The result is a :class:`Dataset` that carries everything downstream methods
need, including the ground-truth coefficients and population prevalences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._links import logit, sigmoid


def nearest_pd_correlation(a: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the nearest positive-definite correlation matrix.

    The R code relies on ``Matrix::nearPD``. Here we clip the eigenvalues to a
    small positive floor and rescale the result to have a unit diagonal, which
    yields a valid (positive-definite, unit-diagonal) correlation matrix.
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
    corr = np.zeros((n_features, n_features))
    iu = np.triu_indices(n_features, k=1)
    corr[iu] = rng.uniform(low, high, size=iu[0].size)
    corr = corr + corr.T
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
    just uses the first two). Pass explicit tuples to control them.
    """

    population_size: int = 11000
    n_features: int = 50
    n_outcomes: int = 5
    predictors_per_outcome: int = 10
    target_population_prevalence: tuple[float, ...] | None = None
    target_sample_prevalence: tuple[float, ...] | None = None
    sample_size: int = 1000
    coef_low: float = -0.5
    coef_high: float = 0.5
    corr_low: float = 0.1
    corr_high: float = 0.5
    test_size: float = 0.25
    seed: int | None = 97

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


@dataclass
class Dataset:
    """Container for a simulated population and its biased sample."""

    X: np.ndarray  # (N, p) covariates for the whole population
    Y: np.ndarray  # (N, Q) binary outcomes
    sample_indicator: np.ndarray  # (N,) 1 if drawn into the biased sample
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

    intercepts = logit(np.asarray(config.target_population_prevalence))
    logits = intercepts[None, :] + X @ coefs.T  # (n, Q)
    probs = sigmoid(logits)
    Y = (rng.uniform(size=probs.shape) < probs).astype(int)

    population_prevalence = Y.mean(axis=0)

    # 3. Biased sampling. Per-outcome selection weights push each outcome toward
    #    its target sample prevalence; the overall weight is their product.
    sample_indicator = _induce_selection(
        Y,
        population_prevalence,
        np.asarray(config.target_sample_prevalence),
        config.sample_size,
        rng,
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


def _induce_selection(
    Y: np.ndarray,
    population_prevalence: np.ndarray,
    target_sample_prevalence: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return a 0/1 indicator selecting ``sample_size`` units with outcome-dependent bias."""
    n, q = Y.shape
    weights = np.ones((n, q))
    for i in range(q):
        pos = target_sample_prevalence[i] / (population_prevalence[i] + eps)
        neg = (1.0 - target_sample_prevalence[i]) / (1.0 - population_prevalence[i] + eps)
        weights[:, i] = np.where(Y[:, i] == 1, pos, neg)

    overall = weights.prod(axis=1)
    overall = overall / overall.sum()

    selected = rng.choice(n, size=sample_size, replace=False, p=overall)
    indicator = np.zeros(n, dtype=int)
    indicator[selected] = 1
    return indicator

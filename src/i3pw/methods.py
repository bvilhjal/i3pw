"""High-level selection-bias correction methods and cross-validation.

Each method estimates the population prevalence of one or more binary outcomes
from a biased sample, and reports the percentage difference from the truth. The
methods mirror those benchmarked in the R scripts:

- :func:`no_correction` – naive prevalence in the observed sample (baseline).
- :func:`lasso_ipw` – a single LASSO logistic inclusion model (``cv.glmnet``),
  one weight per unit shared across outcomes.
- :func:`penalized_ipw` – the informed, prevalence-penalized IPW estimator,
  with cross-validated ``(lambda, gamma)`` and several weight-combination rules.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .dgm import Dataset
from .metrics import percent_difference, weighted_mse, weighted_prevalence
from .penalized import PenalizedIPW
from .weights import combine_weights


@dataclass
class MethodResult:
    """Outcome-level estimates from a correction method."""

    name: str
    weighted_prevalence: np.ndarray  # (Q,)
    percent_diff: np.ndarray  # (Q,)
    population_prevalence: np.ndarray  # (Q,)
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{self.name}:"]
        for q in range(len(self.weighted_prevalence)):
            lines.append(
                f"  Y{q + 1}: est={self.weighted_prevalence[q]:.4f} "
                f"true={self.population_prevalence[q]:.4f} "
                f"(% diff {self.percent_diff[q]:.2f})"
            )
        return "\n".join(lines)


def _test_arrays(dataset: Dataset):
    X_test, Y_test, s_test = dataset.split("test")
    return X_test, Y_test, s_test


def no_correction(dataset: Dataset) -> MethodResult:
    """Naive prevalence: the outcome mean among sampled units in the test fold."""
    _, Y_test, s_test = _test_arrays(dataset)
    mask = s_test == 1
    est = Y_test[mask].mean(axis=0)
    pop = dataset.population_prevalence
    pdiff = np.array([percent_difference(est[q], pop[q]) for q in range(len(pop))])
    return MethodResult("no_correction", est, pdiff, pop)


def lasso_ipw(
    dataset: Dataset,
    *,
    interactions: bool = False,
    cv: int = 5,
    Cs=None,
    max_iter: int = 1000,
) -> MethodResult:
    """LASSO logistic IPW with a single inclusion model shared across outcomes.

    Fits ``P(selected | X)`` by L1-penalized logistic regression with the
    penalty strength chosen by cross-validation (the scikit-learn analogue of
    ``cv.glmnet(..., family="binomial")``). One weight per unit is applied to
    every outcome. With ``interactions=True`` all pairwise products of the
    covariates are added, mirroring the ``(X)^2`` design in the R code.

    ``Cs`` is the inverse-regularization grid; it defaults to
    ``numpy.logspace(-3, 1, 8)`` (moderate-to-mild L1). This is deliberately
    kept away from very large ``C`` (near-zero regularization): the inclusion
    problem is near-separable, so an under-regularized fit both defeats the
    purpose of the LASSO and makes liblinear iterate pathologically.
    """
    X_train, _, s_train = dataset.split("train")
    X_test, Y_test, s_test = _test_arrays(dataset)

    if interactions:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)

    if Cs is None:
        Cs = np.logspace(-3, 1, 8)

    # Pure L1 (LASSO) logistic regression with CV-selected penalty strength —
    # the scikit-learn analogue of cv.glmnet(family="binomial").
    lr_kwargs = dict(Cs=Cs, cv=cv, scoring="neg_log_loss", max_iter=max_iter)
    params = inspect.signature(LogisticRegressionCV).parameters
    if "penalty" in params:
        # liblinear runs glmnet-style coordinate descent: far faster than saga
        # on these problem sizes. `penalty` is deprecated (but present) in recent
        # scikit-learn; silence just that forward-compat notice.
        lr_kwargs.update(solver="liblinear", penalty="l1")
    else:
        # `penalty` removed in a future scikit-learn: use the elastic-net API.
        lr_kwargs.update(solver="saga", l1_ratios=(1.0,))
    if "use_legacy_attributes" in params:
        lr_kwargs["use_legacy_attributes"] = False

    # Standardize first, as glmnet does internally: it puts the shared L1 penalty
    # on a comparable scale across covariates and speeds convergence.
    model = make_pipeline(StandardScaler(), LogisticRegressionCV(**lr_kwargs))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        model.fit(X_train, s_train)
    P_test = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)

    weight = np.where(s_test == 1, (1.0 - P_test) / P_test, 1.0)
    pop = dataset.population_prevalence
    est = np.array([weighted_prevalence(weight, Y_test[:, q]) for q in range(len(pop))])
    pdiff = np.array([percent_difference(est[q], pop[q]) for q in range(len(pop))])
    mse = np.array([weighted_mse(weight, P_test, Y_test[:, q]) for q in range(len(pop))])
    return MethodResult("lasso_ipw", est, pdiff, pop, extra={"weight": weight, "mse": mse})


def _prevalence_recovery_score(est, X_val, s_val, Y_val, pop, sample_prev):
    """Mean squared logit-gap between reweighted validation prevalence and truth."""
    from ._links import logit

    w = combine_weights(
        est.weights(X_val, s_val), "mean",
        pop_prevalence=pop, sample_prevalence=sample_prev,
    )
    gaps = []
    for q in range(len(pop)):
        est_prev = weighted_prevalence(w, Y_val[:, q])
        gaps.append((logit(np.array([est_prev]))[0] - logit(np.array([pop[q]]))[0]) ** 2)
    return float(np.mean(gaps))


def cross_validate(
    X_train: np.ndarray,
    s_train: np.ndarray,
    population_prevalence: np.ndarray,
    lambdas,
    gammas,
    *,
    K: int = 5,
    seed: int | None = 0,
    criterion: str = "prevalence",
    Y_train: np.ndarray | None = None,
    sample_prevalence: np.ndarray | None = None,
    **estimator_kwargs,
):
    """Grid-search ``(lambda, gamma)`` by K-fold cross-validation.

    Parameters
    ----------
    criterion:
        ``"prevalence"`` (default) scores each fold by how closely the reweighted
        validation prevalence matches the known population prevalence (mean
        squared logit gap). This makes the informed penalty ``gamma`` genuinely
        selectable — it rewards weights that recover the target prevalence
        out-of-fold. ``"objective"`` reproduces the R behaviour of minimizing the
        mean penalized objective; because that objective grows with ``gamma`` it
        structurally favours ``gamma = 0``.

    Returns ``(best_lambda, best_gamma, cv_table)``.
    """
    if criterion not in ("prevalence", "objective"):
        raise ValueError("criterion must be 'prevalence' or 'objective'.")
    if criterion == "prevalence" and Y_train is None:
        raise ValueError("criterion='prevalence' requires Y_train.")

    rng = np.random.default_rng(seed)
    pop = np.atleast_1d(np.asarray(population_prevalence, dtype=float))
    n = X_train.shape[0]
    folds = rng.integers(0, K, size=n)
    cv_table: dict[tuple[float, float], float] = {}

    for lam in lambdas:
        for gamma in gammas:
            scores = []
            for k in range(K):
                tr, va = folds != k, folds == k
                if va.sum() == 0 or tr.sum() == 0:
                    continue
                est = PenalizedIPW(lam=lam, gamma=gamma, **estimator_kwargs)
                est.fit(X_train[tr], s_train[tr], pop)
                if criterion == "objective":
                    scores.append(est.score_objective(X_train[va], s_train[va]))
                else:
                    scores.append(
                        _prevalence_recovery_score(
                            est, X_train[va], s_train[va], Y_train[va], pop, sample_prevalence
                        )
                    )
            cv_table[(lam, gamma)] = float(np.mean(scores)) if scores else np.inf

    best = min(cv_table, key=cv_table.get)
    return best[0], best[1], cv_table


def penalized_ipw(
    dataset: Dataset,
    lambdas=(0.001, 0.01, 0.1),
    gammas=(0.0, 0.01, 0.1, 1.0),
    *,
    K: int = 5,
    optimizer: str = "gd",
    combine: str | tuple[str, ...] = ("mean", "product", "harmonic", "absdiff"),
    cv_seed: int | None = 0,
    cv_criterion: str = "prevalence",
    **estimator_kwargs,
) -> dict:
    """Cross-validated, prevalence-penalized IPW — the core ``i3pw`` estimator.

    Selects ``(lambda, gamma)`` by K-fold CV on the training fold, refits on the
    full training fold, then evaluates weighted prevalence on the test fold for
    each requested weight-combination rule. ``cv_criterion`` is passed through to
    :func:`cross_validate` (``"prevalence"`` by default so the informed penalty
    is genuinely selectable).

    Returns a dict with a :class:`MethodResult` per combine rule (keyed by rule
    name), plus the chosen ``best_lambda`` / ``best_gamma`` and the fitted
    estimator under ``"estimator"``.
    """
    combine_methods = (combine,) if isinstance(combine, str) else tuple(combine)
    X_train, Y_train, s_train = dataset.split("train")
    X_test, Y_test, s_test = _test_arrays(dataset)
    pop = dataset.population_prevalence
    sample_prev = dataset.sample_prevalence

    best_lambda, best_gamma, cv_table = cross_validate(
        X_train, s_train, pop, lambdas, gammas,
        K=K, seed=cv_seed, optimizer=optimizer,
        criterion=cv_criterion, Y_train=Y_train, sample_prevalence=sample_prev,
        **estimator_kwargs,
    )

    est = PenalizedIPW(lam=best_lambda, gamma=best_gamma, optimizer=optimizer, **estimator_kwargs)
    est.fit(X_train, s_train, pop)

    per_outcome_w = est.weights(X_test, s_test)  # (n, Q)
    P_test = est.predict_inclusion(X_test)  # (n, Q)

    results: dict = {
        "best_lambda": best_lambda,
        "best_gamma": best_gamma,
        "cv_table": cv_table,
        "estimator": est,
    }
    for method in combine_methods:
        w = combine_weights(
            per_outcome_w, method,
            pop_prevalence=pop, sample_prevalence=sample_prev,
        )
        est_prev = np.array([weighted_prevalence(w, Y_test[:, q]) for q in range(len(pop))])
        pdiff = np.array([percent_difference(est_prev[q], pop[q]) for q in range(len(pop))])
        mse = np.array([weighted_mse(w, P_test[:, q], Y_test[:, q]) for q in range(len(pop))])
        results[method] = MethodResult(
            f"penalized_ipw[{method}]", est_prev, pdiff, pop,
            extra={"weight": w, "mse": mse},
        )
    return results

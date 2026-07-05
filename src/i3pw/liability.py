"""Liability-threshold (probit) model with outcome-dependent selection.

A binary outcome is modelled as a hidden continuous **liability** crossing a
threshold: ``L = f(X) + e`` and ``Y = 1[L > t]``, with ``t = Phi^-1(1 - K)`` set by
the population prevalence ``K``. The estimand is how much of the liability the
predictors explain, ``R2_L = Var(f(X)) / Var(L)`` — a signal-to-total variance
ratio on the latent scale (this is "heritability" in one field, but nothing here
is specific to that).

The catch: the sample is ascertained on the outcome, so cases are over-represented
(sample case fraction ``P`` differs from ``K``), which distorts any naive estimate.
Two ways to undo that distortion and recover ``R2_L``:

- **Lee et al. (2011) transform** — estimate the variance explained on the raw 0/1
  scale, then multiply by two analytic factors: ``K(1-K)/z^2`` (moves a 0/1-scale
  quantity to the latent scale, ``z = phi(t)``) and ``K(1-K)/(P(1-P))`` (undoes the
  case over-sampling). A closed form.
- **IPW** — first reweight so the case fraction goes back to ``K`` (weights ``K/P``
  for cases, ``(1-K)/(1-P)`` for controls — the exact inverse-probability weights
  when selection depends only on ``Y``), run a *weighted* estimate, then apply just
  the ``K(1-K)/z^2`` scale factor. A design-based reweighting.

Same job, two routes: Lee corrects for the over-sampling with a formula, IPW does it
by reweighting. They agree when effects and ascertainment are mild. IPW is exact at
any ascertainment strength; the Lee ascertainment factor is a linearization that
drifts low when effects are large and cases are heavily over-sampled.

Under the hood the 0/1-scale variance explained comes from a method-of-moments
(Haseman–Elston-type) regression: regress each pair's outcome product ``y_i y_j`` on
the predictor similarity ``A_ij`` — the standard estimator for a variance component
spread over many weak predictors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import norm


def liability_threshold(K: float) -> tuple[float, float]:
    """Return ``(t, z)``: the liability threshold and the normal density there."""
    t = norm.ppf(1.0 - K)
    return float(t), float(norm.pdf(t))


def observed_to_liability(r2_obs: float, K: float) -> float:
    """Population observed-scale -> liability-scale factor ``r2_obs * K(1-K)/z^2``.

    The Dempster–Lerner / Robertson transform for a *random* (non-ascertained)
    sample from a threshold model.
    """
    _, z = liability_threshold(K)
    return r2_obs * K * (1.0 - K) / z**2


def lee_transform(r2_obs_ascertained: float, K: float, P: float) -> float:
    """Lee et al. (2011) transform for an ascertained case-control sample.

    ``R2_L = r2_obs * [K(1-K)/z^2] * [K(1-K)/(P(1-P))]`` where ``P`` is the case
    proportion in the sample. Reduces to :func:`observed_to_liability` when ``P == K``.
    """
    _, z = liability_threshold(K)
    return r2_obs_ascertained * (K * (1.0 - K) / z**2) * (K * (1.0 - K) / (P * (1.0 - P)))


def similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Predictor similarity matrix ``X X^T / M`` for standardized predictors ``X``."""
    X = np.asarray(X, dtype=float)
    return (X @ X.T) / X.shape[1]


def moment_slope(A: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Method-of-moments (Haseman–Elston-type) estimate of the observed-scale variance explained.

    Regresses the pairwise products ``y_i y_j`` on the off-diagonal similarities
    ``A_ij`` (self-pairs excluded). ``y`` must be standardized to zero mean / unit
    variance under the relevant weighting. With ``weights`` this is the weighted
    regression with pair weight ``w_i w_j``.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.diag(A)
    if weights is None:
        u = y
        num = u @ A @ u - np.sum(d * u * u)
        den = np.sum(A * A) - np.sum(d * d)
    else:
        w = np.asarray(weights, dtype=float)
        u = w * y
        num = u @ A @ u - np.sum(d * u * u)
        den = w @ (A * A) @ w - np.sum((w * w) * (d * d))
    return float(num / den)


@dataclass
class AscertainedSample:
    X: np.ndarray          # (n, M) standardized predictors of the ascertained sample
    y: np.ndarray          # (n,) 0/1 outcome
    case_fraction: float   # realised P
    true_r2: float         # liability-scale variance explained of the generating model


def simulate_case_control(
    n_cases: int,
    n_controls: int,
    n_predictors: int,
    r2: float,
    K: float,
    rng: np.random.Generator,
    *,
    batch: int = 20000,
) -> AscertainedSample:
    """Simulate an ascertained sample from a probit / liability-threshold model.

    Standardized Gaussian predictors ``X`` with fixed effects ``beta ~ N(0, r2/M)``
    give a signal ``f(X) = X beta`` with ``Var(f) = r2``; liability ``L = f + e``,
    ``e ~ N(0, 1-r2)``; outcome ``Y = 1[L > Phi^-1(1-K)]``. Cases and controls are
    drawn by rejection from the population, so the predictor distribution is
    correctly shifted in cases.
    """
    beta = rng.normal(0.0, np.sqrt(r2 / n_predictors), size=n_predictors)
    var_f = float(beta @ beta)
    true_r2 = var_f / (var_f + (1.0 - r2))
    t, _ = liability_threshold(K)

    Xc: list[np.ndarray] = []
    Xk: list[np.ndarray] = []
    n_c = n_k = 0
    while n_c < n_cases or n_k < n_controls:
        X = rng.standard_normal(size=(batch, n_predictors))
        L = X @ beta + rng.normal(0.0, np.sqrt(1.0 - r2), size=batch)
        is_case = L > t
        if n_c < n_cases:
            take = X[is_case][: n_cases - n_c]
            Xc.append(take)
            n_c += take.shape[0]
        if n_k < n_controls:
            take = X[~is_case][: n_controls - n_k]
            Xk.append(take)
            n_k += take.shape[0]

    X = np.vstack(Xc + Xk)
    y = np.concatenate([np.ones(n_cases), np.zeros(n_controls)])
    return AscertainedSample(
        X=X, y=y, case_fraction=n_cases / (n_cases + n_controls), true_r2=true_r2
    )


def estimate_liability_r2(sample: AscertainedSample, K: float, method: str) -> float:
    """Estimate liability-scale variance explained from an ascertained sample.

    ``method`` is one of:

    - ``"naive"`` — observed-scale moment estimate, population transform only
      (no ascertainment correction).
    - ``"lee"``   — observed-scale moment estimate, full Lee et al. transform.
    - ``"ipw"``   — reweight to population case fraction ``K``, weighted moment
      estimate, population transform (the design-based route).
    """
    A = similarity_matrix(sample.X)
    y = sample.y
    P = float(y.mean())

    if method in ("naive", "lee"):
        ys = (y - P) / np.sqrt(P * (1.0 - P))
        r2_obs = moment_slope(A, ys)
        if method == "naive":
            return observed_to_liability(r2_obs, K)
        return lee_transform(r2_obs, K, P)
    if method == "ipw":
        w = np.where(y == 1, K / P, (1.0 - K) / (1.0 - P))
        yk = (y - K) / np.sqrt(K * (1.0 - K))
        r2_obs_pop = moment_slope(A, yk, weights=w)
        return observed_to_liability(r2_obs_pop, K)
    raise ValueError("method must be 'naive', 'lee', or 'ipw'.")


def liability_r2_from_weights(X_sel, y_sel, K: float, weights: np.ndarray) -> float:
    """Liability-scale ``R2`` from an ascertained sample with *arbitrary* IPW weights.

    Generalizes the ``"ipw"`` branch of :func:`estimate_liability_r2` to any weights
    (e.g. inverse of the *true* or a *fitted* selection probability, not just the
    simple case/control ``K/P`` weights). Weighted moment estimate on the
    population-standardized outcome, then the population observed->liability factor.
    """
    A = similarity_matrix(X_sel)
    yk = (np.asarray(y_sel, dtype=float) - K) / np.sqrt(K * (1.0 - K))
    w = np.asarray(weights, dtype=float)
    return observed_to_liability(moment_slope(A, yk, weights=w / w.sum()), K)


@dataclass
class SelectionPopulation:
    """A simulated population under a (possibly liability-dependent) selection scheme."""

    X: np.ndarray               # (n_pop, M) standardized predictors
    y: np.ndarray               # (n_pop,) 0/1 outcome
    selected: np.ndarray        # (n_pop,) bool inclusion indicator
    inclusion_prob: np.ndarray  # (n_pop,) true P(S=1 | Y, L)
    true_r2: float              # liability-scale variance explained
    K: float                    # realized population prevalence

    def sample(self):
        """Return ``(X, y, inclusion_prob)`` for the selected units only."""
        s = self.selected
        return self.X[s], self.y[s], self.inclusion_prob[s]


def simulate_liability_selection(
    n_pop: int,
    n_predictors: int,
    r2: float,
    prevalence: float,
    rng: np.random.Generator,
    *,
    log_odds_case: float = 2.0,
    log_odds_control: float = -3.0,
    liability_slope: float = 0.0,
) -> SelectionPopulation:
    """Simulate a population and a selection scheme that may depend on the liability.

    Selection is Bernoulli with ``logit P(S=1 | Y, L) = a_Y + delta * L``, where
    ``a_Y`` is ``log_odds_case`` for cases and ``log_odds_control`` for controls,
    and ``delta = liability_slope``. With ``delta = 0`` this is simple case-control
    ascertainment (a function of ``Y`` only); with ``delta != 0`` selection also
    depends on the *latent* liability within each group — the regime where the Lee
    transform and simple ``K/P`` IPW break, but IPW with the true (or a well-fit)
    inclusion probability still works.
    """
    beta = rng.normal(0.0, np.sqrt(r2 / n_predictors), size=n_predictors)
    var_f = float(beta @ beta)
    true_r2 = var_f / (var_f + (1.0 - r2))
    X = rng.standard_normal(size=(n_pop, n_predictors))
    L = X @ beta + rng.normal(0.0, np.sqrt(1.0 - r2), size=n_pop)
    t, _ = liability_threshold(prevalence)
    y = (L > t).astype(float)
    lin = np.where(y == 1, log_odds_case, log_odds_control) + liability_slope * L
    pi = expit(lin)
    selected = rng.uniform(size=n_pop) < pi
    return SelectionPopulation(
        X=X, y=y, selected=selected, inclusion_prob=pi, true_r2=true_r2, K=float(y.mean())
    )

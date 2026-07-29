"""Uncertainty for calibration estimators: linearization SEs, bootstrap, K-sensitivity.

Point estimates and the Kish effective sample size are not enough for a methods
package. This module adds three complementary pieces:

- :func:`weighted_mean_se` — the design-based (linearization / sandwich) standard error
  of a Hájek weighted mean or prevalence with **fixed** weights. Cheap and exact for
  independent units, but it treats the weights as given, so it ignores the extra
  variability from having *estimated* them.
- :func:`bootstrap_calibration_ipw` — a nonparametric bootstrap over the sampled units
  that re-solves the calibration each replicate, so it *does* capture weight-estimation
  variability (and, with ``refit_base=True``, the participation-model uncertainty too).
  Anchored outcomes come back with near-zero SE by construction — that is the honest
  read: conditional on the known prevalences, the anchored margins carry no sampling
  uncertainty; the interesting variance is in the *unanchored* and downstream estimands.
- :func:`prevalence_sensitivity` — registry prevalences are not exact constants (they
  carry age/period, ascertainment, diagnostic, and linkage uncertainty), so this sweeps
  the known ``K`` and reports how each estimand and the ESS move with it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .calibration import (
    CalibrationWarning,
    base_weights,
    effective_sample_size,
    entropy_balance,
)
from .methods import _test_arrays
from .metrics import weighted_prevalence


@dataclass
class Estimate:
    """A point estimate with a standard error and a normal-approximation interval."""

    value: float
    se: float
    ci_low: float
    ci_high: float
    level: float = 0.95

    def summary(self) -> str:
        pct = int(round(100 * self.level))
        return (f"{self.value:.4f} ± {self.se:.4f} "
                f"({pct}% CI [{self.ci_low:.4f}, {self.ci_high:.4f}])")


def weighted_mean_se(values: np.ndarray, weights: np.ndarray, *, level: float = 0.95) -> Estimate:
    r"""Linearization standard error of a Hájek weighted mean with fixed weights.

    For ``mu = sum_i w_i y_i / sum_i w_i`` with the weights treated as given, the
    design-based (influence-function / sandwich) variance is

        Var(mu) = sum_i w_i^2 (y_i - mu)^2 / (sum_i w_i)^2,

    the standard ratio-estimator variance for independent units. Works for a mean or a
    0/1 prevalence.

    It conditions on the weights, so it does not describe the uncertainty of a
    *calibration* estimate — and, importantly, it is **not a bound in either
    direction**. On an anchored margin it badly *overstates*: calibration reproduces the
    known prevalence exactly every time, so that quantity's true sampling variability is
    zero while this formula still returns a positive SE (~0.04 on the package's own
    demo). On an estimand uncorrelated with the anchors it is about right, and it can
    understate when the weights are noisy. Use :func:`bootstrap_calibration_ipw`, which
    re-solves the calibration each replicate, whenever the weights were estimated.
    """
    y = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if y.shape != w.shape:
        raise ValueError("values and weights must have the same length.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    total = w.sum()
    if total == 0:
        raise ValueError("weights sum to zero.")
    mu = float((w * y).sum() / total)
    var = float(np.sum(w**2 * (y - mu) ** 2) / total**2)
    se = float(np.sqrt(var))
    z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
    return Estimate(value=mu, se=se, ci_low=mu - z * se, ci_high=mu + z * se, level=level)


@dataclass
class BootstrapResult:
    """Bootstrap distribution of the per-outcome prevalence estimates from calibration."""

    estimate: np.ndarray   # (Q,) point estimate on the full sample
    se: np.ndarray         # (Q,) bootstrap standard error
    ci_low: np.ndarray     # (Q,) percentile lower bound
    ci_high: np.ndarray    # (Q,) percentile upper bound
    replicates: np.ndarray  # (n_kept, Q) — only the replicates that calibrated successfully
    anchor_outcomes: tuple[int, ...]
    level: float = 0.95
    n_boot: int = 0      # replicates attempted
    n_failed: int = 0    # ...of which discarded for failing to meet the calibration targets

    @property
    def failure_rate(self) -> float:
        """Fraction of attempted replicates discarded as uncalibrated.

        Read this as a warning about the interval, not just a count. Discarded replicates
        are the ones that could not reach an anchored target — overwhelmingly the
        resamples that drew too few cases of a rare outcome. That is a *selective* loss
        from the tail of the sampling distribution, so a non-zero failure rate means the
        reported interval is too narrow, by more the higher this number climbs.
        """
        return self.n_failed / self.n_boot if self.n_boot else 0.0

    def summary(self) -> str:
        pct = int(round(100 * self.level))
        lines = [f"bootstrap ({self.replicates.shape[0]} reps, {pct}% percentile CI):"]
        for q in range(self.estimate.shape[0]):
            tag = " (anchored)" if q in self.anchor_outcomes else ""
            lines.append(
                f"  Y{q + 1}: {self.estimate[q]:.4f} ± {self.se[q]:.4f} "
                f"[{self.ci_low[q]:.4f}, {self.ci_high[q]:.4f}]{tag}"
            )
        if self.n_failed:
            lines.append(
                f"  ({self.n_failed}/{self.n_boot} replicates discarded — "
                f"{100 * self.failure_rate:.1f}% could not meet the calibration targets, "
                "typically resamples with no cases of a rare anchored outcome)"
            )
            lines.append(
                "  NOTE: that discard is not random. It removes precisely the resamples "
                "poorest in cases of the rare anchored outcome, which are the ones "
                "carrying the tail of the sampling distribution, so the interval above "
                "is too NARROW — anti-conservative, and most so for the rarest outcome. "
                "Treat it as a lower bound on the uncertainty and widen it by hand, or "
                "reduce the discard rate (drop the rare anchor, or set shrinkage > 0)."
            )
        return "\n".join(lines)


def bootstrap_calibration_ipw(
    dataset,
    *,
    anchor_outcomes=None,
    base: str = "lasso",
    base_scheme: str = "inverse",
    shrinkage: float = 0.0,
    interactions: bool = False,
    cv: int = 5,
    n_boot: int = 200,
    refit_base: bool = False,
    seed: int = 0,
    level: float = 0.95,
) -> BootstrapResult:
    """Nonparametric bootstrap of :func:`calibration_ipw` over the sampled units.

    Each replicate resamples the sampled test units with replacement and re-solves the
    calibration, so the spread reflects the sampling variability of the *estimated*
    weights — not captured by :func:`weighted_mean_se`. With ``refit_base=True`` the
    covariate participation model is also refit on a resampled training frame each
    replicate, folding in base-model uncertainty (slower: one LASSO fit per replicate).

    Parameters mirror :func:`calibration_ipw`; ``n_boot`` replicates, ``seed`` for the
    resampling, ``level`` for the percentile interval.

    **The interval is anti-conservative when replicates are discarded.** A resample that
    draws no cases of a rare anchored outcome cannot meet its target, and is dropped
    rather than folded in (see :func:`estimate_from` below for why keeping it would be
    worse). But that rule fires on exactly the resamples poorest in rare cases — the ones
    that would have populated the tail — so what is left is a truncated sampling
    distribution and the percentile interval comes out too narrow. It is not a bias that
    can be corrected here, only reported: check
    :attr:`BootstrapResult.failure_rate`, and treat a non-zero one as saying the true
    interval is wider than the printed one. To shrink the effect rather than live with
    it, drop the rare anchor or set ``shrinkage > 0`` so replicates stop failing.
    """
    X_train, _, s_train = dataset.split("train")
    X_test, Y_test, s_test = _test_arrays(dataset)
    pop = np.asarray(dataset.population_prevalence, dtype=float)
    q = len(pop)
    anchors = tuple(range(q)) if anchor_outcomes is None else tuple(anchor_outcomes)
    sel = s_test == 1
    X_sel, Y_sel = X_test[sel], Y_test[sel]
    n = Y_sel.shape[0]
    n_train = X_train.shape[0]
    anchor_targets = pop[list(anchors)]

    def estimate_from(Yb: np.ndarray, bw: np.ndarray) -> np.ndarray | None:
        """Estimates for one resample, or ``None`` if its calibration did not solve.

        A resample that happens to contain no cases of a rare anchored outcome puts the
        target outside its convex hull; the tilt then runs to a corner instead of meeting
        the constraint. Such a replicate is not a draw from the estimator's sampling
        distribution, and one infeasible constraint corrupts the *other* outcomes'
        estimates too — so it is discarded rather than folded into the interval.
        """
        w, diag = entropy_balance(Yb[:, anchors], anchor_targets, base_weights=bw,
                                  ridge=shrinkage, warn=False, return_diagnostics=True)
        if shrinkage == 0.0 and (not diag.converged or diag.max_abs_residual > 1e-6):
            return None
        return np.array([weighted_prevalence(w, Yb[:, j]) for j in range(q)])

    bw_full = base_weights(base, base_scheme, X_train, s_train, X_sel,
                           interactions=interactions, cv=cv)
    point = estimate_from(Y_sel, bw_full)
    if point is None:
        raise ValueError(
            "bootstrap_calibration_ipw: the calibration does not solve on the full "
            "sample, so there is no point estimate to bootstrap. Check anchor support "
            "with calibration_ipw(...).diagnostics_summary()."
        )

    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    n_failed = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Yb = Y_sel[idx]
        if refit_base and base == "lasso":
            tr = rng.integers(0, n_train, size=n_train)
            bw = base_weights(base, base_scheme, X_train[tr], s_train[tr], X_sel[idx],
                              interactions=interactions, cv=cv)
        else:
            bw = bw_full[idx]
        rep = estimate_from(Yb, bw)
        if rep is None:
            n_failed += 1
        else:
            kept.append(rep)

    if len(kept) < 2:
        raise ValueError(
            f"bootstrap_calibration_ipw: only {len(kept)} of {n_boot} replicates "
            "calibrated successfully; the anchored outcome(s) are too rare in this "
            "sample to bootstrap. Anchor fewer outcomes, or use shrinkage > 0."
        )
    reps = np.vstack(kept)
    if n_failed:
        warnings.warn(
            f"bootstrap_calibration_ipw: discarded {n_failed}/{n_boot} "
            f"({100 * n_failed / n_boot:.1f}%) replicates whose calibration did not "
            "meet the targets (typically resamples with no cases of a rare anchored "
            "outcome). Intervals are computed from the remaining replicates.",
            CalibrationWarning, stacklevel=2,
        )

    alpha = 1.0 - level
    return BootstrapResult(
        estimate=point,
        se=reps.std(axis=0, ddof=1),
        ci_low=np.quantile(reps, alpha / 2.0, axis=0),
        ci_high=np.quantile(reps, 1.0 - alpha / 2.0, axis=0),
        replicates=reps,
        anchor_outcomes=anchors,
        level=level,
        n_boot=n_boot,
        n_failed=n_failed,
    )


@dataclass
class SensitivityResult:
    """Estimand and ESS as the known anchored prevalences are scaled by ``1 + delta``."""

    rel_deltas: np.ndarray    # (D,) relative perturbations applied to the anchored K
    anchor_outcomes: tuple[int, ...]
    estimates: np.ndarray     # (D, Q) weighted prevalence of each outcome at each delta
    ess: np.ndarray           # (D,) effective sample size at each delta
    spread: np.ndarray        # (Q,) max - min estimate over the grid, per outcome

    def summary(self) -> str:
        grid = [round(float(d), 3) for d in self.rel_deltas]
        lines = [f"prevalence sensitivity over relative deltas {grid}:"]
        for q in range(self.spread.shape[0]):
            tag = " (anchored)" if q in self.anchor_outcomes else ""
            lo, hi = self.estimates[:, q].min(), self.estimates[:, q].max()
            lines.append(f"  Y{q + 1}: range [{lo:.4f}, {hi:.4f}] spread {self.spread[q]:.4f}{tag}")
        return "\n".join(lines)


def prevalence_sensitivity(
    dataset,
    *,
    anchor_outcomes=None,
    base: str = "lasso",
    base_scheme: str = "inverse",
    shrinkage: float = 0.0,
    interactions: bool = False,
    cv: int = 5,
    rel_deltas=(-0.2, -0.1, 0.0, 0.1, 0.2),
) -> SensitivityResult:
    """Sensitivity of the calibration estimand to error in the known prevalences.

    Holding the base weights fixed, scale every anchored target prevalence by
    ``1 + delta`` (a common relative perturbation, e.g. registry miscalibration) across
    the ``rel_deltas`` grid and record the resulting weighted prevalence of every outcome
    and the effective sample size. An anchored outcome tracks its perturbed target by
    construction; the informative movement is in the *unanchored* outcomes (and, with the
    weights, any downstream estimand) and in how fast the ESS degrades.
    """
    X_train, _, s_train = dataset.split("train")
    X_test, Y_test, s_test = _test_arrays(dataset)
    pop = np.asarray(dataset.population_prevalence, dtype=float)
    q = len(pop)
    anchors = tuple(range(q)) if anchor_outcomes is None else tuple(anchor_outcomes)
    sel = s_test == 1
    X_sel, Y_sel = X_test[sel], Y_test[sel]
    bw = base_weights(base, base_scheme, X_train, s_train, X_sel,
                      interactions=interactions, cv=cv)

    deltas = np.asarray(rel_deltas, dtype=float)
    base_targets = pop[list(anchors)]
    estimates = np.empty((deltas.shape[0], q))
    ess = np.empty(deltas.shape[0])
    for i, d in enumerate(deltas):
        targets = np.clip(base_targets * (1.0 + d), 1e-9, 1.0 - 1e-9)
        w = entropy_balance(Y_sel[:, anchors], targets, base_weights=bw,
                            ridge=shrinkage, warn=False)
        estimates[i] = [weighted_prevalence(w, Y_sel[:, j]) for j in range(q)]
        ess[i] = effective_sample_size(w)

    spread = estimates.max(axis=0) - estimates.min(axis=0)
    return SensitivityResult(
        rel_deltas=deltas, anchor_outcomes=anchors,
        estimates=estimates, ess=ess, spread=spread,
    )

"""Monte Carlo evaluation of the correction methods.

A single simulated dataset shows a method *can* reduce bias on one draw; it does
not show the method is reliably (approximately) unbiased. This module repeats the
whole simulate → bias → correct pipeline across many random populations and
summarizes each method's error distribution — the honest way to compare methods.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .calibration import calibration_ipw
from .dgm import make_dataset
from .methods import lasso_ipw, no_correction


@dataclass
class MonteCarloSummary:
    """Per-method error distribution across replications."""

    method: str
    mean_pct_error: np.ndarray  # (Q,) mean absolute % error per outcome
    sd_pct_error: np.ndarray    # (Q,) SD of absolute % error per outcome
    n_reps: int
    n_nan: int = 0              # error values excluded as NaN (zero realised prevalence)

    def overall(self) -> float:
        """Mean absolute % error averaged over outcomes (a single headline number)."""
        return float(np.mean(self.mean_pct_error))


def monte_carlo(
    n_reps: int = 20,
    *,
    base_seed: int = 0,
    sim_kwargs: dict | None = None,
    weighting: str = "inverse",
    include_lasso: bool = True,
    include_calibration: bool = True,
    calibration_base: str = "lasso",
    anchor_outcomes=None,
) -> dict[str, MonteCarloSummary]:
    """Repeat the correction pipeline over ``n_reps`` random populations.

    Each replication draws a fresh dataset (seed ``base_seed + rep``), applies the
    methods, and records the absolute percentage error of each method's prevalence
    estimate versus the realised population prevalence. Returns one
    :class:`MonteCarloSummary` per method, keyed by name.

    Parameters
    ----------
    sim_kwargs:
        Overrides forwarded to :func:`i3pw.make_dataset` (minus ``seed``).
    weighting:
        ``"inverse"`` (default, deployable Hájek) or ``"oracle_odds"`` (a
        simulation-only diagnostic) — passed to :func:`i3pw.lasso_ipw`.
    include_calibration / include_lasso:
        Toggle each prevalence-informed / baseline method.
    calibration_base:
        ``base=`` passed to :func:`i3pw.calibration_ipw` (``"lasso"`` or ``"uniform"``).
        Worth sweeping: when the simulation has no covariate selection channel
        (``SimConfig.selection_covariate_strength == 0``, the default) the LASSO base has
        no signal to fit and only adds estimation noise, so ``"uniform"`` is the fairer
        reference point.
    anchor_outcomes:
        Which outcomes :func:`i3pw.calibration_ipw` may calibrate on. **Read this before
        quoting a headline number.** With the default (``None`` = anchor everything) the
        reported error for ``calibration_ipw`` is ``0.00`` *by construction* — the
        estimator reproduces the very prevalences it was handed, which is an algebraic
        identity, not evidence that the method works. Pass a strict subset (e.g. ``[0]``)
        to leave the remaining outcomes unanchored: their error then measures the honest
        quantity, namely how well calibrating on the diseases whose prevalence you know
        *transfers* to one you do not.
    """
    sim_kwargs = dict(sim_kwargs or {})
    sim_kwargs.pop("seed", None)

    errors: dict[str, list[np.ndarray]] = {}

    def record(name: str, pct_error: np.ndarray) -> None:
        errors.setdefault(name, []).append(np.asarray(pct_error, dtype=float))

    for rep in range(n_reps):
        ds = make_dataset(seed=base_seed + rep, **sim_kwargs)

        record("no_correction", no_correction(ds).percent_diff)
        if include_lasso:
            record("lasso_ipw", lasso_ipw(ds, weighting=weighting).percent_diff)
        if include_calibration:
            record(
                "calibration_ipw",
                calibration_ipw(
                    ds, base=calibration_base, anchor_outcomes=anchor_outcomes
                ).percent_diff,
            )

    summaries: dict[str, MonteCarloSummary] = {}
    for name, rows in errors.items():
        stacked = np.vstack(rows)  # (n_reps, Q)
        # percent_difference is NaN when the realised population prevalence is 0
        # (a rare outcome drew no cases). A plain mean would let one such
        # replication turn the whole column NaN with no explanation; exclude the
        # NaNs instead, count them, and say it happened.
        n_nan = int(np.isnan(stacked).sum())
        if n_nan:
            warnings.warn(
                f"monte_carlo: {n_nan} of {stacked.size} error value(s) for {name!r} "
                "are NaN — a replication whose realised population prevalence was 0 "
                "has no defined percent difference. Those values are excluded from "
                "the mean/SD (see MonteCarloSummary.n_nan).",
                stacklevel=2,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(stacked, axis=0)
                sd = (np.nanstd(stacked, axis=0, ddof=1) if len(rows) > 1
                      else np.zeros(stacked.shape[1]))
        else:
            mean = stacked.mean(axis=0)
            sd = stacked.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(stacked.shape[1])
        summaries[name] = MonteCarloSummary(
            method=name,
            mean_pct_error=mean,
            sd_pct_error=sd,
            n_reps=len(rows),
            n_nan=n_nan,
        )
    return summaries


def format_summary(summaries: dict[str, MonteCarloSummary]) -> str:
    """Render Monte Carlo summaries as a fixed-width table (mean % error ± SD)."""
    if not summaries:
        raise ValueError("summaries must contain at least one method summary.")
    any_summary = next(iter(summaries.values()))
    q = len(any_summary.mean_pct_error)
    header = f"{'method':<18}" + "".join(f"{'Y' + str(i + 1) + ' %err':>16}" for i in range(q))
    lines = [header, "-" * len(header)]
    for name, s in summaries.items():
        cells = "".join(f"{s.mean_pct_error[i]:>8.2f}±{s.sd_pct_error[i]:<7.2f}" for i in range(q))
        lines.append(f"{name:<18}{cells}")
    return "\n".join(lines)

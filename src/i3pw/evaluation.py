"""Monte Carlo evaluation of the correction methods.

A single simulated dataset shows a method *can* reduce bias on one draw; it does
not show the method is reliably (approximately) unbiased. This module repeats the
whole simulate → bias → correct pipeline across many random populations and
summarizes each method's error distribution — the honest way to compare methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dgm import make_dataset
from .methods import lasso_ipw, no_correction, penalized_ipw


@dataclass
class MonteCarloSummary:
    """Per-method error distribution across replications."""

    method: str
    mean_pct_error: np.ndarray  # (Q,) mean absolute % error per outcome
    sd_pct_error: np.ndarray    # (Q,) SD of absolute % error per outcome
    n_reps: int

    def overall(self) -> float:
        """Mean absolute % error averaged over outcomes (a single headline number)."""
        return float(np.mean(self.mean_pct_error))


def monte_carlo(
    n_reps: int = 20,
    *,
    base_seed: int = 0,
    sim_kwargs: dict | None = None,
    penalized_kwargs: dict | None = None,
    weighting: str = "odds",
    include_lasso: bool = True,
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
    penalized_kwargs:
        Overrides forwarded to :func:`i3pw.penalized_ipw` (e.g. grids, ``K``,
        ``learning_rate``). ``weighting`` is set separately.
    weighting:
        ``"odds"`` or ``"inverse"`` — passed to both IPW methods.
    """
    sim_kwargs = dict(sim_kwargs or {})
    penalized_kwargs = dict(penalized_kwargs or {})
    sim_kwargs.pop("seed", None)

    errors: dict[str, list[np.ndarray]] = {}

    def record(name: str, pct_error: np.ndarray) -> None:
        errors.setdefault(name, []).append(np.asarray(pct_error, dtype=float))

    for rep in range(n_reps):
        ds = make_dataset(seed=base_seed + rep, **sim_kwargs)

        record("no_correction", no_correction(ds).percent_diff)
        if include_lasso:
            record("lasso_ipw", lasso_ipw(ds, weighting=weighting).percent_diff)
        res = penalized_ipw(ds, weighting=weighting, combine="mean", **penalized_kwargs)
        record("penalized_ipw", res["mean"].percent_diff)

    summaries: dict[str, MonteCarloSummary] = {}
    for name, rows in errors.items():
        stacked = np.vstack(rows)  # (n_reps, Q)
        sd = stacked.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(stacked.shape[1])
        summaries[name] = MonteCarloSummary(
            method=name,
            mean_pct_error=stacked.mean(axis=0),
            sd_pct_error=sd,
            n_reps=len(rows),
        )
    return summaries


def format_summary(summaries: dict[str, MonteCarloSummary]) -> str:
    """Render Monte Carlo summaries as a fixed-width table (mean % error ± SD)."""
    any_summary = next(iter(summaries.values()))
    q = len(any_summary.mean_pct_error)
    header = f"{'method':<18}" + "".join(f"{'Y' + str(i + 1) + ' %err':>16}" for i in range(q))
    lines = [header, "-" * len(header)]
    for name, s in summaries.items():
        cells = "".join(f"{s.mean_pct_error[i]:>8.2f}±{s.sd_pct_error[i]:<7.2f}" for i in range(q))
        lines.append(f"{name:<18}{cells}")
    return "\n".join(lines)

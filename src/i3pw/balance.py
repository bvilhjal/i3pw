"""Covariate balance — the falsification test prevalence calibration admits.

Every diagnostic in :mod:`i3pw.calibration` is a statement about the *solve*: did the
optimizer converge, did it hit the targets, how concentrated are the weights. None of
them can tell you the weights are **wrong**, because a calibration that matches its
constraints exactly does so whether or not the underlying density-ratio model is right.
Worse, a broken run can look *healthier*: fitting a base model that misses a real
participation driver can leave the residual at machine precision with a perfectly
comfortable effective sample size, while the estimand is badly biased.

The way out is standard in the weighting literature and is what this module provides:
check the reweighted sample against population quantities the calibration **did not**
use. Constrained moments match by construction and carry no information. *Unconstrained*
ones do not have to match — so when they do, the density-ratio model has survived a test
it could have failed, and when they do not, the model is refuted.

This is an **overidentification test**: supply more known population quantities than you
calibrate on, and the surplus becomes evidence. Concretely, for a biobank: calibrate to
the known disease prevalences, then check the reweighted sample against register margins
you held back — age, sex, region, education. A large post-weighting discrepancy on a
held-out margin says the tilt family does not contain the true selection mechanism, which
is exactly the assumption that "What is identified?" in ``docs/theory.md`` rests on.

The standardized mean difference (SMD) is the usual currency — the gap between the
weighted sample mean and the population target, in sample standard deviations, so
variables on different scales are comparable. ``|SMD| < 0.1`` is the conventional
"balanced" threshold (Austin 2009; Stuart 2010); it is a rule of thumb, not a test.

References
----------
- Austin, P. C. (2009), *Statistics in Medicine* 28, 3083–3107 — standardized
  differences for comparing covariate distributions between weighted groups.
- Stuart, E. A. (2010), *Statistical Science* 25, 1–21 — matching/weighting diagnostics.
- Sargan (1958) / Hansen (1982) — overidentifying restrictions as a specification test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .calibration import _require_finite


@dataclass
class BalanceReport:
    """Per-quantity balance of a reweighted sample against known population values."""

    names: list[str]
    target: np.ndarray            # (k,) known population mean of each quantity
    unweighted_mean: np.ndarray   # (k,) raw sample mean
    weighted_mean: np.ndarray     # (k,) reweighted sample mean
    smd_before: np.ndarray        # (k,) standardized mean difference, raw sample
    smd_after: np.ndarray         # (k,) ...after reweighting
    constrained: np.ndarray       # (k,) bool: was this quantity a calibration constraint?

    @property
    def worst_held_out(self) -> float:
        """Largest ``|SMD|`` after weighting among quantities that were *not* constrained.

        This single number is the falsification test: constrained moments match by
        construction, so only the held-out ones can refute the weighting. ``nan`` when
        every supplied quantity was constrained (nothing was held out, nothing tested).
        """
        free = ~self.constrained
        if not np.any(free):
            return float("nan")
        return float(np.max(np.abs(self.smd_after[free])))

    def passed(self, threshold: float = 0.1) -> bool:
        """Whether every held-out quantity is balanced to within ``threshold`` SMD.

        Vacuously ``True`` when nothing was held out — check :attr:`worst_held_out` for
        ``nan`` rather than reading that as evidence of a good weighting.
        """
        worst = self.worst_held_out
        return True if np.isnan(worst) else bool(worst <= threshold)

    def summary(self, threshold: float = 0.1) -> str:
        lines = [
            f"{'quantity':<22}{'target':>10}{'unweighted':>12}{'weighted':>10}"
            f"{'SMD before':>12}{'SMD after':>11}",
            "-" * 77,
        ]
        for i, nm in enumerate(self.names):
            tag = " [constrained]" if self.constrained[i] else ""
            ok = self.constrained[i] or abs(self.smd_after[i]) <= threshold
            flag = "" if ok else "  <-- FAILS"
            lines.append(
                f"{nm[:22]:<22}{self.target[i]:>10.4f}{self.unweighted_mean[i]:>12.4f}"
                f"{self.weighted_mean[i]:>10.4f}{self.smd_before[i]:>12.3f}"
                f"{self.smd_after[i]:>11.3f}{tag}{flag}"
            )
        worst = self.worst_held_out
        lines.append("-" * 77)
        if np.isnan(worst):
            lines.append(
                "no held-out quantities: every supplied moment was constrained, so this "
                "report cannot refute the weighting (it only confirms the solve)."
            )
        else:
            verdict = "PASS" if worst <= threshold else "FAIL"
            lines.append(
                f"worst held-out |SMD| after weighting: {worst:.3f} "
                f"({verdict} at threshold {threshold})"
            )
        return "\n".join(lines)


def balance_report(
    features: npt.ArrayLike,
    weights: npt.ArrayLike,
    targets: npt.ArrayLike,
    *,
    constrained: npt.ArrayLike | None = None,
    names: list[str] | None = None,
) -> BalanceReport:
    """Compare a reweighted sample against known population quantities.

    Parameters
    ----------
    features:
        ``(n, k)`` quantities evaluated on the sampled units — covariates, outcomes,
        interactions, anything whose population mean you know.
    weights:
        Length-``n`` calibration (or IPW) weights for those units.
    targets:
        Length-``k`` known population means.
    constrained:
        Length-``k`` boolean flags marking which columns were used as calibration
        constraints. Those match by construction and are excluded from the verdict.
        Defaults to all ``False`` — i.e. treat everything as a held-out test, which is
        the right default when weighting came from a participation model alone.
    names:
        Optional column labels for the report.

    Returns
    -------
    BalanceReport
        Per-quantity SMDs before and after weighting, and a pass/fail verdict driven
        only by the held-out columns.
    """
    F = np.atleast_2d(np.asarray(features, dtype=float))
    t = np.atleast_1d(np.asarray(targets, dtype=float))
    w = np.asarray(weights, dtype=float).ravel()
    if F.shape[0] == 1 and F.shape[1] != t.shape[0]:
        F = F.T
    n, k = F.shape
    if t.shape[0] != k:
        raise ValueError("targets must have one entry per feature column.")
    if w.shape[0] != n:
        raise ValueError("weights must have one entry per row of features.")
    _require_finite(F, "features")
    _require_finite(t, "targets")
    _require_finite(w, "weights")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    if w.sum() == 0:
        raise ValueError("weights sum to zero.")

    flags = (np.zeros(k, dtype=bool) if constrained is None
             else np.asarray(constrained, dtype=bool).ravel())
    if flags.shape[0] != k:
        raise ValueError("constrained must have one flag per feature column.")
    labels = list(names) if names is not None else [f"x{j}" for j in range(k)]
    if len(labels) != k:
        raise ValueError("names must have one label per feature column.")

    p = w / w.sum()
    raw = F.mean(axis=0)
    wtd = p @ F
    # Scale by the *sample* SD so before/after are on one ruler. A degenerate column
    # (zero variance) has no scale, so report its raw gap rather than dividing by zero.
    sd = F.std(axis=0, ddof=1) if n > 1 else np.zeros(k)
    scale = np.where(sd > 0, sd, 1.0)
    return BalanceReport(
        names=labels,
        target=t,
        unweighted_mean=raw,
        weighted_mean=wtd,
        smd_before=(raw - t) / scale,
        smd_after=(wtd - t) / scale,
        constrained=flags,
    )

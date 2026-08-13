"""B4 --- prevalence fixes the case count; it does not fix the case mix.

``docs/theory.md`` states this and the README makes it recommendation 3 -- calibrate
within strata, "usually the single most important step for psychiatric cohorts" --
without a number behind it. This benchmark supplies one, and sharpens the advice:
what matters is not that the calibration is stratified but that it is stratified
*along the axis recruitment acts on*.

Two ways for a matched prevalence to hide a wrong case mix:

``stratum-differential``  cases are recruited harder in some strata than others, so
                          the pooled count is right and its distribution over strata
                          is wrong. Within-stratum prevalences span this exactly.
``within-case severity``  recruitment increases with liability among cases, so the
                          sampled cases are the more severely ill. Demographic strata
                          are the wrong axis and do not touch it; separate mild- and
                          severe-case prevalences do.

Three quantities are scored, and the distinction between them is the point. The
worst within-stratum prevalence error is *constrained* for the stratified estimator
and is reported only to show the identity. The mean liability among cases and the
held-out trait mean are constrained by nothing, in any arm, and are where the
comparison is decided.
"""

from __future__ import annotations

import warnings

import numpy as np

from benchmarks import estimators as E
from benchmarks.harness import Progress, Row, rmse, summarize
from benchmarks.simulate import Design, simulate

BENCHMARK = "B4_case_mix"

BASE_LAW = dict(delta_x=0.65, delta_y=1.70)
CONDITIONS = {
    "pooled only": dict(),
    "stratum-differential": dict(delta_y_stratum=1.30),
    "within-case severity": dict(delta_sev=0.90),
}
METHODS = ("naive", "ipw", "ipw+cal", "ipw+cal/s", "ipw+cal/v", "oracle")


def _worst_stratum_error(pop, wt: E.Weighting) -> float:
    """Largest absolute error in a within-stratum prevalence of the anchored disease."""
    mask = pop.mask
    levels = len(pop.stratum_share)
    dummies = (pop.stratum[mask][:, None] == np.arange(levels)[None, :]).astype(float)
    w = wt.weights[:, None]
    weighted = (w * dummies * pop.Y[mask, 0][:, None]).sum(0) / (w * dummies).sum(0)
    return float(np.max(np.abs(weighted - pop.within_stratum_prevalence()[:, 0])))


def _severe_share_error(pop, wt: E.Weighting) -> float:
    """Error in the share of cases the register would call severe."""
    mask = pop.mask
    case = pop.Y[mask, 0] == 1
    if not np.any(case):
        return float("nan")
    truth = float(pop.severe[pop.Y[:, 0] == 1].mean())
    return E.weighted_mean(pop.severe[mask][case], wt.weights[case]) - truth


def run(n_reps: int = 40) -> list[Row]:
    rows: list[Row] = []
    progress = Progress(BENCHMARK, len(CONDITIONS) * n_reps)
    for condition, channels in CONDITIONS.items():
        acc: dict[str, dict[str, list[float]]] = {
            m: {k: [] for k in ("trait", "mix", "severe", "stratum", "ess")} for m in METHODS
        }
        for seed in range(n_reps):
            pop = simulate(Design(seed=seed, **BASE_LAW, **channels))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_hat = E.propensity(pop)
                weightings = {m: E.fit_weighting(pop, m, p_hat=p_hat) for m in METHODS}
            for name, wt in weightings.items():
                a = acc[name]
                a["trait"].append(E.trait_error(pop, wt))
                a["mix"].append(E.case_mix_error(pop, wt))
                a["severe"].append(_severe_share_error(pop, wt))
                a["stratum"].append(_worst_stratum_error(pop, wt))
                a["ess"].append(wt.ess)
            progress.step()

        for name in METHODS:
            a = acc[name]
            rows.append(summarize(BENCHMARK, condition, name, "trait_bias_sd", a["trait"],
                                  notes="held out in every arm"))
            rows.append(Row(BENCHMARK, condition, name, "trait_rmse_sd", rmse(a["trait"]),
                            None, None, len(a["trait"]),
                            "root mean square of the per-replication signed error"))
            rows.append(summarize(BENCHMARK, condition, name, "case_mix_liability_error",
                                  a["mix"], notes="mean liability among cases; held out "
                                                  "in every arm"))
            rows.append(summarize(
                BENCHMARK, condition, name, "severe_case_share_error", a["severe"],
                notes=("constrained for ipw+cal/v: an identity there"
                       if name == "ipw+cal/v" else "held out"),
            ))
            rows.append(summarize(
                BENCHMARK, condition, name, "worst_within_stratum_prevalence_error",
                a["stratum"],
                notes=("constrained for ipw+cal/s: an identity there"
                       if name == "ipw+cal/s" else "held out"),
            ))
            rows.append(summarize(BENCHMARK, condition, name, "kish_ess", a["ess"]))
    progress.close()
    return rows


def main() -> None:
    for row in run(n_reps=6):
        print(row.as_tsv())


if __name__ == "__main__":
    main()

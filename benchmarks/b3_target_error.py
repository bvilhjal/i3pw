"""B3 --- what a wrong register prevalence costs.

The validation matrix asks for perturbed targets, and the package's fifth
recommendation asks users to report a sensitivity sweep over ``K``. Neither says how
much a given error costs, which is the only form in which the advice is actionable:
register prevalences are estimates -- diagnostic thresholds drift, coding practice
changes, the register's catchment is not exactly the cohort's source population --
so the question is how much bias a plausible discrepancy induces.

The anchored prevalence is scaled by ``1 + delta`` and everything else is held
fixed. What the sweep measures is the *transfer* of that error: the anchored margin
moves with ``delta`` by construction and is reported only to make the identity
visible, while the trait mean and the unanchored disease move by an amount the
estimator's structure decides. The naive and covariate-IPW baselines do not depend
on ``delta`` at all, so they enter as horizontal reference lines: the useful reading
is how large ``delta`` must become before the correction stops being worth making.
"""

from __future__ import annotations

import warnings

import numpy as np

from benchmarks import estimators as E
from benchmarks.harness import Progress, Row, rmse, summarize
from benchmarks.simulate import Design, simulate

BENCHMARK = "B3_target_error"

LAW = dict(delta_x=0.65, delta_y=1.70)
DELTAS = (-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30)


def run(n_reps: int = 40) -> list[Row]:
    rows: list[Row] = []
    progress = Progress(BENCHMARK, n_reps)
    keys = ("trait", "anchored", "prev2", "ess")
    acc: dict[str, dict[str, list[float]]] = {
        f"{d:+.2f}": {k: [] for k in keys} for d in DELTAS
    }
    reference: dict[str, list[float]] = {"naive": [], "ipw": [], "oracle": []}

    for seed in range(n_reps):
        pop = simulate(Design(seed=seed, **LAW))
        truth = pop.population_prevalence
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_hat = E.propensity(pop)
            for name in reference:
                reference[name].append(E.trait_error(pop, E.fit_weighting(pop, name, p_hat=p_hat)))
            for d in DELTAS:
                supposed = np.array([truth[0] * (1.0 + d)])
                wt = E.fit_weighting(pop, "ipw+cal", p_hat=p_hat, targets=supposed)
                a = acc[f"{d:+.2f}"]
                a["trait"].append(E.trait_error(pop, wt))
                a["anchored"].append(E.unanchored_prevalence_error(pop, wt, outcome=0))
                a["prev2"].append(E.unanchored_prevalence_error(pop, wt, outcome=1))
                a["ess"].append(wt.ess)
        progress.step()
    progress.close()

    for d in DELTAS:
        condition = f"{d:+.2f}"
        a = acc[condition]
        rows.append(summarize(BENCHMARK, condition, "ipw+cal", "trait_bias_sd", a["trait"],
                              notes="held-out trait mean under a mis-stated register value"))
        rows.append(Row(BENCHMARK, condition, "ipw+cal", "trait_rmse_sd", rmse(a["trait"]),
                        None, None, len(a["trait"]),
                        "root mean square of the per-replication signed error"))
        rows.append(summarize(BENCHMARK, condition, "ipw+cal", "anchored_prevalence_error_pct",
                              a["anchored"],
                              notes="tracks delta by construction: an identity, not a finding"))
        rows.append(summarize(BENCHMARK, condition, "ipw+cal", "unanchored_prevalence_error_pct",
                              a["prev2"], notes="disease 2, never supplied"))
        rows.append(summarize(BENCHMARK, condition, "ipw+cal", "kish_ess", a["ess"]))

    for name, values in reference.items():
        rows.append(summarize(BENCHMARK, "reference", name, "trait_bias_sd", values,
                              notes="does not depend on delta; a horizontal reference line"))
    return rows


def main() -> None:
    for row in run(n_reps=6):
        print(row.as_tsv())


if __name__ == "__main__":
    main()

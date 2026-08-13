"""B1 --- what each estimator recovers, as the recruitment mechanism changes.

The axis the report's validation matrix puts first. Five recruitment laws, from
one a covariate model handles alone to two that no base-plus-marginal tilt can
represent, scored on quantities none of the estimators were given:

* the population mean of a continuous trait ``Z`` observed only in participants;
* the prevalence of a second disease whose register value was never supplied;
* four covariate margins held out of the solve;
* the mean liability among cases -- the case *mix* behind a matched case count.

The comparator set is the one the matrix asks for: unweighted, covariate IPW,
calibration alone, the two combined, the stratified variant, augmented IPW, and the
oracle weights ``1/pi`` that only a simulation has. The oracle is the row that makes
the others readable: it is what a perfect participation model would deliver at this
sample size, so the gap between it and ``ipw+cal`` is misspecification, and the
distance from zero to the oracle is irreducible Monte Carlo noise.
"""

from __future__ import annotations

import warnings

from benchmarks import estimators as E
from benchmarks.harness import Progress, Row, rmse, summarize
from benchmarks.simulate import SELECTION_LAWS, Design, simulate
from i3pw import aipw_mean

BENCHMARK = "B1_selection_law"
METHODS = ("naive", "ipw", "cal", "ipw+cal", "ipw+cal/s", "oracle")


def run(n_reps: int = 40) -> list[Row]:
    rows: list[Row] = []
    progress = Progress(BENCHMARK, len(SELECTION_LAWS) * n_reps)
    for law, channels in SELECTION_LAWS.items():
        acc: dict[str, dict[str, list[float]]] = {
            m: {k: [] for k in ("trait", "prev2", "smd", "ess", "share", "mix")}
            for m in (*METHODS, "aipw")
        }
        for seed in range(n_reps):
            pop = simulate(Design(seed=seed, **channels))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_hat = E.propensity(pop)
                weightings = {m: E.fit_weighting(pop, m, p_hat=p_hat) for m in METHODS}
            for name, wt in weightings.items():
                a = acc[name]
                a["trait"].append(E.trait_error(pop, wt))
                a["prev2"].append(E.unanchored_prevalence_error(pop, wt))
                a["smd"].append(E.held_out_balance(pop, wt).worst_held_out)
                a["ess"].append(wt.ess)
                a["share"].append(wt.max_share)
                a["mix"].append(E.case_mix_error(pop, wt))

            # Augmentation: the recommended weights plus an outcome regression for
            # the trait. Cross-fitted so the residual term is not read off a model
            # that has already seen the unit.
            frame = pop.frame()
            augmented = aipw_mean(
                frame, pop.mask, pop.Z[pop.mask], weightings["ipw+cal"].weights,
                crossfit=5, random_state=seed, truth=pop.trait_mean,
            )
            acc["aipw"]["trait"].append((augmented.estimate - pop.trait_mean) / pop.trait_sd)
            progress.step()

        for name in (*METHODS, "aipw"):
            a = acc[name]
            rows.append(summarize(BENCHMARK, law, name, "trait_bias_sd", a["trait"],
                                  notes="signed error of the weighted trait mean, "
                                        "in population SD units; held out"))
            rows.append(Row(BENCHMARK, law, name, "trait_rmse_sd", rmse(a["trait"]),
                            None, None, len(a["trait"]),
                            "root mean square of the per-replication signed error"))
            if name == "aipw":
                continue
            rows.append(summarize(BENCHMARK, law, name, "unanchored_prevalence_error_pct",
                                  a["prev2"], notes="disease 2, whose register value "
                                                    "was never supplied"))
            rows.append(summarize(BENCHMARK, law, name, "worst_held_out_smd", a["smd"],
                                  notes="four covariate margins excluded from the solve"))
            rows.append(summarize(BENCHMARK, law, name, "kish_ess", a["ess"],
                                  notes="participants ~2500"))
            rows.append(summarize(BENCHMARK, law, name, "max_weight_share", a["share"],
                                  notes="largest weight in units of an equal share"))
            rows.append(summarize(BENCHMARK, law, name, "case_mix_liability_error", a["mix"],
                                  notes="mean liability among cases minus its "
                                        "population value; held out"))
    progress.close()
    return rows


def main() -> None:
    for row in run(n_reps=6):
        print(row.as_tsv())


if __name__ == "__main__":
    main()

"""B6 --- how rare can the anchored disease be before the machinery stops working?

The support axis of the validation matrix. A calibration target is reachable only if
the sample contains cases: an exponential tilt cannot move a weighted mean outside
the convex hull of the sampled values, and with no sampled case the hull does not
contain a positive prevalence. Between "plenty of cases" and "none" is a regime the
documentation warns about qualitatively -- the solve reports success, the weights
concentrate on a handful of people, and the bootstrap begins discarding replicates
from exactly the tail that would have widened the interval.

The direction of the disease channel decides which failure appears, so the sweep
runs both:

``over-recruited``   cases participate more than controls, as in a volunteer cohort
                     recruited through clinics. The sample is rich in cases, the
                     calibration *down*weights them, and rarity costs little.
``under-recruited``  cases participate less, which is the realistic direction for
                     severe psychiatric illness, dementia, or anything that makes
                     answering an invitation harder. Now the few sampled cases must
                     carry the whole population's worth of disease, weight
                     concentrates on them, and the cliff arrives early.

The two arms use the same population size, prevalence grid and participant count, so
the difference between their columns is the direction of recruitment and nothing
else. The population is 30,000 for this sweep alone -- the rarest levels need enough
population cases for a prevalence to mean anything -- so its ESS values are not
comparable with the other benchmarks'.
"""

from __future__ import annotations

import warnings

import numpy as np

from benchmarks import estimators as E
from benchmarks.harness import Progress, Row, proportion, rmse, summarize
from benchmarks.simulate import Design, simulate
from i3pw import entropy_balance

BENCHMARK = "B6_support"

POPULATION = 30_000
PREVALENCES = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
DIRECTIONS = {"over-recruited": 1.70, "under-recruited": -1.70}
METHOD = "ipw+cal"


def _bootstrap_discard(features, targets, base, *, n_boot: int, seed: int) -> float:
    """Share of resamples whose calibration cannot reach the target."""
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    failures = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, diag = entropy_balance(
                features[idx], targets, base_weights=base[idx],
                warn=False, return_diagnostics=True,
            )
        if not diag.converged or diag.max_abs_residual > 1e-6:
            failures += 1
    return failures / n_boot


def run(n_reps: int = 30, n_boot: int = 200) -> list[Row]:
    rows: list[Row] = []
    progress = Progress(BENCHMARK, len(DIRECTIONS) * len(PREVALENCES) * n_reps)
    for direction, delta_y in DIRECTIONS.items():
        for prevalence in PREVALENCES:
            condition = f"{direction} | K={prevalence:g}"
            cases, solved, discards = [], [], []
            trait, naive_trait, ess, share, residual = [], [], [], [], []
            for seed in range(n_reps):
                pop = simulate(Design(
                    seed=seed, population_size=POPULATION, prevalence=prevalence,
                    delta_x=0.65, delta_y=delta_y,
                ))
                cases.append(int(pop.Y[pop.mask, 0].sum()))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p_hat = E.propensity(pop)
                    wt = E.fit_weighting(pop, METHOD, p_hat=p_hat)
                naive_trait.append(E.trait_error(pop, E.fit_weighting(pop, "naive")))
                solved.append(wt.converged)
                if wt.converged:
                    trait.append(E.trait_error(pop, wt))
                    ess.append(wt.ess)
                    share.append(wt.max_share)
                    residual.append(wt.extra["max_abs_residual"])
                    discards.append(_bootstrap_discard(
                        wt.features, wt.targets, 1.0 / p_hat,
                        n_boot=n_boot, seed=20_000 + seed,
                    ))
                progress.step()

            rows.append(summarize(BENCHMARK, condition, METHOD, "sampled_cases", cases,
                                  notes="cases among ~2500 participants"))
            rows.append(proportion(BENCHMARK, condition, METHOD, "solve_success_rate",
                                   solved, notes="calibration reached the target"))
            rows.append(summarize(BENCHMARK, condition, METHOD, "bootstrap_discard_rate",
                                  discards, notes="resamples that could not reach the "
                                                  "target; conditional on the solve"))
            rows.append(summarize(BENCHMARK, condition, METHOD, "trait_bias_sd", trait,
                                  notes="conditional on the solve succeeding"))
            rows.append(Row(BENCHMARK, condition, METHOD, "trait_rmse_sd", rmse(trait),
                            None, None, len(trait), "conditional on the solve succeeding"))
            rows.append(summarize(BENCHMARK, condition, "naive", "trait_bias_sd",
                                  naive_trait,
                                  notes="the bias being corrected; it shrinks with K "
                                        "because a rarer disease tilts less"))
            rows.append(summarize(BENCHMARK, condition, METHOD, "kish_ess", ess,
                                  notes="population 30,000 for this sweep only"))
            rows.append(summarize(BENCHMARK, condition, METHOD, "max_weight_share", share,
                                  notes="largest weight in units of an equal share"))
            rows.append(summarize(BENCHMARK, condition, METHOD, "max_abs_residual",
                                  residual))
    progress.close()
    return rows


def main() -> None:
    for row in run(n_reps=4, n_boot=50):
        print(row.as_tsv())


if __name__ == "__main__":
    main()

"""B2 --- how much does each additional register quantity actually buy?

The second axis of the validation matrix. One recruitment law, held fixed, and a
ladder of increasingly informative register inputs. Recruitment here has three
outcome-side channels -- one per disease and one extra for people carrying both --
so each rung of the ladder has something left to correct:

    none              the covariate participation model alone.
    K(Y1)             the pooled prevalence of the first disease.
    K(Y1), K(Y2)      both prevalences.
    + co-occurrence   the joint P(Y1 = 1, Y2 = 1) as well.
    per stratum       K(Y1 | stratum) and the stratum shares instead of the pooled
                      margin.

The estimand stays the same across the ladder -- the population mean of a trait
observed only in participants -- so the rungs are directly comparable. This is the
question a cohort actually faces: the register can supply more, and each extra
constraint costs effective sample size, so the useful answer is which ones pay for
themselves.
"""

from __future__ import annotations

import warnings

import numpy as np

from benchmarks import estimators as E
from benchmarks.harness import Progress, Row, rmse, summarize
from benchmarks.simulate import Design, simulate
from i3pw import effective_sample_size, entropy_balance, stratified_calibration_weights

BENCHMARK = "B2_anchor_information"

LAW = dict(delta_x=0.65, delta_y=1.70, delta_comorbid=1.40)

LADDER = ("none", "K(Y1)", "K(Y1), K(Y2)", "+ co-occurrence", "per stratum")


def _weights_for(rung: str, pop, base: np.ndarray) -> E.Weighting:
    """Weights under one rung of the information ladder, sharing one base model."""
    mask = pop.mask
    Y = pop.Y[mask]
    prevalence = pop.population_prevalence
    if rung == "none":
        return E.Weighting(rung, base / base.sum())
    if rung == "per stratum":
        within = pop.within_stratum_prevalence()[:, :1]
        w, diag = stratified_calibration_weights(
            Y[:, :1], pop.stratum[mask], within, pop.stratum_share,
            base_weights=base, warn=False, return_diagnostics=True,
        )
        features = E.stratified_features(Y[:, :1], pop.stratum[mask], len(pop.stratum_share))
        return E.Weighting(rung, w, features=features, converged=bool(diag.converged))

    if rung == "K(Y1)":
        features, targets = Y[:, :1], prevalence[:1]
    elif rung == "K(Y1), K(Y2)":
        features, targets = Y, prevalence
    else:  # "+ co-occurrence"
        both = (pop.Y[:, 0] * pop.Y[:, 1]).astype(float)
        features = np.column_stack([Y, both[mask]])
        targets = np.concatenate([prevalence, [both.mean()]])
    w, diag = entropy_balance(
        features, targets, base_weights=base, warn=False, return_diagnostics=True,
    )
    return E.Weighting(rung, w, features=features, targets=targets,
                       converged=bool(diag.converged))


def run(n_reps: int = 40) -> list[Row]:
    rows: list[Row] = []
    progress = Progress(BENCHMARK, n_reps)
    acc: dict[str, dict[str, list[float]]] = {
        rung: {k: [] for k in ("trait", "ess", "share", "smd", "comorbid")}
        for rung in (*LADDER, "oracle")
    }
    for seed in range(n_reps):
        pop = simulate(Design(seed=seed, **LAW))
        truth_comorbid = float((pop.Y[:, 0] * pop.Y[:, 1]).mean())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = 1.0 / E.propensity(pop)
            weightings = {rung: _weights_for(rung, pop, base) for rung in LADDER}
        oracle = E.fit_weighting(pop, "oracle")
        weightings["oracle"] = oracle
        for rung, wt in weightings.items():
            a = acc[rung]
            a["trait"].append(E.trait_error(pop, wt))
            a["ess"].append(effective_sample_size(wt.weights))
            a["share"].append(wt.max_share)
            a["smd"].append(E.held_out_balance(pop, wt).worst_held_out)
            both = (pop.Y[pop.mask, 0] * pop.Y[pop.mask, 1]).astype(float)
            a["comorbid"].append(E.weighted_mean(both, wt.weights) - truth_comorbid)
        progress.step()
    progress.close()

    for rung in (*LADDER, "oracle"):
        a = acc[rung]
        rows.append(summarize(BENCHMARK, rung, "ipw+cal", "trait_bias_sd", a["trait"],
                              notes="held-out trait mean, population SD units"))
        rows.append(Row(BENCHMARK, rung, "ipw+cal", "trait_rmse_sd", rmse(a["trait"]),
                        None, None, len(a["trait"]),
                        "root mean square of the per-replication signed error"))
        rows.append(summarize(BENCHMARK, rung, "ipw+cal", "kish_ess", a["ess"],
                              notes="cost of each extra constraint"))
        rows.append(summarize(BENCHMARK, rung, "ipw+cal", "max_weight_share", a["share"]))
        rows.append(summarize(BENCHMARK, rung, "ipw+cal", "worst_held_out_smd", a["smd"],
                              notes="four covariate margins excluded from every rung"))
        note = ("constrained at this rung: an identity, not a finding"
                if rung == "+ co-occurrence" else "held out at this rung")
        rows.append(summarize(BENCHMARK, rung, "ipw+cal", "comorbidity_error", a["comorbid"],
                              notes=note))
    return rows


def main() -> None:
    for row in run(n_reps=6):
        print(row.as_tsv())


if __name__ == "__main__":
    main()

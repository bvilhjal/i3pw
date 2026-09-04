"""B8 --- what register error costs when the anchor is not the simplest one.

B3 sweeps a mis-stated prevalence through one recruitment law and the simplest
anchor (the pooled margin), and B2/B4 compare anchors -- pooled, per stratum, by
severity -- at a *correct* register. Neither establishes what the identification
condition's practical caveat actually requires: whether a wrong register reverses
the anchor recommendations, or whether richer anchors import more damage per unit
of register error. Every quantity a stratified or severity calibration constrains
is itself a register estimate, so the question is not academic: if per-stratum
prevalences carry more error than the pooled figure, the B2 ladder's top rung and
the B4 repairs could lose what they gain.

The register error model is the same common relative error as
:func:`i3pw.prevalence_sensitivity`: every prevalence-type target is scaled by
``1 + delta`` and demographic shares are left alone. That is one hypothesis about
how registers are wrong (a drifted diagnostic threshold), not a distribution over
them; the sweep is read as a bound, in the same way B3 is.

Seven scenarios cross the anchor with the recruitment law:

    pooled / X + Y                     B3's setting, reproduced as the continuity check.
    stratified / X + Y                 a richer anchor under a law that does not need it.
    pooled / X x Y                     the tilt family cannot represent the law at all:
                                       the identification condition fails structurally.
    pooled / stratum-differential      B4's first wrong-case-mix channel, unrepaired.
    stratified / stratum-differential  its repair, now with a wrong register.
    pooled / severity                  B4's second channel, unrepaired.
    severity / severity                its repair (mild/severe anchors), wrong register.

The held-out trait mean is the scored estimand throughout; the unanchored second
disease and the effective sample size are recorded beside it. Every cell carries
the across-replication SD and the Monte Carlo SE, so the curves can be read with
confidence intervals rather than as point wins: the figure draws the 95% MC
interval of the mean and a +-1 SD envelope, and the table prints the interval.
"""

from __future__ import annotations

import warnings

from benchmarks import estimators as E
from benchmarks.harness import Progress, Row, rmse, summarize
from benchmarks.simulate import Design, simulate

BENCHMARK = "B8_k_error"

DELTAS = (-0.20, -0.10, 0.0, 0.10, 0.20)
"""Relative error on every prevalence-type target. Half B3's range: the point is
the crossing with the anchor scenarios, not the extremes, and seven scenarios
times five deltas is already 35 cells per replication."""

LAWS: dict[str, dict[str, float]] = {
    "X + Y": dict(delta_x=0.65, delta_y=1.70),
    "X x Y": dict(delta_x=0.65, delta_y=1.70, delta_xy=1.00),
    "stratum-differential": dict(delta_x=0.65, delta_y=1.70, delta_y_stratum=1.30),
    "severity": dict(delta_x=0.65, delta_y=1.70, delta_sev=0.90),
}
"""The laws B1--B4 already name, so a law means the same thing here as there."""

ANCHOR_METHOD = {
    "pooled": "ipw+cal",
    "stratified": "ipw+cal/s",
    "severity": "ipw+cal/v",
}

SCENARIOS = (
    ("pooled / X + Y", "pooled", "X + Y"),
    ("stratified / X + Y", "stratified", "X + Y"),
    ("pooled / X x Y", "pooled", "X x Y"),
    ("pooled / stratum-differential", "pooled", "stratum-differential"),
    ("stratified / stratum-differential", "stratified", "stratum-differential"),
    ("pooled / severity", "pooled", "severity"),
    ("severity / severity", "severity", "severity"),
)


def run(n_reps: int = 40) -> list[Row]:
    rows: list[Row] = []
    progress = Progress(BENCHMARK, n_reps * len(LAWS))
    keys = ("trait", "prev2", "ess")
    acc: dict[str, dict[str, list[float]]] = {
        f"{name} | delta={d:+.2f}": {k: [] for k in keys}
        for name, _, _ in SCENARIOS for d in DELTAS
    }
    reference: dict[str, dict[str, list[float]]] = {
        law: {"naive": [], "ipw": [], "oracle": []} for law in LAWS
    }

    for seed in range(n_reps):
        for law, channels in LAWS.items():
            pop = simulate(Design(seed=seed, **channels))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_hat = E.propensity(pop)
                for name in reference[law]:
                    wt = E.fit_weighting(pop, name, p_hat=p_hat)
                    reference[law][name].append(E.trait_error(pop, wt))
                for label, anchor, law_of in SCENARIOS:
                    if law_of != law:
                        continue
                    for d in DELTAS:
                        wt = E.fit_weighting(
                            pop, ANCHOR_METHOD[anchor], p_hat=p_hat,
                            target_scale=1.0 + d,
                        )
                        a = acc[f"{label} | delta={d:+.2f}"]
                        a["trait"].append(E.trait_error(pop, wt))
                        a["prev2"].append(
                            E.unanchored_prevalence_error(pop, wt, outcome=1))
                        a["ess"].append(wt.ess)
            progress.step()
    progress.close()

    for label, _, _ in SCENARIOS:
        for d in DELTAS:
            condition = f"{label} | delta={d:+.2f}"
            a = acc[condition]
            rows.append(summarize(BENCHMARK, condition, "ipw+cal", "trait_bias_sd",
                                  a["trait"],
                                  notes="held-out trait mean; sd and mcse give the "
                                        "SD envelope and the 95% MC interval"))
            rows.append(Row(BENCHMARK, condition, "ipw+cal", "trait_rmse_sd",
                            rmse(a["trait"]), None, None, len(a["trait"]),
                            "root mean square of the per-replication signed error"))
            rows.append(summarize(BENCHMARK, condition, "ipw+cal",
                                  "unanchored_prevalence_error_pct", a["prev2"],
                                  notes="disease 2, never supplied"))
            rows.append(summarize(BENCHMARK, condition, "ipw+cal", "kish_ess", a["ess"]))

    for law, values in reference.items():
        for name, vals in values.items():
            rows.append(summarize(BENCHMARK, f"reference | {law}", name,
                                  "trait_bias_sd", vals,
                                  notes="does not depend on delta; the scale the "
                                        "curves sit on"))
    return rows


def main() -> None:
    for row in run(n_reps=6):
        print(row.as_tsv())


if __name__ == "__main__":
    main()

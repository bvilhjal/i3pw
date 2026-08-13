"""B7 --- the ridge, and whether relaxing an exact constraint ever pays.

``calibrate(shrinkage=...)`` trades exact calibration for less extreme weights. The
package documents the mechanism and leaves the number to the user, which is
unhelpful advice: nothing tells you what a given ridge costs in bias or buys in
variance, and the natural guess -- that shrinkage is a pure concession -- is only
true when the tilt family is right.

The sweep runs the ridge across four orders of magnitude under three arms, chosen so
that the answer cannot be read off one of them alone:

``correct``       outcome-only recruitment, uniform base, calibrated on the outcome.
                  The tilt family contains the truth, so the exact solve is
                  consistent and every unit of ridge should be a pure concession.
``base + margin`` additive covariate and outcome channels with a fitted base: the
                  ordinary applied situation, mildly misspecified.
``misspecified``  a covariate-by-outcome interaction the tilt family cannot
                  represent at all.

and reports the whole decomposition -- absolute bias, across-replication SD, their
combination as RMSE, and the effective sample size and residual the ridge is buying.
Under a correct specification the exact solve should sit at or near the RMSE
minimum. Under misspecification the exact solve is chasing a constraint that points
the wrong way, and there is no reason for the minimum to sit at zero.
"""

from __future__ import annotations

import warnings

import numpy as np

from benchmarks import estimators as E
from benchmarks.harness import Progress, Row, rmse, summarize
from benchmarks.simulate import SELECTION_LAWS, Design, simulate

BENCHMARK = "B7_shrinkage"

ARMS = {
    "correct": dict(law="Y only", method="cal"),
    "base + margin": dict(law="X + Y", method="ipw+cal"),
    "misspecified": dict(law="X x Y", method="ipw+cal"),
}
RIDGES = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)


def run(n_reps: int = 40) -> list[Row]:
    rows: list[Row] = []
    progress = Progress(BENCHMARK, len(ARMS) * n_reps)
    for arm, spec in ARMS.items():
        law, method = spec["law"], spec["method"]
        acc: dict[float, dict[str, list[float]]] = {
            r: {k: [] for k in ("trait", "ess", "share", "residual")} for r in RIDGES
        }
        for seed in range(n_reps):
            pop = simulate(Design(seed=seed, **SELECTION_LAWS[law]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_hat = E.propensity(pop) if method.startswith("ipw") else None
                for ridge in RIDGES:
                    wt = E.fit_weighting(pop, method, p_hat=p_hat, shrinkage=ridge)
                    a = acc[ridge]
                    a["trait"].append(E.trait_error(pop, wt))
                    a["ess"].append(wt.ess)
                    a["share"].append(wt.max_share)
                    a["residual"].append(wt.extra["max_abs_residual"])
            progress.step()

        for ridge in RIDGES:
            a = acc[ridge]
            condition = f"{arm} | ridge={ridge:g}"
            values = np.asarray(a["trait"], dtype=float)
            rows.append(summarize(BENCHMARK, condition, method, "trait_bias_sd", a["trait"],
                                  notes="mean is the bias; sd is the across-replication "
                                        "spread the ridge is buying"))
            rows.append(Row(BENCHMARK, condition, method, "trait_abs_bias_sd",
                            float(abs(values.mean())), None, None, values.size,
                            "absolute value of the mean signed error"))
            rows.append(Row(BENCHMARK, condition, method, "trait_rmse_sd", rmse(a["trait"]),
                            None, None, values.size,
                            "root mean square of the per-replication signed error"))
            rows.append(summarize(BENCHMARK, condition, method, "kish_ess", a["ess"]))
            rows.append(summarize(BENCHMARK, condition, method, "max_weight_share",
                                  a["share"]))
            rows.append(summarize(BENCHMARK, condition, method, "max_abs_residual",
                                  a["residual"],
                                  notes="non-zero by design once the ridge is positive"))
    progress.close()
    return rows


def main() -> None:
    for row in run(n_reps=6):
        print(row.as_tsv())


if __name__ == "__main__":
    main()

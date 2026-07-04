"""Monte Carlo comparison of the correction methods across many populations.

Repeats simulate -> bias -> correct over ``N_REPS`` random draws and reports each
method's mean absolute percentage error (+/- SD). It runs the comparison under
both weighting schemes to make an important point explicit:

- ``odds``    -- the R construction. Its weighted mean runs over the *whole* test
                 set, so unselected units contribute their (in reality
                 unobservable) outcomes. It looks great, but is not deployable.
- ``inverse`` -- the textbook Horvitz-Thompson estimator over the *sample only*
                 (the realistic case). It reveals how much correction is actually
                 achievable when selection is outcome-driven and X is a proxy.

    python examples/monte_carlo.py
"""

from __future__ import annotations

import time

import i3pw

N_REPS = 20

SIM = dict(
    population_size=5000,
    n_features=12,
    n_outcomes=2,
    predictors_per_outcome=6,
    target_population_prevalence=(0.4, 0.08),
    target_sample_prevalence=(0.2, 0.01),
    sample_size=1200,
)

PENALIZED = dict(
    lambdas=(0.001, 0.01),
    gammas=(0.0, 1.0),
    K=3,
    learning_rate=0.05,
    max_iter=3000,
    decay_interval=1000,
)


def main() -> None:
    i3pw.warmup()
    t0 = time.time()
    for weighting in ("odds", "inverse"):
        summaries = i3pw.monte_carlo(
            n_reps=N_REPS,
            sim_kwargs=SIM,
            penalized_kwargs=PENALIZED,
            weighting=weighting,
        )
        label = (
            "oracle (reads unselected outcomes)"
            if weighting == "odds"
            else "deployable (sample only)"
        )
        print(f"\n=== weighting = {weighting!r}  [{label}]  ({N_REPS} reps) ===")
        print(i3pw.format_summary(summaries))
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

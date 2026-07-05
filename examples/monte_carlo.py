"""Monte Carlo comparison of the correction methods across many populations.

Repeats simulate -> bias -> correct over ``N_REPS`` random draws and reports each
method's mean absolute percentage error (+/- SD), using the *deployable*
sample-only estimator throughout. The point it makes:

- ``no_correction`` and ``lasso_ipw`` (the covariate-only participation model)
  barely dent the bias, because participation here is driven by the outcomes,
  which the covariates only weakly proxy.
- ``calibration_ipw`` uses the known population prevalences directly, so it
  reproduces them (the anchored outcomes) essentially exactly. That is the
  whole idea: the known prevalences carry the information the covariate model
  cannot.

Also prints the Kish effective sample size for the calibration weights — strong
ascertainment concentrates weight on few units, the price of the correction.

    python examples/monte_carlo.py
"""

from __future__ import annotations

import time

import numpy as np

import i3pw

N_REPS = 20

SIM = dict(
    population_size=5000,
    n_features=12,
    n_outcomes=2,
    predictors_per_outcome=6,
    target_population_prevalence=(0.4, 0.15),
    target_sample_prevalence=(0.2, 0.03),
    sample_size=1200,
)


def main() -> None:
    t0 = time.time()

    summaries = i3pw.monte_carlo(
        n_reps=N_REPS,
        sim_kwargs=SIM,
        weighting="inverse",  # deployable, sample-only
    )
    print(f"=== mean absolute % error over {N_REPS} reps (deployable estimator) ===")
    print(i3pw.format_summary(summaries))

    # Effective sample size of the calibration weights.
    ess = []
    for rep in range(N_REPS):
        ds = i3pw.make_dataset(seed=rep, **SIM)
        ess.append(i3pw.calibration_ipw(ds, base="lasso").ess)
    print(f"\ncalibration_ipw Kish ESS: {np.mean(ess):.0f} ± {np.std(ess):.0f}")
    print("(the variance cost of concentrating weight on the ascertained units)")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

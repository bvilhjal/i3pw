"""Compare selection-bias corrections against the known population prevalence.

Simulates a population with outcome-dependent selection bias, then compares:

    * no correction    (naive sample prevalence)
    * LASSO IPW        (covariate-only inclusion model — the approach that fails)
    * calibration IPW  (the recommended estimator: calibrate to known prevalences)

Kept deliberately light so it runs in a few seconds; scale ``population_size`` up
for a serious comparison.

    python examples/benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

import i3pw


def main() -> None:
    t_start = time.time()

    # Two-outcome scenario (one common, one rare).
    ds = i3pw.make_dataset(
        seed=97,
        population_size=8000,
        n_features=15,
        n_outcomes=2,
        predictors_per_outcome=8,
        target_population_prevalence=(0.4, 0.05),
        target_sample_prevalence=(0.2, 0.005),
        sample_size=2000,
    )

    pop = ds.population_prevalence
    print("=" * 64)
    print("Population prevalence :", np.round(pop, 4))
    print("Biased sample prev.   :", np.round(ds.sample_prevalence, 4))
    print("=" * 64)

    rows: list[tuple[str, np.ndarray]] = []
    rows.append(("no_correction", i3pw.no_correction(ds).percent_diff))
    rows.append(("lasso_ipw", i3pw.lasso_ipw(ds, cv=5).percent_diff))
    rows.append(("calibration_ipw", i3pw.calibration_ipw(ds, base="lasso").percent_diff))

    print(f"{'method':<28}{'% diff Y1':>12}{'% diff Y2':>12}")
    print("-" * 52)
    for name, pdiff in rows:
        print(f"{name:<28}{pdiff[0]:>12.2f}{pdiff[1]:>12.2f}")
    print("-" * 52)
    print("(lower percentage difference from the population prevalence is better)")
    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

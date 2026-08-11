"""What calibration actually buys you — measured on quantities it was not handed.

``examples/benchmark.py`` anchors every outcome, so ``calibration_ipw`` reports 0.00%
error. That number is an algebraic identity: the estimator reproduces the prevalences it
was given. It says nothing about whether the method works.

This script measures the three things that *can* fail:

1. **Transfer.** Anchor one outcome, evaluate the other. Does calibrating on a disease
   whose prevalence you know help with one you do not?
2. **Balance (a held-out diagnostic).** Check the reweighted sample against population
   covariate means that were never used as constraints. Constrained moments match by
   construction; held-out ones do not have to, so only they can refute the weighting.
3. **Does the base model earn its place?** ``base="lasso"`` is the default. On a
   simulation with no covariate selection channel it has no signal to fit and only adds
   noise — so the comparison is run both with and without that channel, which is what
   ``SimConfig.selection_covariate_strength`` exists for.

    python examples/honest_benchmark.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np

import i3pw

N_REPS = 12
SIM = dict(
    population_size=5000,
    n_features=12,
    n_outcomes=2,
    predictors_per_outcome=6,
    target_population_prevalence=(0.4, 0.15),
    target_sample_prevalence=(0.2, 0.03),
    sample_size=1200,
)


def _run(strength: float) -> dict[str, list[float]]:
    """Per-rep transfer error, held-out balance and covariate-mean error, by base."""
    out: dict[str, list[float]] = {
        "naive_free": [], "lasso_free": [], "uniform_free": [],
        "lasso_smd": [], "uniform_smd": [],
        "lasso_covmean": [], "uniform_covmean": [],
    }
    for rep in range(N_REPS):
        ds = i3pw.make_dataset(seed=rep, selection_covariate_strength=strength, **SIM)
        X_test, Y_test, s_test = ds.split("test")
        sel = s_test == 1
        pop = ds.population_prevalence

        # Naive: the unanchored outcome's error with no correction at all.
        out["naive_free"].append(
            abs(Y_test[sel][:, 1].mean() - pop[1]) / pop[1] * 100.0
        )

        for base in ("lasso", "uniform"):
            # Anchor outcome 0 only; outcome 1 is held out -> a real test.
            res = i3pw.calibration_ipw(ds, anchor_outcomes=[0], base=base)
            out[f"{base}_free"].append(res.percent_diff[1])

            w = res.extra["weight"][sel]
            # Held-out population covariate means: never used as constraints.
            cols = [0, 1, 2]
            rep_bal = i3pw.balance_report(
                np.column_stack([Y_test[sel][:, 0], X_test[sel][:, cols]]),
                w,
                np.concatenate([[pop[0]], X_test[:, cols].mean(axis=0)]),
                constrained=[True, False, False, False],
                names=["Y1 (anchored)", "X0", "X1", "X2"],
            )
            out[f"{base}_smd"].append(rep_bal.worst_held_out)
            truth = X_test[:, 0].mean()
            got = float((w * X_test[sel][:, 0]).sum() / w.sum())
            out[f"{base}_covmean"].append(abs(got - truth))
    return out


def _fmt(v: list[float]) -> str:
    return f"{np.mean(v):7.3f} ± {np.std(v):5.3f}"


def main() -> None:
    t0 = time.time()
    warnings.simplefilter("ignore")

    for strength, label in ((0.0, "no covariate channel (package default)"),
                            (1.5, "covariate channel ON")):
        res = _run(strength)
        print("=" * 78)
        print(f"selection_covariate_strength = {strength}   ({label})")
        print("=" * 78)
        print(f"{'quantity':<44}{'lasso base':>17}{'uniform base':>17}")
        print("-" * 78)
        print(f"{'% err, UNANCHORED outcome (transfer)':<44}"
              f"{_fmt(res['lasso_free']):>17}{_fmt(res['uniform_free']):>17}")
        print(f"{'  ...vs no correction at all':<44}{_fmt(res['naive_free']):>17}")
        print(f"{'worst held-out |SMD| (diagnostic)':<44}"
              f"{_fmt(res['lasso_smd']):>17}{_fmt(res['uniform_smd']):>17}")
        print(f"{'|error| on held-out covariate mean':<44}"
              f"{_fmt(res['lasso_covmean']):>17}{_fmt(res['uniform_covmean']):>17}")
        print()

    print("Read this table, not the 0.00% one:")
    print("  * The anchored outcome is exact by construction and is not shown.")
    print("  * Transfer to an UNANCHORED outcome is the honest question. Calibrating on")
    print("    one disease does little for another whose prevalence you never supplied —")
    print("    marginal calibration fixes marginal quantities, nothing more.")
    print("  * With no covariate channel the LASSO base has nothing to learn, so it adds")
    print("    estimation noise; turn the channel on and it starts to earn its place.")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

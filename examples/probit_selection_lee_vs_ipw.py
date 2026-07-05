"""Benchmark: Lee et al. (2011) liability transform vs IPW under selection bias.

A probit / liability-threshold model: latent Gaussian liability ``L = f(X) + e``,
binary outcome ``Y = 1[L > t]``, population prevalence ``K``. The estimand is the
liability-scale variance explained ``R2_L = Var(f)/Var(L)``. The sample is
ascertained on ``Y`` — cases over-represented — so the sample case fraction ``P``
differs from ``K`` (outcome-dependent selection).

Three estimators of ``R2_L`` from the ascertained sample are compared to the truth:

- ``naive`` — observed-scale moment estimate + population transform only.
- ``lee``   — observed-scale moment estimate + the full Lee et al. transform.
- ``ipw``   — reweight the case fraction back to ``K``, weighted moment estimate,
              population transform (the design-based route).

Expected picture: ``naive`` is badly biased; ``lee`` and ``ipw`` agree under mild
ascertainment; as ascertainment strengthens (``P`` far from ``K``, rarer ``K``)
the Lee linearization drifts while IPW tracks the truth — at some variance cost.

    python examples/probit_selection_lee_vs_ipw.py
"""

from __future__ import annotations

import time

import numpy as np

from i3pw.liability import estimate_liability_r2, simulate_case_control

N_PREDICTORS = 250
N_SAMPLE = 2000
N_REPS = 25
R2S = (0.5, 0.8)
KS = (0.01, 0.05, 0.10)
PS = (0.5, 0.2)
METHODS = ("naive", "lee", "ipw")


def run_cell(r2: float, K: float, P: float):
    n_cases = int(round(P * N_SAMPLE))
    n_controls = N_SAMPLE - n_cases
    ests = {m: [] for m in METHODS}
    truth = []
    for rep in range(N_REPS):
        seed = 10_000 * int(r2 * 100) + 1000 * int(K * 1000) + 100 * int(P * 100) + rep
        smp = simulate_case_control(
            n_cases, n_controls, N_PREDICTORS, r2, K, np.random.default_rng(seed)
        )
        truth.append(smp.true_r2)
        for m in METHODS:
            ests[m].append(estimate_liability_r2(smp, K, m))
    return np.mean(truth), {m: np.array(v) for m, v in ests.items()}


def main():
    t0 = time.time()
    print(f"Liability-scale variance explained R2_L; {N_REPS} reps, "
          f"{N_PREDICTORS} predictors, n={N_SAMPLE}")
    header = f"{'K':>6}{'P':>6}{'truth':>9}" + "".join(f"{m:>16}" for m in METHODS)
    for r2 in R2S:
        print(f"\n--- true R2_L ~= {r2} ---")
        print(header)
        print("-" * len(header))
        for K in KS:
            for P in PS:
                truth, ests = run_cell(r2, K, P)
                cells = "".join(f"{ests[m].mean():>9.3f}±{ests[m].std():<6.3f}" for m in METHODS)
                print(f"{K:>6}{P:>6}{truth:>9.3f}{cells}")
    print("\n(entries are mean ± SD of the R2_L estimate over reps)")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""Doubly-robust recovery of a participant-only trait under ascertainment.

A binary outcome ``Y`` is ascertained — cases are over-represented relative to the
known population prevalence ``K``. A separate trait ``V`` (a biomarker, say) is
measured *only on participants* and correlates with the outcome's liability, so the
ascertained sample's mean ``V`` is inflated. The trait is missing at random given
the covariates (``S ⊥ V | X``), so the population mean ``E[V]`` is recoverable.

Goal: recover ``E[V]``. Compared:

- ``naive``        — sample mean of V (biased by ascertainment).
- ``ipw_lasso``    — reweight by a covariate participation model (``1/P̂``).
- ``calibration``  — reweight so the sample matches the known prevalence ``K``.
- ``aipw``         — doubly-robust: an outcome model for V plus the calibration weights.

The doubly-robust estimator (:func:`i3pw.aipw_mean`) is consistent if *either* the
outcome model or the weights are correct, and is the lower-variance choice for a
downstream mean. Using calibration weights — which are tuned to the *ascertained
margin* — as a raw weighted mean for an unrelated trait is comparatively noisy.

    python examples/doubly_robust_trait.py
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import Ridge

import i3pw

N_REPS = 20
SIM = dict(
    population_size=6000,
    n_features=12,
    n_outcomes=2,
    predictors_per_outcome=6,
    target_population_prevalence=(0.15, 0.4),  # outcome 0 = the ascertained outcome, K=0.15
    target_sample_prevalence=(0.03, 0.2),
    sample_size=1500,
)


def hajek(weights, values):
    return float(np.sum(weights * values) / np.sum(weights))


def one_rep(seed):
    ds = i3pw.make_dataset(seed=seed, **SIM)
    X_tr, _, s_tr = ds.split("train")
    X_te, _, s_te = ds.split("test")
    sel = s_te == 1

    # Participant-only trait V: the outcome's liability (a function of X) + independent
    # noise. Noise ⟂ selection, so V is missing at random given X.
    rng = np.random.default_rng(1000 + seed)
    liability = X_te @ ds.coefficients[0]
    liability = (liability - liability.mean()) / liability.std()
    V = liability + rng.normal(size=liability.shape[0])
    truth = V.mean()  # population mean over the test fold

    # Weights.
    P_sel = i3pw.lasso_propensity(X_tr, s_tr, X_te[sel])
    w_lasso = 1.0 / P_sel
    cal = i3pw.calibration_ipw(ds, anchor_outcomes=[0], base="lasso")
    w_cal = cal.extra["weight"][sel]

    est = {
        "naive": V[sel].mean(),
        "ipw_lasso": hajek(w_lasso, V[sel]),
        "calibration": hajek(w_cal, V[sel]),
        "aipw": i3pw.aipw_mean(X_te, sel, V[sel], w_cal, outcome_model=Ridge(1.0)).estimate,
    }
    return {k: v - truth for k, v in est.items()}, cal.ess


def main():
    t0 = time.time()
    rows = [one_rep(s) for s in range(N_REPS)]
    biases = {k: np.array([r[0][k] for r in rows]) for k in rows[0][0]}
    ess = np.mean([r[1] for r in rows])

    print(f"=== recover population mean of a participant-only trait ({N_REPS} reps) ===")
    print(f"{'method':<14}{'mean bias':>12}{'|bias|':>10}{'SD':>10}")
    print("-" * 46)
    for k, b in biases.items():
        print(f"{k:<14}{b.mean():>12.4f}{np.abs(b).mean():>10.4f}{b.std():>10.4f}")
    print(f"\ncalibration Kish ESS ~ {ess:.0f}")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

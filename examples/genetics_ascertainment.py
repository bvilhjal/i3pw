"""Genetics-flavoured ascertainment demo: recovering a participant-only trait.

Scenario (a case-cohort / ascertained cohort in statistical genetics):

- A disease ``Y`` is ascertained — cases are over-represented relative to the
  population prevalence ``K``, which is known from a registry.
- A trait ``V`` (think polygenic score / biomarker) is measured *only on
  participants*. Because ``V`` correlates with disease liability, the ascertained
  sample's mean ``V`` is inflated — a real phenomenon (participation is heritable;
  cases carry higher genetic liability).

Goal: recover the population mean ``E[V]``. We compare:

- ``naive``        — sample mean of V (biased by ascertainment).
- ``ipw_lasso``    — reweight by a covariate participation model (1/P).
- ``calibration``  — reweight so the sample matches the known disease prevalence K.
- ``aipw``         — doubly-robust: an outcome model for V plus the calibration
                     weights.

Notes on the theory this mirrors: the known ``K`` playing the correcting role is
the same anchor used in the observed->liability heritability transform
(Lee, Wray, Goddard & Visscher 2011); pure case-control ascertainment leaves
*logistic slopes* unbiased (Prentice & Pyke 1979) but biases means, absolute
risks, and — as here — the mean of a liability-correlated trait.

    python examples/genetics_ascertainment.py
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
    target_population_prevalence=(0.15, 0.4),  # outcome 0 = the ascertained disease, K=0.15
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

    # Participant-only trait V: disease liability (function of X) + independent noise.
    rng = np.random.default_rng(1000 + seed)
    liability = X_te @ ds.coefficients[0]
    liability = (liability - liability.mean()) / liability.std()
    V = liability + rng.normal(size=liability.shape[0])  # MAR given X (noise ⟂ selection)
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
    i3pw.warmup()
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

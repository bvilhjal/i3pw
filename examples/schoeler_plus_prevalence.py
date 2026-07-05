"""Modifying a Schoeler-style participation weight to leverage known prevalences.

A covariate participation model (Schoeler et al. 2023; van Alten et al. 2024)
weights the UK Biobank by 1/P̂(S | X_socio). It corrects the sociodemographic
(healthy-volunteer) tilt but is blind to selection that depends on the disorder
itself. Here selection depends on *both*:

    logit P(S | X_socio, Y) = α + X_socio·c + θ·Y

The modification: use the Schoeler weights as a *base*, then calibrate (rake)
them to the known population prevalence K of the disorder. Equivalently, this
adds a θ·Y term to the log-participation model whose coefficient is identified by
the known prevalence — the calibration-for-nonignorable-nonresponse construction
of Kott & Chang (2010).

Estimand: the liability-scale variance explained R²_L of the disorder (Lee et al.
2011 territory), estimated by a weighted Haseman–Elston moment regression plus
the observed→liability transform.

    naive     — no correction.
    schoeler  — 1/P̂(S | X_socio) only.
    modified  — Schoeler base weights raked to the known prevalence K.
    oracle    — 1/P(S | X_socio, Y), the true probabilities.

Result: Schoeler-alone leaves the disease ascertainment uncorrected (R²_L badly
biased); the modified weights recover the truth. Marginal-prevalence calibration
suffices here because the estimand is a variance component, not an effect size —
effect-size (collider) bias would additionally require covariate-stratified or
joint prevalences.

    python examples/schoeler_plus_prevalence.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from i3pw import entropy_balance, liability_r2_from_weights

N_POP = 90_000
N_PRED = 120        # weak liability predictors
N_COV = 6           # sociodemographic covariates
R2 = 0.5
K = 0.10
THETA = 1.5         # disease-driven participation (cases over-recruited)
ALPHA = -3.7        # base participation ~ few %
N_REPS = 6
METHODS = ("naive", "schoeler", "modified", "oracle")


def one_rep(seed):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((N_POP, N_PRED))
    beta = rng.normal(0, np.sqrt(R2 / N_PRED), N_PRED)
    signal = G @ beta
    L = signal + rng.normal(0, np.sqrt(1 - R2), N_POP)
    truth = signal.var() / L.var()
    Y = (L > norm.ppf(1 - K)).astype(float)
    Kpop = Y.mean()

    Xs = rng.standard_normal((N_POP, N_COV))      # socio covariates, independent of G
    c = rng.normal(0, 0.6, N_COV)
    pi = expit(ALPHA + Xs @ c + THETA * Y)
    S = rng.uniform(size=N_POP) < pi

    Gs, ys = G[S], Y[S]
    # Schoeler-style covariate participation model (sees only X_socio).
    clf = LogisticRegression(max_iter=300).fit(Xs, S.astype(int))
    w_scho = 1.0 / np.clip(clf.predict_proba(Xs[S])[:, 1], 1e-4, 1 - 1e-4)
    # Modification: rake the Schoeler weights to the known prevalence.
    w_mod = entropy_balance(ys.reshape(-1, 1), [Kpop], base_weights=w_scho)

    weights = {
        "naive": np.ones_like(ys),
        "schoeler": w_scho,
        "modified": w_mod,
        "oracle": 1.0 / pi[S],
    }
    est = {m: liability_r2_from_weights(Gs, ys, Kpop, w) for m, w in weights.items()}
    return truth, float(ys.mean()), est


def main():
    t0 = time.time()
    acc = {m: [] for m in METHODS}
    truth, pfrac = [], []
    for rep in range(N_REPS):
        tr, p, est = one_rep(700 + rep)
        truth.append(tr)
        pfrac.append(p)
        for m in METHODS:
            acc[m].append(est[m])
    print(f"true R2_L ~ {np.mean(truth):.3f}; participant case fraction ~ {np.mean(pfrac):.2f} "
          f"(K={K}); {N_REPS} reps\n")
    print(f"{'method':<12}{'R2_L':>10}{'sd':>8}")
    print("-" * 30)
    for m in METHODS:
        a = np.array(acc[m])
        print(f"{m:<12}{a.mean():>10.3f}{a.std():>8.3f}")
    print("\nschoeler (covariate IPW) leaves the disease ascertainment uncorrected;\n"
          "raking those weights to the known prevalence recovers the truth.")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""Optimal weights for several case-control outcomes with known prevalences.

Two correlated binary outcomes (`Y1`, `Y2`) with known population prevalences,
assembled into one sample by *multiplicative* case-control selection:

    P(S=1 | Y1, Y2) ∝ base · f1^Y1 · f2^Y2 · g^(Y1·Y2)

``g = 1`` means the two outcomes drive selection independently (log-linear /
additive on the log scale); ``g > 1`` means comorbid cases (both outcomes) are
over-recruited — an *interaction* in the selection.

We compare weighting strategies by how well the reweighted sample recovers two
population quantities that are *not* directly calibrated: an additive target
``E[L1 + L2]`` and a joint target ``E[L1·L2]`` (``L`` = latent liabilities).

- ``mean`` / ``product`` — per-outcome case/control weights, combined heuristically.
- ``calib_marginal`` — joint entropy balancing to the two known marginals.
- ``calib_joint``    — plus the known co-occurrence ``P(Y1=1, Y2=1)``.
- ``oracle``         — inverse of the true selection probability.

Expected: joint calibration beats the heuristic combines; marginal calibration is
exact when ``g = 1`` (even for the joint target); once ``g > 1`` it is biased on the
joint target, and only ``calib_joint`` (with the co-occurrence constraint) recovers
the oracle. The optimum is: calibrate to every population moment the selection needs.

    python examples/multi_outcome_calibration.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm

from i3pw import entropy_balance, outcome_calibration_weights

N_POP = 200_000
N_REPS = 10
RHO = 0.5
K1, K2 = 0.15, 0.08
BASE, F1, F2 = 0.006, 7.0, 8.0
G_VALUES = (1.0, 2.5)
METHODS = ("naive", "mean", "product", "calib_marginal", "calib_joint", "oracle")


def hajek(w, z):
    return float(np.sum(w * z) / np.sum(w))


def one_rep(g, seed):
    rng = np.random.default_rng(seed)
    L = rng.multivariate_normal([0, 0], [[1, RHO], [RHO, 1]], size=N_POP)
    t1, t2 = norm.ppf(1 - K1), norm.ppf(1 - K2)
    Y1 = (L[:, 0] > t1).astype(float)
    Y2 = (L[:, 1] > t2).astype(float)
    z_add, z_joint = L[:, 0] + L[:, 1], L[:, 0] * L[:, 1]
    truth = {"add": z_add.mean(), "joint": z_joint.mean()}

    pi = np.clip(BASE * F1**Y1 * F2**Y2 * g ** (Y1 * Y2), 1e-9, 1.0)
    s = rng.uniform(size=N_POP) < pi
    y1, y2, pis = Y1[s], Y2[s], pi[s]
    za, zj = z_add[s], z_joint[s]

    k1, k2 = Y1.mean(), Y2.mean()
    k12 = (Y1 * Y2).mean()
    p1, p2 = y1.mean(), y2.mean()
    w1 = np.where(y1 == 1, k1 / p1, (1 - k1) / (1 - p1))
    w2 = np.where(y2 == 1, k2 / p2, (1 - k2) / (1 - p2))

    weights = {
        "naive": np.ones_like(y1),
        "mean": 0.5 * (w1 + w2),
        "product": w1 * w2,
        "calib_marginal": outcome_calibration_weights(np.column_stack([y1, y2]), [k1, k2]),
        "calib_joint": entropy_balance(np.column_stack([y1, y2, y1 * y2]), [k1, k2, k12]),
        "oracle": 1.0 / pis,
    }
    return truth, {
        m: (hajek(w, za) - truth["add"], hajek(w, zj) - truth["joint"])
        for m, w in weights.items()
    }


def main():
    t0 = time.time()
    print("Bias in two population targets not directly calibrated "
          f"(E[L1+L2] | E[L1*L2]); {N_REPS} reps\n")
    for g in G_VALUES:
        regime = "independent selection (g=1)" if g == 1.0 else f"comorbid interaction (g={g})"
        print(f"--- {regime} ---")
        acc = {m: [] for m in METHODS}
        for rep in range(N_REPS):
            _, est = one_rep(g, 900 + int(g * 10) * 50 + rep)
            for m in METHODS:
                acc[m].append(est[m])
        print(f"{'method':<16}{'E[L1+L2] bias':>16}{'E[L1*L2] bias':>16}")
        for m in METHODS:
            a = np.array(acc[m])
            print(f"{m:<16}{a[:, 0].mean():>16.4f}{a[:, 1].mean():>16.4f}")
        print()
    print("calib_marginal is exact at g=1 (both targets); at g>1 it is biased on the "
          "joint\ntarget until the co-occurrence constraint is added (calib_joint).")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

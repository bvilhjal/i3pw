"""Inferring selection probabilities from many outcomes with known prevalences.

A realistic ascertainment structure. A *latent* selection variable ``U`` drives
participation, ``logit P(S=1 | U) = α + γ·U``. There are ``N`` binary outcomes,
each a noisy proxy for ``U`` (loading ``λ_j``); most are associated with ``U``.
Data available:

- the selection indicator ``S`` for the whole frame;
- ``k`` of the ``N`` outcomes observed for *everyone* (registry-linked);
- the other ``N-k`` observed only in the sample;
- the population **means** (prevalences) of *all* ``N`` outcomes.

Question: how best to infer the selection probabilities (equivalently, the
weights ``1/P(S)``)? Approaches:

- ``naive``      — no weights.
- ``registry``   — fit ``P(S | Y_1..Y_k)`` on the frame (only the outcomes you
  observe for everyone) and use ``1/P̂``.
- ``calib_all``  — calibrate the weights to reproduce all ``N`` known means
  (entropy balancing); every outcome is a proxy for ``U``.
- ``combined``   — registry weights as a base, then calibrate to all ``N`` means.
- ``oracle``     — ``1/P(S|U)`` from the true latent variable.

Each method is scored by how well its (log) weights track the oracle's, and by
the bias it leaves on two population quantities it never calibrated to: a
continuous trait ``Z`` and a held-out outcome ``Y_held`` (both correlated with
``U``). The best feasible strategy uses *both* individual-level outcomes and the
known means; more of either helps; none reaches the oracle, because marginals and
a few proxies cannot fully reconstruct the latent selection variable.

    python examples/selection_probability_inference.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from i3pw import entropy_balance

N_POP = 150_000
N_OUT = 20          # outcomes with known population means
K_OBS = 5           # observed individually for the whole frame (registry-linked)
GAMMA = 1.0         # selection loading on the latent variable
ALPHA = -1.3
N_REPS = 8
METHODS = ("naive", "registry", "calib_all", "combined", "oracle")


def hajek(w, z):
    return float(np.sum(w * z) / np.sum(w))


def one_rep(seed):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(N_POP)
    lam = rng.uniform(0.3, 0.5, N_OUT)
    lam[:K_OBS] = rng.uniform(0.7, 0.85, K_OBS)         # registry outcomes: strong proxies
    Kj = rng.uniform(0.03, 0.30, N_OUT)
    Y = (lam * U[:, None] + np.sqrt(1 - lam**2) * rng.standard_normal((N_POP, N_OUT))
         > norm.ppf(1 - Kj)).astype(float)
    Kpop = Y.mean(axis=0)

    # Held-out population quantities (their means are NOT provided to any method).
    Z = 0.8 * U + rng.standard_normal(N_POP)
    z_truth = Z.mean()
    Yh = (0.5 * U + np.sqrt(0.75) * rng.standard_normal(N_POP) > norm.ppf(1 - 0.1)).astype(float)
    yh_truth = Yh.mean()

    pi = expit(ALPHA + GAMMA * U)
    S = rng.uniform(size=N_POP) < pi
    Ys, pis, Zs, Yhs = Y[S], pi[S], Z[S], Yh[S]

    clf = LogisticRegression(max_iter=300).fit(Y[:, :K_OBS], S.astype(int))
    w_reg = 1.0 / np.clip(clf.predict_proba(Ys[:, :K_OBS])[:, 1], 1e-4, 1 - 1e-4)
    weights = {
        "naive": np.ones_like(pis),
        "registry": w_reg,
        "calib_all": entropy_balance(Ys, Kpop),
        "combined": entropy_balance(Ys, Kpop, base_weights=w_reg),
        "oracle": 1.0 / pis,
    }
    w_or = weights["oracle"]
    out = {}
    for m, w in weights.items():
        corr = np.nan if np.ptp(w) == 0 else np.corrcoef(np.log(w), np.log(w_or))[0, 1]
        out[m] = (corr, abs(hajek(w, Zs) - z_truth), abs(hajek(w, Yhs) - yh_truth))
    return out


def main():
    t0 = time.time()
    acc = {m: [] for m in METHODS}
    for rep in range(N_REPS):
        for m, v in one_rep(1000 + rep).items():
            acc[m].append(v)
    print(f"latent-selection ascertainment: N={N_OUT} outcomes (means known), "
          f"k={K_OBS} observed frame-wide; {N_REPS} reps\n")
    header = (f"{'method':<11}{'corr w/ oracle':>16}"
              f"{'|E[Z]| held-out':>18}{'|E[Yh]-K| held-out':>20}")
    print(header)
    print("-" * len(header))
    for m in METHODS:
        a = np.array(acc[m])
        corr = np.nanmean(a[:, 0])
        cstr = "   n/a" if np.isnan(corr) else f"{corr:+.3f}"
        print(f"{m:<11}{cstr:>16}{a[:, 1].mean():>20.4f}{a[:, 2].mean():>22.4f}")
    print("\ncalibration to the known means is the workhorse; combining it with a model of\n"
          "the individually-observed outcomes is best; none reaches the oracle because the\n"
          "latent selection variable is only imperfectly reconstructed from the proxies.")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

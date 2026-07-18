"""Participation bias and *effect sizes*: what known prevalences can and cannot fix.

The scientific target is usually an **effect size** — an exposure→outcome
association, a genetic association, a Mendelian-randomization estimate — not a
prevalence. This example shows where the known-prevalence machinery helps and,
more importantly, where it does not.

Exposure ``E``, outcome ``Y = 1[βE + noise > t]`` (true effect ``β``), and
participation

    logit P(S=1 | E, Y) = α + δ_E·E + δ_Y·Y

Two regimes:

- **Outcome-only** (``δ_E = 0``): selection depends only on the outcome. By
  Prentice–Pyke, the logistic slope is *already unbiased* — there is nothing to
  correct, and reweighting only adds variance.
- **Collider** (``δ_E ≠ 0``): participation depends on the exposure *and* the
  outcome (the regime behind Schoeler et al.'s distorted genetic associations and
  MR estimates). Now the effect size is biased.

Estimators of ``β``:

- ``naive``       — unweighted logistic regression among participants.
- ``prev_calib``  — weights calibrated to the known outcome prevalence (and the
  known exposure mean). This is the "known prevalence" tool.
- ``model_ipw``   — weights ``1 / P̂(S | E, Y)`` from a participation model that
  *includes the exposure and outcome*.
- ``oracle``      — weights ``1 / P(S | E, Y)`` from the true probabilities.

Takeaway: an effect size is a *conditional association* (a joint moment).
Calibrating the outcome *marginal* does not touch the exposure–outcome joint
selection, so ``prev_calib`` does **not** fix collider bias. Only a
sampling-probability model that includes the exposure does. Known prevalences are
the right information for prevalences and means — not for effect sizes.

    python examples/ukb_participation.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from i3pw import entropy_balance

N_POP = 300_000
N_REPS = 8
BETA = 0.5
K = 0.10


def beta_hat(E, Y, w=None):
    model = LogisticRegression(C=1e6, max_iter=2000)
    model.fit(E.reshape(-1, 1), Y, sample_weight=w)
    return float(model.coef_[0, 0])


def one_rep(delta_E, delta_Y, seed):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal(N_POP)
    L = BETA * E + np.sqrt(1 - BETA**2) * rng.standard_normal(N_POP)
    Y = (L > norm.ppf(1 - K)).astype(float)
    truth = beta_hat(E, Y)  # population effect (large-N)

    pi = 1.0 / (1.0 + np.exp(-(-1.0 + delta_E * E + delta_Y * Y)))
    S = rng.uniform(size=N_POP) < pi
    Es, Ys = E[S], Y[S]
    Kpop, Emean = Y.mean(), E.mean()

    # Calibrate to known outcome prevalence + exposure mean (the "known prevalence" tool).
    w_prev = entropy_balance(np.column_stack([Ys, Es]), [Kpop, Emean]) * Es.size
    # Participation model that includes the exposure and outcome, fit on the frame.
    clf = LogisticRegression(C=1e6, max_iter=2000).fit(np.column_stack([E, Y]), S.astype(int))
    p_model = np.clip(clf.predict_proba(np.column_stack([Es, Ys]))[:, 1], 1e-4, 1 - 1e-4)

    return {
        "truth": truth,
        "naive": beta_hat(Es, Ys),
        "prev_calib": beta_hat(Es, Ys, w_prev),
        "model_ipw": beta_hat(Es, Ys, 1.0 / p_model),
        "oracle": beta_hat(Es, Ys, 1.0 / pi[S]),
    }


def main():
    t0 = time.time()
    methods = ("truth", "naive", "prev_calib", "model_ipw", "oracle")
    for delta_E, tag in ((0.0, "outcome-only selection (delta_E=0)"),
                         (0.8, "collider: exposure + outcome (delta_E=0.8)")):
        acc = {m: [] for m in methods}
        for rep in range(N_REPS):
            for m, v in one_rep(delta_E, -1.5, 1234 + rep).items():
                acc[m].append(v)
        print(f"--- {tag};  true beta={BETA} on the liability scale ---")
        for m in methods:
            print(f"   beta_{m:<11} = {np.mean(acc[m]):+.3f}")
        print()
    print("prev_calib fixes marginals, not the exposure-outcome joint selection, so it\n"
          "does NOT correct collider effect-size bias; a participation model including the\n"
          "exposure (model_ipw) does. Known prevalences are for prevalences, not effect sizes.")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

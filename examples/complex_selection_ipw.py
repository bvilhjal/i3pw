"""When selection also depends on the latent liability: Lee fails, IPW can win.

Extends the probit / liability-threshold benchmark to a selection scheme that
depends not only on case/control status but *slightly on the latent liability*:

    logit P(S=1 | Y, L) = a_Y + delta * L

With ``delta = 0`` this is simple case-control ascertainment. With ``delta != 0``,
higher-liability individuals are over-selected within each group — a common,
realistic distortion (severity-dependent recruitment, super-normal controls).

Estimating the liability-scale variance explained ``R2_L`` under this scheme:

- ``naive`` / ``lee`` — know only ``(K, P)``, so they assume selection is a pure
  function of ``Y``. They cannot see the liability dependence and are biased.
- ``ipw_simple`` — inverse ``K/P`` case/control weights. Same blind spot: corrects
  the case fraction but not the within-group liability selection. Biased.
- ``ipw_fitted`` — weights from a *fitted* ``P(S | X, Y)`` (logistic on a reference
  population). Recovers the part of the liability selection that the predictors
  ``X`` explain; residual (unobserved) liability selection remains.
- ``ipw_oracle`` — weights ``1 / P(S=1 | Y, L)`` from the *true* inclusion
  probabilities. Exact, because it has the actual sampling probabilities.

Takeaway: once selection depends on more than the outcome, the closed-form Lee
transform is stuck — but if you can *estimate* or *know* the sampling
probabilities, IPW still succeeds.

    python examples/complex_selection_ipw.py
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from i3pw.liability import (
    AscertainedSample,
    estimate_liability_r2,
    liability_r2_from_weights,
    simulate_liability_selection,
)

N_POP = 25000
N_PREDICTORS = 150
R2 = 0.6
PREVALENCE = 0.10
N_REPS = 15
DELTAS = (0.0, 0.6, 1.2)
METHODS = ("naive", "lee", "ipw_simple", "ipw_fitted", "ipw_oracle")


def one_rep(delta, seed):
    rng = np.random.default_rng(seed)
    pop = simulate_liability_selection(
        N_POP, N_PREDICTORS, R2, PREVALENCE, rng, liability_slope=delta
    )
    X_sel, y_sel, pi_sel = pop.sample()
    K = pop.K

    # Fitted selection model P(S | X, Y) on the reference population.
    feats = np.column_stack([pop.X, pop.y])
    clf = LogisticRegression(max_iter=500).fit(feats, pop.selected.astype(int))
    p_fit = np.clip(clf.predict_proba(np.column_stack([X_sel, y_sel]))[:, 1], 1e-4, 1 - 1e-4)

    # A plain view of the selected sample for the naive/lee closed forms.
    smp = AscertainedSample(
        X=X_sel, y=y_sel, case_fraction=float(y_sel.mean()), true_r2=pop.true_r2
    )
    return pop.true_r2, {
        "naive": estimate_liability_r2(smp, K, "naive"),
        "lee": estimate_liability_r2(smp, K, "lee"),
        "ipw_simple": estimate_liability_r2(smp, K, "ipw"),
        "ipw_fitted": liability_r2_from_weights(X_sel, y_sel, K, 1.0 / p_fit),
        "ipw_oracle": liability_r2_from_weights(X_sel, y_sel, K, 1.0 / pi_sel),
    }


def main():
    t0 = time.time()
    print(f"Liability-scale R2_L (truth ~= {R2}); selection logit = a_Y + delta*L; "
          f"{N_REPS} reps, prevalence={PREVALENCE}\n")
    header = f"{'delta':>7}{'truth':>9}" + "".join(f"{m:>13}" for m in METHODS)
    print(header)
    print("-" * len(header))
    for delta in DELTAS:
        truth, acc = [], {m: [] for m in METHODS}
        for rep in range(N_REPS):
            tr, est = one_rep(delta, 7000 + int(delta * 10) * 100 + rep)
            truth.append(tr)
            for m in METHODS:
                acc[m].append(est[m])
        cells = "".join(f"{np.mean(acc[m]):>13.3f}" for m in METHODS)
        print(f"{delta:>7.1f}{np.mean(truth):>9.3f}{cells}")
    print("\n(delta=0 is simple case-control; delta>0 adds latent-liability selection)")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

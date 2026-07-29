"""The recommended recipe end to end, on a cohort shaped like a real one.

Every other example here drives a :class:`i3pw.Dataset` — a simulated population that
knows its own ground truth. This one uses :func:`i3pw.calibrate`, which takes only what a
real cohort has: outcomes on the participants, a register prevalence, and a participation
model fitted on the sampling frame. The population is still simulated (there has to be a
truth to check against), but it is consumed the way a cohort would be.

The population: 300k people, a disease at ~5% driven by age and sex, a trait (say BMI)
that depends on both age and disease status. Participation depends on the disease
(``+1.8`` on the log-odds — the channel a covariate model cannot see) *and* on age
(``+0.05`` per year — the channel it can). Both halves of the correction have work to do,
which is the regime the package is about.

Four weightings, in increasing order of how much the docs recommend them:

- ``naive``       — no correction.
- ``model only``  — invert a participation model. Handles age, blind to the disease.
- ``calib only``  — calibrate to the register prevalence from uniform base weights.
                    Handles the disease, blind to age.
- ``both``        — the recommended recipe: calibrate on top of the model.

What to read: the disease column is *exact by construction* for anything calibrated and
proves nothing. The held-out age margin and the trait mean are the honest columns — the
calibration was never given either.

    python examples/real_cohort_workflow.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

import i3pw

N_POP = 300_000
SEED = 11


def simulate():
    """A population and a biased sample of it; returns only what a cohort would hold."""
    rng = np.random.default_rng(SEED)
    age = rng.normal(45.0, 12.0, N_POP)
    female = (rng.uniform(size=N_POP) < 0.51).astype(float)
    disease = (rng.uniform(size=N_POP)
               < expit(-3.2 + 0.03 * (age - 45.0) + 0.4 * female)).astype(float)
    trait = 25.0 + 0.05 * (age - 45.0) + 2.0 * disease + rng.normal(0.0, 3.0, N_POP)

    # Participation: on the disease (invisible to covariates) and on age (visible).
    S = rng.uniform(size=N_POP) < expit(-2.0 + 1.8 * disease + 0.05 * (age - 45.0))

    # A participation model needs covariates on participants AND non-participants; that
    # is what a sampling frame is for. Only the fitted probabilities cross into the
    # cohort-side data below.
    frame = np.column_stack([age, female])
    p_hat = LogisticRegression().fit(frame, S.astype(int)).predict_proba(frame[S])[:, 1]

    cohort = {                      # everything below is observable in a real study
        "disease": disease[S], "age": age[S], "female": female[S], "trait": trait[S],
        "p_hat": p_hat,
    }
    register = {                    # ...and everything here comes from a register
        "prevalence": float(disease.mean()),
        "mean_age": float(age.mean()),
        "pct_female": float(female.mean()),
    }
    return cohort, register, {"trait_mean": float(trait.mean())}  # truth, for scoring only


def main():
    t0 = time.time()
    cohort, register, truth = simulate()
    n = cohort["disease"].size
    base = i3pw.inverse_probability_weights(cohort["p_hat"])
    holdout = {
        "mean age": (cohort["age"], register["mean_age"]),
        "% female": (cohort["female"], register["pct_female"]),
    }

    print(f"cohort: {n} participants of {N_POP}")
    print(f"  disease  sample {cohort['disease'].mean():.4f}  "
          f"register {register['prevalence']:.4f}")
    print(f"  mean age sample {cohort['age'].mean():.2f}    "
          f"register {register['mean_age']:.2f}\n")

    # The recommended recipe, in one call.
    fit = i3pw.calibrate(
        cohort["disease"], [register["prevalence"]],
        base_weights=base, holdout=holdout, outcome_names=["disease"],
    )
    print(fit.summary())

    # The three alternatives, scored on quantities none of them were given.
    runs = {
        "naive": np.full(n, 1.0 / n),
        "model only": base / base.sum(),
        "calib only": i3pw.calibrate(cohort["disease"], [register["prevalence"]]).weights,
        "both (recommended)": fit.weights,
    }
    print(f"\n{'weighting':<20}{'disease':>10}{'held-out age':>15}"
          f"{'trait mean':>13}{'ESS':>9}")
    print("-" * 67)
    for name, w in runs.items():
        prev = float(w @ cohort["disease"])
        age_m = float(w @ cohort["age"])
        trait_m = float(w @ cohort["trait"])
        print(f"{name:<20}{prev:>10.4f}{age_m:>15.2f}{trait_m:>13.3f}"
              f"{i3pw.effective_sample_size(w):>9.0f}")
    print("-" * 67)
    print(f"{'TRUTH':<20}{register['prevalence']:>10.4f}{register['mean_age']:>15.2f}"
          f"{truth['trait_mean']:>13.3f}")

    print(f"\ntrait mean with a calibration-aware interval: "
          f"{fit.mean(cohort['trait']).summary()}")
    print(f"the anchored margin's SE is {fit.mean(cohort['disease']).se:.1e} — zero, "
          "because\nconditional on the register there was never anything to estimate.")

    print("\nRead the last two columns, not the first. 'disease' is exact for anything\n"
          "calibrated (it was the constraint) and means nothing. Age and the trait were\n"
          "never supplied to any of these weightings, so they are what separates them —\n"
          "and only the combination gets both, which is the whole argument for the recipe.")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

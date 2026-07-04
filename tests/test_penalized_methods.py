import numpy as np
import pytest

from i3pw import PenalizedIPW, make_dataset, no_correction, penalized_ipw
from i3pw.methods import cross_validate, lasso_ipw


@pytest.fixture(scope="module")
def dataset():
    return make_dataset(
        seed=1, population_size=5000, n_features=12, n_outcomes=2,
        predictors_per_outcome=6,
        target_population_prevalence=(0.4, 0.1),
        target_sample_prevalence=(0.2, 0.02), sample_size=1200,
    )


def test_prevalence_penalty_pulls_mean_prediction(dataset):
    Xtr, _, s = dataset.split("train")
    pop = dataset.population_prevalence
    plain = PenalizedIPW(lam=0.001, gamma=0.0, learning_rate=0.05,
                         max_iter=3000, decay_interval=1000).fit(Xtr, s, pop)
    informed = PenalizedIPW(lam=0.001, gamma=50.0, learning_rate=0.05,
                            max_iter=3000, decay_interval=1000).fit(Xtr, s, pop)
    mean_plain = plain.predict_inclusion(Xtr).mean(axis=0)
    mean_informed = informed.predict_inclusion(Xtr).mean(axis=0)
    # Strong penalty should drive each outcome's mean prediction to its target.
    assert np.allclose(mean_informed, pop, atol=0.02)
    # ...and closer to the target than the unpenalized fit for both outcomes.
    assert np.all(np.abs(mean_informed - pop) <= np.abs(mean_plain - pop) + 1e-9)


def test_optimizers_agree(dataset):
    Xtr, _, s = dataset.split("train")
    pop = dataset.population_prevalence
    gd = PenalizedIPW(lam=0.01, gamma=0.5, optimizer="gd", learning_rate=0.05,
                      max_iter=8000, decay_interval=4000, tol=1e-10).fit(Xtr, s, pop)
    bfgs = PenalizedIPW(lam=0.01, gamma=0.5, optimizer="bfgs",
                        max_iter=8000, tol=1e-9).fit(Xtr, s, pop)
    # Same optimum reached from two very different optimizers (mean predictions).
    assert np.allclose(gd.predict_inclusion(Xtr).mean(axis=0),
                       bfgs.predict_inclusion(Xtr).mean(axis=0), atol=0.02)


def test_lbfgs_respects_bounds(dataset):
    Xtr, _, s = dataset.split("train")
    pop = dataset.population_prevalence
    est = PenalizedIPW(lam=0.0, gamma=0.0, optimizer="lbfgs",
                       bounds=(-0.1, 0.1), max_iter=5000).fit(Xtr, s, pop)
    # Non-intercept coefficients (columns 1:) must stay inside the box.
    assert est.coef_[:, 1:].max() <= 0.1 + 1e-6
    assert est.coef_[:, 1:].min() >= -0.1 - 1e-6


def test_penalized_beats_no_correction(dataset):
    nc = no_correction(dataset)
    res = penalized_ipw(dataset, lambdas=(0.001, 0.01), gammas=(0.0, 0.1, 1.0),
                        K=3, learning_rate=0.05, max_iter=3000, decay_interval=1000)
    corrected = res["mean"]
    # Correction should reduce the percentage error on both outcomes.
    assert np.all(corrected.percent_diff < nc.percent_diff)


def test_cross_validate_returns_grid_point(dataset):
    Xtr, Ytr, s = dataset.split("train")
    lambdas, gammas = (0.001, 0.01), (0.0, 0.1)
    lam, gamma, table = cross_validate(
        Xtr, s, dataset.population_prevalence, lambdas, gammas, K=3,
        Y_train=Ytr, sample_prevalence=dataset.sample_prevalence,
        learning_rate=0.05, max_iter=2000, decay_interval=1000,
    )
    assert lam in lambdas and gamma in gammas
    assert len(table) == len(lambdas) * len(gammas)


def test_lasso_ipw_runs(dataset):
    res = lasso_ipw(dataset, cv=3)
    assert res.weighted_prevalence.shape == (2,)
    assert np.all(np.isfinite(res.percent_diff))

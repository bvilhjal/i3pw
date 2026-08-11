import numpy as np
import pytest

from i3pw import make_dataset, nearest_pd_correlation, random_correlation
from i3pw._links import sigmoid
from i3pw.dgm import SimConfig, _selection_probabilities


def _within_six_binomial_sd(realized, expected, n):
    variance_bound = expected * (1.0 - expected) / n
    return abs(realized - expected) <= 6.0 * np.sqrt(variance_bound) + 1.0 / n


def test_default_prevalences_adapt_to_n_outcomes():
    # Overriding n_outcomes without passing prevalence tuples must just work: the
    # defaults adapt (the README/dgm docstring examples rely on this).
    cfg = SimConfig(n_outcomes=2)
    assert cfg.target_population_prevalence == (0.4, 0.2)
    assert cfg.target_sample_prevalence == (0.2, 0.1)
    assert len(SimConfig().target_population_prevalence) == 5  # full default unchanged
    ds = make_dataset(seed=1, n_outcomes=2, population_size=1500, sample_size=400)
    assert ds.Y.shape[1] == 2


def test_shapes_and_types():
    ds = make_dataset(seed=3, population_size=2000, n_features=10, n_outcomes=2,
                      predictors_per_outcome=5,
                      target_population_prevalence=(0.4, 0.1),
                      target_sample_prevalence=(0.2, 0.02), sample_size=500)
    assert ds.X.shape == (2000, 10)
    assert ds.Y.shape == (2000, 2)
    assert ds.sample_indicator.shape == (2000,)
    # sample_size is the Bernoulli model's expected count, not a fixed-size draw.
    expected_fraction = 500 / 2000
    assert _within_six_binomial_sd(ds.sample_indicator.mean(), expected_fraction, 2000)
    assert set(np.unique(ds.Y)) <= {0, 1}


def test_outcome_intercepts_match_expected_marginal_prevalence():
    targets = np.array([0.4, 0.1])
    ds = make_dataset(seed=13, population_size=3000, n_features=10, n_outcomes=2,
                      predictors_per_outcome=5,
                      target_population_prevalence=tuple(targets),
                      target_sample_prevalence=(0.2, 0.02), sample_size=600)
    probabilities = sigmoid(ds.intercepts + ds.X @ ds.coefficients.T)

    # Calibration is conditional on the realized covariate matrix, before Y is drawn.
    assert np.allclose(probabilities.mean(axis=0), targets, atol=1e-11, rtol=0.0)
    # Realized binary prevalences fluctuate at the ordinary Monte Carlo scale.
    assert all(
        _within_six_binomial_sd(realized, target, ds.X.shape[0])
        for realized, target in zip(ds.population_prevalence, targets, strict=True)
    )


def test_logistic_selection_matches_expected_count_and_joint_margins():
    # Deliberately use correlated outcomes and a nonconstant covariate offset. The
    # moment equations must be solved jointly; multiplying marginal weights is wrong.
    Y = np.repeat(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        [450, 150, 250, 150],
        axis=0,
    )
    target = np.array([0.2, 0.1])
    expected_size = 300
    probability = _selection_probabilities(
        Y,
        target,
        expected_size,
        covariate_score=np.linspace(-1.0, 1.0, len(Y)),
    )

    assert np.all((probability > 0.0) & (probability < 1.0))
    assert probability.sum() == pytest.approx(expected_size, abs=1e-4)
    expected_margins = (probability[:, None] * Y).sum(axis=0) / probability.sum()
    assert np.allclose(expected_margins, target, atol=1e-7, rtol=0.0)


def test_logistic_selection_rejects_incompatible_joint_margins():
    outcome = np.r_[np.zeros(50), np.ones(50)]
    Y = np.column_stack((outcome, outcome))
    with pytest.raises(ValueError, match="joint margins"):
        _selection_probabilities(Y, np.array([0.2, 0.4]), sample_size=30)


def test_selection_is_biased():
    # The biased sample should skew each outcome's prevalence relative to the
    # population (that is the whole point of the DGM).
    ds = make_dataset(seed=5, population_size=4000, n_features=10, n_outcomes=2,
                      predictors_per_outcome=5,
                      target_population_prevalence=(0.4, 0.1),
                      target_sample_prevalence=(0.2, 0.02), sample_size=1000)
    assert ds.sample_prevalence[0] < ds.population_prevalence[0]
    assert ds.sample_prevalence[1] < ds.population_prevalence[1]


def test_reproducible():
    kw = dict(seed=7, population_size=1500, n_features=8, n_outcomes=2,
              predictors_per_outcome=4,
              target_population_prevalence=(0.4, 0.1),
              target_sample_prevalence=(0.2, 0.02), sample_size=400)
    a = make_dataset(**kw)
    b = make_dataset(**kw)
    assert np.array_equal(a.Y, b.Y)
    assert np.array_equal(a.sample_indicator, b.sample_indicator)


def test_train_test_partition():
    ds = make_dataset(seed=1, population_size=1000, n_features=6, n_outcomes=2,
                      predictors_per_outcome=3,
                      target_population_prevalence=(0.4, 0.1),
                      target_sample_prevalence=(0.2, 0.02), sample_size=300)
    assert len(np.intersect1d(ds.train_idx, ds.test_idx)) == 0
    assert len(ds.train_idx) + len(ds.test_idx) == 1000


def test_nearest_pd_is_positive_definite():
    rng = np.random.default_rng(0)
    corr = random_correlation(20, rng)
    eig = np.linalg.eigvalsh(corr)
    assert eig.min() > 0
    assert np.allclose(np.diag(corr), 1.0)
    # Already-PD identity is returned essentially unchanged.
    assert np.allclose(nearest_pd_correlation(np.eye(5)), np.eye(5))


def test_config_validation():

    with pytest.raises(ValueError):
        SimConfig(n_outcomes=3, target_population_prevalence=(0.4, 0.1))
    with pytest.raises(ValueError, match="target_population_prevalence"):
        SimConfig(n_outcomes=1, target_population_prevalence=(1.0,))
    with pytest.raises(ValueError, match="target_sample_prevalence"):
        SimConfig(n_outcomes=1, target_sample_prevalence=(0.0,))
    with pytest.raises(ValueError, match="sample_size"):
        SimConfig(population_size=100, sample_size=100)


def test_split_rejects_unknown_fold():

    ds = make_dataset(seed=0, population_size=500, n_features=5, n_outcomes=1,
                      predictors_per_outcome=3, sample_size=100)
    with pytest.raises(ValueError, match="train"):
        ds.split("trian")


def test_selection_covariate_strength_creates_a_learnable_participation_signal():
    """Without a covariate channel a participation model has almost nothing to learn.

    The package default (strength 0) makes selection a function of the outcomes alone.
    P(S|X) is then only weakly learnable, and only second-hand — through X -> Y -> S —
    so `lasso_ipw` failing to beat `no_correction` on that setting is a property of the
    simulation, not a discovery about the method. Turning the channel on makes selection
    directly predictable from X, which is what lets a benchmark tell the two channels
    apart.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    def auc(strength):
        ds = make_dataset(seed=0, population_size=4000, n_features=10, n_outcomes=2,
                          predictors_per_outcome=5,
                          target_population_prevalence=(0.4, 0.15),
                          target_sample_prevalence=(0.2, 0.05), sample_size=1500,
                          selection_covariate_strength=strength)
        return cross_val_score(
            LogisticRegression(max_iter=300), ds.X, ds.sample_indicator,
            cv=3, scoring="roc_auc",
        ).mean()

    # At strength 0 the only path from X to S is the indirect one through Y, so a
    # participation model finds only weak, second-hand signal.
    assert auc(0.0) < 0.65
    # With the channel on, selection is directly and strongly predictable from X.
    assert auc(2.0) > 0.85


def test_selection_covariate_strength_preserves_expected_sample_size_and_outcome_bias():
    ds = make_dataset(seed=1, population_size=3000, n_features=8, n_outcomes=2,
                      predictors_per_outcome=4,
                      target_population_prevalence=(0.4, 0.15),
                      target_sample_prevalence=(0.2, 0.05), sample_size=800,
                      selection_covariate_strength=1.0)
    expected_fraction = 800 / 3000
    assert _within_six_binomial_sd(ds.sample_indicator.mean(), expected_fraction, 3000)
    # outcome-dependent selection still bites: the sample is still biased downward
    assert ds.sample_prevalence[0] < ds.population_prevalence[0]

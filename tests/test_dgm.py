import numpy as np

from i3pw import make_dataset, nearest_pd_correlation, random_correlation
from i3pw.dgm import SimConfig


def test_shapes_and_types():
    ds = make_dataset(seed=3, population_size=2000, n_features=10, n_outcomes=2,
                      predictors_per_outcome=5,
                      target_population_prevalence=(0.4, 0.1),
                      target_sample_prevalence=(0.2, 0.02), sample_size=500)
    assert ds.X.shape == (2000, 10)
    assert ds.Y.shape == (2000, 2)
    assert ds.sample_indicator.shape == (2000,)
    assert ds.sample_indicator.sum() == 500
    assert set(np.unique(ds.Y)) <= {0, 1}


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
    import pytest

    with pytest.raises(ValueError):
        SimConfig(n_outcomes=3, target_population_prevalence=(0.4, 0.1))

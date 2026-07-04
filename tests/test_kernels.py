import numpy as np
import pytest
from scipy.optimize import check_grad

from i3pw import _kernels as K
from i3pw import logit


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    n, d = 250, 7
    X = np.ascontiguousarray(rng.normal(size=(n, d)))
    s = (rng.uniform(size=n) < 0.3).astype(float)
    l1_mask = np.ones(d)
    l1_mask[0] = 0.0
    return X, s, l1_mask


@pytest.mark.parametrize("lam,gamma", [(0.0, 0.0), (0.05, 0.0), (0.0, 0.7), (0.05, 0.7)])
def test_gradient_matches_numerical(problem, lam, gamma):
    X, s, l1_mask = problem
    logit_pi = logit(np.array([0.35]))[0]
    beta0 = np.full(X.shape[1], 0.05)  # avoid the L1 kink at 0
    args = (X, s, lam, gamma, logit_pi, l1_mask, 1e-6)
    err = check_grad(K.objective, K.gradient, beta0, *args)
    assert err < 1e-5


def test_gradient_descent_reduces_objective(problem):
    X, s, l1_mask = problem
    logit_pi = logit(np.array([0.35]))[0]
    beta0 = np.zeros(X.shape[1])
    args_obj = (X, s, 0.01, 0.5, logit_pi, l1_mask, 1e-6)
    start = K.objective(beta0, *args_obj)
    beta = K.gradient_descent(X, s, 0.01, 0.5, logit_pi, l1_mask, beta0,
                              0.1, 2000, 500, 1e-8, 1e-6)
    assert K.objective(beta, *args_obj) < start


def test_fit_all_matches_individual_fits(problem):
    # The batched fan-out must return exactly the same coefficients as fitting
    # each outcome on its own.
    X, s, l1_mask = problem
    q, d = 3, X.shape[1]
    betas0 = np.zeros((q, d))
    lams = np.full(q, 0.01)
    gammas = np.array([0.0, 0.5, 1.0])
    logit_pis = logit(np.array([0.2, 0.35, 0.5]))
    batched = K.fit_all_gradient_descent(
        X, s, betas0, lams, gammas, logit_pis, l1_mask, 0.1, 1500, 500, 1e-8, 1e-6
    )
    for j in range(q):
        individual = K.gradient_descent(
            X, s, lams[j], gammas[j], logit_pis[j], l1_mask, betas0[j],
            0.1, 1500, 500, 1e-8, 1e-6,
        )
        assert np.allclose(batched[j], individual, atol=1e-8)

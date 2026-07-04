"""Numba-compiled numerical kernels for the penalized IPW selection model.

The heavy inner loops of the original R code — evaluating the penalized
negative log-likelihood, its gradient, and the gradient-descent updates — are
JIT-compiled here with :func:`numba.njit`. :func:`fit_all_gradient_descent`
loops over the ``Q`` outcome-specific models in a single compiled call.

The fan-out is deliberately serial rather than a ``numba.prange`` across cores:
each per-outcome fit takes only milliseconds, so thread-level parallelism buys
nothing here, and a ``parallel=True`` kernel spawns a full thread pool per
process — which thrashes badly if several fits run concurrently (e.g. multiple
analyses, or a test suite). Serial keeps it fast, robust, and quicker to compile.

Objective for one outcome (mean form), fitting coefficients ``beta`` of a
logistic inclusion model to the sample indicator ``s``::

    f(beta) = -mean( s*log(p) + (1-s)*log(1-p) )          # negative log-likelihood
              + lambda * sum(|beta| * l1_mask)             # LASSO (L1) penalty
              + gamma  * (logit(mean(p)) - logit(pi))^2    # prevalence penalty

where ``p = sigmoid(X @ beta)`` and ``pi`` is the known population prevalence of
the outcome. The final term is the "informed" penalty: it pulls the model's
average predicted inclusion probability toward the outcome's population
prevalence. ``l1_mask`` is 0 for an (unpenalized) intercept column and 1
elsewhere. The gradient is exact.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _sigmoid(z):
    out = np.empty(z.shape[0])
    for i in range(z.shape[0]):
        if z[i] >= 0.0:
            out[i] = 1.0 / (1.0 + np.exp(-z[i]))
        else:
            e = np.exp(z[i])
            out[i] = e / (1.0 + e)
    return out


@njit(cache=True, fastmath=True)
def objective(beta, X, s, lam, gamma, logit_pi, l1_mask, eps):
    n = X.shape[0]
    p = _sigmoid(X @ beta)
    nll = 0.0
    mean_p = 0.0
    for i in range(n):
        pc = min(max(p[i], eps), 1.0 - eps)
        nll += s[i] * np.log(pc) + (1.0 - s[i]) * np.log(1.0 - pc)
        mean_p += p[i]
    nll = -nll / n
    mean_p /= n

    l1 = 0.0
    for j in range(beta.shape[0]):
        l1 += abs(beta[j]) * l1_mask[j]
    l1 *= lam

    m = min(max(mean_p, eps), 1.0 - eps)
    logit_mean = np.log(m / (1.0 - m))
    pen = gamma * (logit_mean - logit_pi) ** 2
    return nll + l1 + pen


@njit(cache=True, fastmath=True)
def gradient(beta, X, s, lam, gamma, logit_pi, l1_mask, eps):
    n, d = X.shape
    p = _sigmoid(X @ beta)

    mean_p = 0.0
    for i in range(n):
        mean_p += p[i]
    mean_p /= n
    m = min(max(mean_p, eps), 1.0 - eps)
    r = np.log(m / (1.0 - m)) - logit_pi
    pen_coef = gamma * 2.0 * r / (m * (1.0 - m))

    # residual vectors for the log-likelihood and the prevalence penalty
    resid = np.empty(n)
    pw = np.empty(n)
    for i in range(n):
        resid[i] = (p[i] - s[i]) / n
        pw[i] = p[i] * (1.0 - p[i]) / n

    grad = X.T @ resid + pen_coef * (X.T @ pw)
    for j in range(d):
        if beta[j] > 0.0:
            grad[j] += lam * l1_mask[j]
        elif beta[j] < 0.0:
            grad[j] -= lam * l1_mask[j]
    return grad


@njit(cache=True, fastmath=True)
def gradient_descent(
    X, s, lam, gamma, logit_pi, l1_mask, beta0, lr, max_iter, decay_interval, tol, eps
):
    """Fixed-step gradient descent with periodic learning-rate halving."""
    beta = beta0.copy()
    prev = objective(beta, X, s, lam, gamma, logit_pi, l1_mask, eps)
    learning_rate = lr
    for it in range(1, max_iter + 1):
        g = gradient(beta, X, s, lam, gamma, logit_pi, l1_mask, eps)
        for j in range(beta.shape[0]):
            beta[j] -= learning_rate * g[j]
        if decay_interval > 0 and it % decay_interval == 0:
            learning_rate *= 0.5
        obj = objective(beta, X, s, lam, gamma, logit_pi, l1_mask, eps)
        if it > 1 and abs(obj - prev) < tol:
            break
        prev = obj
    return beta


@njit(cache=True, fastmath=True)
def fit_all_gradient_descent(
    X, s, betas0, lams, gammas, logit_pis, l1_mask, lr, max_iter, decay_interval, tol, eps
):
    """Fit one inclusion model per outcome in a single compiled call.

    ``betas0`` has shape ``(Q, d)``; ``lams``, ``gammas`` and ``logit_pis`` have
    length ``Q``. Returns the fitted coefficients, shape ``(Q, d)``.
    """
    q = betas0.shape[0]
    out = np.empty_like(betas0)
    for j in range(q):
        out[j] = gradient_descent(
            X, s, lams[j], gammas[j], logit_pis[j], l1_mask,
            betas0[j], lr, max_iter, decay_interval, tol, eps,
        )
    return out

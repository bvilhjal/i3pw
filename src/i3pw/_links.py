"""Numerically stable link functions shared across the package."""

from __future__ import annotations

import numpy as np

# Clamp probabilities away from {0, 1} before taking logs / odds, matching the
# ``epsilon = 1e-6`` guards used throughout the original R implementation.
EPS = 1e-6


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, evaluated without overflow for large ``|z|``."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Inverse of :func:`sigmoid`, with probabilities clipped to ``[eps, 1-eps]``."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def clip_prob(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clip probabilities into the open interval ``(0, 1)``."""
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)

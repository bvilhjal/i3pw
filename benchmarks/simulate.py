"""One simulated population, rich enough for every axis of the validation matrix.

``i3pw.make_dataset`` is deliberately minimal: correlated covariates, binary
outcomes, and a Bernoulli-logistic participation model whose only channels are the
outcomes and an optional additive covariate score. That is enough to illustrate the
estimator and not enough to *test* it. The report's validation matrix asks for
selection laws it cannot express (an outcome-by-covariate interaction, recruitment
that depends on severity *within* case status), for a continuous downstream estimand
that is not one of the calibrated margins, for strata whose prevalence and
participation both vary, and for the true inclusion probabilities so an oracle row
can be reported.

This module supplies exactly that, and nothing that belongs in the package: it is
benchmark scaffolding, not API. The generative model is

    A          ~ Categorical(stratum_shares)                     stratum (sex / birth year)
    X          ~ N(0, Sigma)                                     frame covariates
    u = X @ b                                                    socioeconomic index, Var 1
    L  = h*(X @ a) + m[A] + e                                    liability, disease 1
    Y1 = 1[L > t1],   t1 chosen so mean(Y1) = prevalence
    L2 = h*(X @ a2) + e2                                         liability, disease 2
    Y2 = 1[L2 > t2],  t2 chosen so mean(Y2) = second_prevalence
    Z  = trait_on_covariate*u + trait_on_liability*(L - E L) + noise      the estimand

    logit P(S=1 | .) = alpha
                     + delta_x   * u
                     + delta_y   * Y1
                     + delta_xy  * u * Y1
                     + delta_sev * (L - t1) * Y1
                     + delta_stratum * (A - mean A)

with ``alpha`` solved so the expected participant count equals
``expected_participants``. Each ``delta_`` switches on one channel of the selection
law, which is what makes a sweep over them a sweep over misspecification:

===================  =========================================================
channel              what it does to the identifying assumption
===================  =========================================================
``delta_x``          visible to a participation model on the frame; the base
                     weights are the only ingredient that can correct it.
``delta_y``          invisible to the covariates, recoverable from a register
                     prevalence; the tilt family contains it exactly.
``delta_xy``         in neither span: the log density ratio has a ``u*Y1`` term
                     that base-plus-marginal cannot represent. Misspecified.
``delta_sev``        selection on severity within case status: the prevalence
                     is right and the *case mix* is wrong.
``delta_stratum``    representable only if the calibration is stratified; a
                     pooled margin fixes the count and not its distribution.
===================  =========================================================

Everything is drawn once per seed and returned whole, including ``pi`` — the true
inclusion probabilities, which only a simulation has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from i3pw import random_correlation, sigmoid


@dataclass(frozen=True)
class Design:
    """Parameters of the simulated population and its recruitment."""

    population_size: int = 20_000
    n_features: int = 8
    expected_participants: int = 2_500

    prevalence: float = 0.10
    """Population prevalence of the anchored disease ``Y1``."""

    second_prevalence: float = 0.05
    """Population prevalence of ``Y2``, normally left unanchored to measure transfer."""

    stratum_shares: tuple[float, ...] = (0.50, 0.30, 0.20)
    stratum_liability: tuple[float, ...] = (-0.30, 0.0, 0.45)
    """Liability shift per stratum, so prevalence genuinely varies across strata."""

    heritable_fraction: float = 0.55
    """How much of the liability the frame covariates explain (correlation scale)."""

    severe_fraction: float = 0.35
    """Share of cases the register calls severe, i.e. ``P(severe) = severe_fraction * K``."""

    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_xy: float = 0.0
    delta_sev: float = 0.0
    delta_stratum: float = 0.0
    delta_y_stratum: float = 0.0
    """Stratum-differential ascertainment: cases are recruited harder in some strata."""
    delta_comorbid: float = 0.0
    """Extra recruitment of people carrying *both* diseases; only a co-occurrence
    target can represent it, because it is an interaction between two margins."""

    trait_on_covariate: float = 0.55
    trait_on_liability: float = 0.55
    trait_noise: float = 0.65

    seed: int = 0

    def replace(self, **changes) -> Design:
        """A copy with ``changes`` applied (``dataclasses.replace`` without the import)."""
        from dataclasses import replace as _replace

        return _replace(self, **changes)


@dataclass
class Population:
    """A simulated population, its participants, and the truths only a simulation has."""

    X: np.ndarray               # (N, p) frame covariates, known for everybody
    stratum: np.ndarray         # (N,) integer stratum label
    u: np.ndarray               # (N,) socioeconomic index, unit variance
    liability: np.ndarray       # (N,) latent liability behind Y1
    Y: np.ndarray               # (N, 2) binary diseases
    severe: np.ndarray          # (N,) 1 where Y1 is a severe case, 0 otherwise
    Z: np.ndarray               # (N,) continuous trait: the held-out estimand
    pi: np.ndarray              # (N,) true inclusion probability
    selected: np.ndarray        # (N,) 0/1 participation indicator
    threshold: float            # liability threshold defining Y1
    design: Design = field(repr=False)

    # ---- what an analyst can actually see -------------------------------------
    @property
    def mask(self) -> np.ndarray:
        """Boolean participant mask."""
        return self.selected == 1

    @property
    def n_participants(self) -> int:
        return int(self.selected.sum())

    def frame(self) -> np.ndarray:
        """Covariates a participation model may use: ``X`` plus stratum dummies."""
        dummies = self._stratum_dummies()
        return np.column_stack([self.X, dummies[:, :-1]])

    def _stratum_dummies(self) -> np.ndarray:
        levels = len(self.design.stratum_shares)
        return (self.stratum[:, None] == np.arange(levels)[None, :]).astype(float)

    # ---- register quantities: known without observing participants ------------
    @property
    def population_prevalence(self) -> np.ndarray:
        """``(2,)`` realized population prevalence of each disease."""
        return self.Y.mean(axis=0)

    @property
    def stratum_share(self) -> np.ndarray:
        """``(A,)`` realized population share of each stratum."""
        return self._stratum_dummies().mean(axis=0)

    def within_stratum_prevalence(self, columns: np.ndarray | None = None) -> np.ndarray:
        """``(A, Q)`` realized ``P(column_q = 1 | stratum = a)``; ``Y`` by default."""
        target = self.Y if columns is None else np.atleast_2d(columns)
        if target.shape[0] != self.Y.shape[0]:
            target = target.T
        dummies = self._stratum_dummies()
        counts = dummies.sum(axis=0)
        return (dummies.T @ target) / counts[:, None]

    def severity_columns(self) -> np.ndarray:
        """``(N, 2)`` indicators ``[mild case, severe case]``, which partition the cases.

        A register that records a severity or subtype code knows the prevalence of
        each separately. Calibrating to both anchors the case *mix*, where the pooled
        prevalence anchors only the case count.
        """
        return np.column_stack([self.Y[:, 0] - self.severe, self.severe])

    @property
    def trait_mean(self) -> float:
        """The estimand: the population mean of ``Z``, which participants misreport."""
        return float(self.Z.mean())

    @property
    def trait_sd(self) -> float:
        return float(self.Z.std())


def _threshold_for(values: np.ndarray, prevalence: float) -> float:
    """Liability cut giving exactly the requested realized prevalence."""
    return float(np.quantile(values, 1.0 - prevalence))


def _solve_intercept(score: np.ndarray, expected_count: float) -> float:
    """Solve ``sum_i expit(alpha + score_i) = expected_count`` by bisection.

    Monotone in ``alpha``, so bisection between brackets built from the extreme
    scores converges without magic constants.
    """
    n = score.shape[0]
    target = expected_count / n
    if not 0.0 < target < 1.0:
        raise ValueError("expected_participants must lie strictly between 0 and N.")
    target_logit = float(np.log(target) - np.log1p(-target))
    lo, hi = target_logit - float(score.max()), target_logit - float(score.min())
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if float(sigmoid(mid + score).mean()) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _unit_variance(v: np.ndarray) -> np.ndarray:
    sd = float(v.std())
    return v / sd if sd > 0 else v


def simulate(design: Design) -> Population:
    """Draw one population and its participants under ``design``."""
    rng = np.random.default_rng(design.seed)
    n, p = design.population_size, design.n_features

    corr = random_correlation(p, rng, 0.05, 0.35)
    X = rng.multivariate_normal(np.zeros(p), corr, size=n)

    # Three distinct directions in covariate space: the socioeconomic index that
    # drives participation, and one liability loading per disease. They overlap
    # only through the covariate correlation, which is what makes the covariate
    # model informative about participation but not a substitute for a prevalence.
    b = rng.normal(size=p)
    a1 = rng.normal(size=p)
    a2 = rng.normal(size=p)
    u = _unit_variance(X @ b)
    g1 = _unit_variance(X @ a1)
    g2 = _unit_variance(X @ a2)

    shares = np.asarray(design.stratum_shares, dtype=float)
    shares = shares / shares.sum()
    stratum = rng.choice(len(shares), size=n, p=shares)
    shift = np.asarray(design.stratum_liability, dtype=float)[stratum]

    h = design.heritable_fraction
    residual = float(np.sqrt(max(1.0 - h**2, 1e-12)))
    liability = h * g1 + shift + residual * rng.normal(size=n)
    liability2 = h * g2 + residual * rng.normal(size=n)

    t1 = _threshold_for(liability, design.prevalence)
    t2 = _threshold_for(liability2, design.second_prevalence)
    Y = np.column_stack([(liability > t1).astype(int), (liability2 > t2).astype(int)])
    t_severe = _threshold_for(liability, design.prevalence * design.severe_fraction)
    severe = (liability > t_severe).astype(int)

    Z = (
        design.trait_on_covariate * u
        + design.trait_on_liability * (liability - liability.mean())
        + design.trait_noise * rng.normal(size=n)
    )

    centred_stratum = stratum - stratum.mean()
    score = (
        design.delta_x * u
        + design.delta_y * Y[:, 0]
        + design.delta_xy * u * Y[:, 0]
        + design.delta_sev * (liability - t1) * Y[:, 0]
        + design.delta_stratum * centred_stratum
        + design.delta_y_stratum * centred_stratum * Y[:, 0]
        + design.delta_comorbid * Y[:, 0] * Y[:, 1]
    )
    alpha = _solve_intercept(score, float(design.expected_participants))
    pi = sigmoid(alpha + score)
    selected = (rng.uniform(size=n) < pi).astype(int)

    return Population(
        X=X, stratum=stratum, u=u, liability=liability, Y=Y, severe=severe, Z=Z,
        pi=pi, selected=selected, threshold=t1, design=design,
    )


# Selection laws referenced by name across the benchmark scripts, so that a law
# means the same thing in every table and figure. Strengths are chosen to give
# comparable naive bias (roughly 0.3-0.5 trait SD) rather than comparable
# coefficients, which would make the laws incomparable.
SELECTION_LAWS: dict[str, dict[str, float]] = {
    "X only": dict(delta_x=0.90, delta_y=0.0, delta_xy=0.0, delta_sev=0.0),
    "Y only": dict(delta_x=0.0, delta_y=2.20, delta_xy=0.0, delta_sev=0.0),
    "X + Y": dict(delta_x=0.65, delta_y=1.70, delta_xy=0.0, delta_sev=0.0),
    "X x Y": dict(delta_x=0.65, delta_y=1.70, delta_xy=1.00, delta_sev=0.0),
    "severity": dict(delta_x=0.65, delta_y=1.70, delta_xy=0.0, delta_sev=0.90),
}
"""The five recruitment mechanisms swept in B1; ``X x Y`` and ``severity`` are the
two the base-plus-marginal tilt family cannot represent."""

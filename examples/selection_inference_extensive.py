"""Extensive benchmark: inferring selection probabilities from many outcomes.

Setting (a biobank-style ascertainment). A population of ``N_POP`` units carries
``N`` binary outcomes. Each outcome is a noisy probit proxy for a latent variable
``U`` (loading ``lambda_j``); the first few are strong proxies. Participation is
Bernoulli with

    logit P(S = 1 | U, Y) = alpha + gamma_U * U + sum_j gamma_j * Y_j ,

which lets us dial between three regimes with the *same* estimands:

- ``latent``        — selection driven only by the latent ``U`` (gamma_j = 0).
  The outcomes merely *proxy* the thing that drives participation. This is the
  realistic biobank regime.
- ``case_control``  — selection driven only by a few observed outcomes
  (gamma_U = 0). Classic case-control ascertainment; the analytic Lee-style
  weights are (nearly) exact here.
- ``mixed``         — both a latent driver and outcome-specific over-recruitment.

Information available to the estimators (never the true ``U`` or ``pi``):

- the selection indicator ``S`` for the whole frame;
- ``k`` of the ``N`` outcomes observed for *everyone* (registry-linked);
- the remaining outcomes observed only in the sample;
- the population **means** (prevalences) of *all* ``N`` outcomes.

Methods compared:

- ``naive``      no weights.
- ``lee_cc``     Lee et al. (2011)-style analytic ascertainment weights: for each
  outcome the case-control ratio ``K_j / P_j`` (case) or ``(1-K_j)/(1-P_j)``
  (control), multiplied across all ``N`` outcomes. Model-free; uses exactly the
  ``N`` known means plus the sample outcomes.
- ``registry``   logistic ``P(S | Y_1..Y_k)`` fit on the frame; weights ``1/P_hat``.
- ``calib_all``  entropy balancing of the sample to reproduce all ``N`` known
  means. Same information as ``lee_cc`` but max-entropy joint calibration rather
  than an independence assumption.
- ``combined``   registry weights as a base, then calibrated to the ``N`` means.
- ``oracle``     ``1 / pi`` from the true participation probability.

Scoring. Weights are judged by (a) the correlation of ``log w`` with the oracle's
``log w`` and (b) the bias they leave on two held-out population quantities they
never calibrated to: a continuous trait ``Z`` and a held-out outcome ``Yh``, both
correlated with ``U``. Three studies run:

  A. scenario comparison at a fixed (N, k);
  B. sweep over N (number of known prevalences), latent regime;
  C. sweep over k (number of frame-wide outcomes), latent regime;
  D. a Schoeler et al. (2023)-style covariate model vs prevalence calibration, in a
     population where selection depends on both socioeconomic covariates ``X`` and
     the disease latent ``U`` (see :func:`simulate_cov`).

    python examples/selection_inference_extensive.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from i3pw import effective_sample_size, entropy_balance

N_POP = 60_000
ALPHA = -1.4                       # baseline participation log-odds
SEL_OUTCOMES = (0, 1, 2)           # outcomes selection acts on in case_control/mixed
STRONG = 3                         # this many leading outcomes are strong U-proxies


@dataclass
class Rep:
    Ys: np.ndarray       # (n_sel, N) sample outcomes
    Kpop: np.ndarray     # (N,) population prevalences
    Yframe: np.ndarray   # (N_POP, N) frame outcomes (for the registry model)
    S: np.ndarray        # (N_POP,) selection indicator
    pis: np.ndarray      # (n_sel,) true inclusion probabilities
    Zs: np.ndarray       # (n_sel,) held-out continuous trait
    Yhs: np.ndarray      # (n_sel,) held-out outcome
    z_truth: float
    yh_truth: float


def simulate(seed: int, n_out: int, scenario: str, gamma_u: float, gamma_y: float) -> Rep:
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(N_POP)
    lam = rng.uniform(0.30, 0.50, n_out)
    lam[:STRONG] = rng.uniform(0.70, 0.85, min(STRONG, n_out))
    Kj = rng.uniform(0.05, 0.30, n_out)
    noise = rng.standard_normal((N_POP, n_out))
    liab = lam * U[:, None] + np.sqrt(1.0 - lam**2) * noise
    Y = (liab > norm.ppf(1.0 - Kj)).astype(float)
    Kpop = Y.mean(axis=0)

    # Held-out population quantities (their means are given to no method).
    Z = 0.8 * U + rng.standard_normal(N_POP)
    Yh = (0.5 * U + np.sqrt(0.75) * rng.standard_normal(N_POP) > norm.ppf(1.0 - 0.1)).astype(float)

    lin = np.full(N_POP, ALPHA)
    if scenario in ("latent", "mixed"):
        lin = lin + gamma_u * U
    if scenario in ("case_control", "mixed"):
        sel = [j for j in SEL_OUTCOMES if j < n_out]
        lin = lin + gamma_y * Y[:, sel].sum(axis=1)
    pi = expit(lin)
    S = rng.uniform(size=N_POP) < pi
    return Rep(
        Ys=Y[S], Kpop=Kpop, Yframe=Y, S=S, pis=pi[S],
        Zs=Z[S], Yhs=Yh[S], z_truth=float(Z.mean()), yh_truth=float(Yh.mean()),
    )


def lee_cc_weights(Ys: np.ndarray, Kpop: np.ndarray) -> np.ndarray:
    """Product of per-outcome case-control ratios (Lee-style analytic weights)."""
    P = np.clip(Ys.mean(axis=0), 1e-3, 1 - 1e-3)
    K = np.clip(Kpop, 1e-3, 1 - 1e-3)
    ratio_case = K / P
    ratio_ctrl = (1 - K) / (1 - P)
    logw = (Ys * np.log(ratio_case) + (1 - Ys) * np.log(ratio_ctrl)).sum(axis=1)
    logw -= logw.max()
    return np.exp(logw)


def hajek(w: np.ndarray, z: np.ndarray) -> float:
    return float(np.sum(w * z) / np.sum(w))


def run_methods(rep: Rep, k: int, methods: tuple[str, ...]) -> dict[str, tuple]:
    Ys, Kpop = rep.Ys, rep.Kpop
    w_oracle = 1.0 / rep.pis
    weights: dict[str, np.ndarray] = {"naive": np.ones(len(Ys)), "oracle": w_oracle}
    if "lee_cc" in methods:
        weights["lee_cc"] = lee_cc_weights(Ys, Kpop)
    w_reg = None
    if k > 0 and ("registry" in methods or "combined" in methods):
        clf = LogisticRegression(max_iter=300).fit(rep.Yframe[:, :k], rep.S.astype(int))
        w_reg = 1.0 / np.clip(clf.predict_proba(Ys[:, :k])[:, 1], 1e-4, 1 - 1e-4)
    if "registry" in methods and w_reg is not None:
        weights["registry"] = w_reg
    if "calib_all" in methods:
        weights["calib_all"] = entropy_balance(Ys, Kpop)
    if "combined" in methods:
        base = w_reg if w_reg is not None else None
        weights["combined"] = entropy_balance(Ys, Kpop, base_weights=base)

    lw_or = np.log(w_oracle)
    out = {}
    for m in methods:
        w = weights.get(m)
        if w is None:
            out[m] = None
            continue
        corr = np.nan if np.ptp(w) == 0 else float(np.corrcoef(np.log(w), lw_or)[0, 1])
        ess = effective_sample_size(w) / len(w)
        out[m] = (corr, abs(hajek(w, rep.Zs) - rep.z_truth),
                  abs(hajek(w, rep.Yhs) - rep.yh_truth), ess)
    return out


def aggregate(runs: list[dict[str, tuple]], methods: tuple[str, ...]) -> dict[str, tuple]:
    agg = {}
    for m in methods:
        vals = [r[m] for r in runs if r[m] is not None]
        if not vals:
            agg[m] = None
            continue
        a = np.array(vals)
        corr = np.nan if np.all(np.isnan(a[:, 0])) else np.nanmean(a[:, 0])
        agg[m] = (corr, a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean())
    return agg


def print_idx_table(title: str, rowlabel: str, rows: list[tuple[str, dict]],
                    methods: tuple[str, ...], idx: int, fmt: str, nan_ok: bool = False):
    print(f"\n{title}")
    head = f"{rowlabel:<14}" + "".join(f"{m:>12}" for m in methods)
    print(head)
    print("-" * len(head))
    for label, agg in rows:
        cells = []
        for m in methods:
            v = agg[m]
            if v is None or (nan_ok and np.isnan(v[idx])):
                cells.append("     n/a " if nan_ok else "     --  ")
            else:
                cells.append(format(v[idx], fmt).rjust(12))
        print(f"{label:<14}" + "".join(cells))


def study_scenarios(reps: int):
    methods = ("naive", "lee_cc", "registry", "calib_all", "combined", "oracle")
    n_out, k = 16, 5
    configs = [
        ("latent", "latent", 1.2, 0.0),
        ("case_control", "case_control", 0.0, 1.3),
        ("mixed", "mixed", 0.7, 1.0),
    ]
    bias_rows, corr_rows = [], []
    for label, scen, gu, gy in configs:
        runs = [run_methods(simulate(700 + r, n_out, scen, gu, gy), k, methods)
                for r in range(reps)]
        agg = aggregate(runs, methods)
        bias_rows.append((label, agg))
        corr_rows.append((label, agg))
    print("\n" + "=" * 72)
    print(f"STUDY A — scenario comparison  (N={n_out} known means, k={k} frame-wide)")
    print("=" * 72)
    print_idx_table("Held-out |E[Z] - truth| (trait correlated with U) - lower is better",
                    "scenario", bias_rows, methods, idx=1, fmt=".4f")
    print_idx_table("Effective sample size (fraction of n) - higher is less variance",
                    "scenario", bias_rows, methods, idx=3, fmt=".3f")
    print_idx_table("corr( log w, log w_oracle ) - higher is better",
                    "scenario", corr_rows, methods, idx=0, fmt="+.3f", nan_ok=True)


def study_sweep_n(reps: int):
    methods = ("naive", "lee_cc", "registry", "calib_all", "combined", "oracle")
    k = 3
    rows = []
    for n_out in (4, 8, 16, 32):
        runs = [run_methods(simulate(1700 + r, n_out, "latent", 1.2, 0.0), k, methods)
                for r in range(reps)]
        rows.append((f"N={n_out}", aggregate(runs, methods)))
    print("\n" + "=" * 72)
    print(f"STUDY B — more known prevalences  (latent regime, k={k} frame-wide)")
    print("=" * 72)
    print_idx_table("Held-out |E[Z] - truth|", "N outcomes", rows, methods, idx=1, fmt=".4f")


def study_sweep_k(reps: int):
    methods = ("naive", "lee_cc", "registry", "calib_all", "combined", "oracle")
    n_out = 16
    rows = []
    for k in (1, 3, 6, 12):
        runs = [run_methods(simulate(2700 + r, n_out, "latent", 1.2, 0.0), k, methods)
                for r in range(reps)]
        rows.append((f"k={k}", aggregate(runs, methods)))
    print("\n" + "=" * 72)
    print(f"STUDY C — more frame-wide outcomes  (latent regime, N={n_out} known means)")
    print("=" * 72)
    print_idx_table("Held-out |E[Z] - truth|", "k frame-wide", rows, methods, idx=1, fmt=".4f")


@dataclass
class CovRep:
    Xs: np.ndarray       # (n_sel, p) sample covariates
    Xframe: np.ndarray   # (N_POP, p) frame covariates (for the Schoeler model)
    Ys: np.ndarray       # (n_sel, N) sample outcomes
    Kpop: np.ndarray     # (N,) population outcome prevalences
    S: np.ndarray        # (N_POP,) selection indicator
    pis: np.ndarray      # (n_sel,) true inclusion probabilities
    Zs: np.ndarray       # (n_sel,) held-out trait (loads on BOTH X and U)
    z_truth: float


N_COV = 10               # socioeconomic covariates (only the first few drive anything)
N_COV_ACTIVE = 4


def simulate_cov(seed: int, n_out: int, c_x: float, c_u: float) -> CovRep:
    """Selection driven by BOTH socioeconomic covariates X and a disease latent U.

    ``logit P(S) = alpha + c_x * (X @ b) + c_u * U``, with ``X`` independent of ``U``.
    A held-out trait ``Z`` loads on the same X direction *and* on U, so removing the
    selection bias in ``E[Z]`` requires correcting both channels. Schoeler-style
    covariate models see only the X channel; prevalence calibration sees only the U
    channel (through the outcomes that proxy it).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_POP, N_COV))
    b = np.zeros(N_COV)
    b[:N_COV_ACTIVE] = rng.uniform(0.4, 0.9, N_COV_ACTIVE) * rng.choice([-1.0, 1.0], N_COV_ACTIVE)
    xb = X @ b
    xb = xb / xb.std()                                   # unit-variance X selection index

    U = rng.standard_normal(N_POP)
    lam = rng.uniform(0.30, 0.50, n_out)
    lam[:STRONG] = rng.uniform(0.70, 0.85, min(STRONG, n_out))
    Kj = rng.uniform(0.05, 0.30, n_out)
    Y = (lam * U[:, None] + np.sqrt(1.0 - lam**2) * rng.standard_normal((N_POP, n_out))
         > norm.ppf(1.0 - Kj)).astype(float)
    Kpop = Y.mean(axis=0)

    Z = 0.7 * xb + 0.7 * U + rng.standard_normal(N_POP)  # loads on both channels
    pi = expit(-1.4 + c_x * xb + c_u * U)
    S = rng.uniform(size=N_POP) < pi
    return CovRep(Xs=X[S], Xframe=X, Ys=Y[S], Kpop=Kpop, S=S, pis=pi[S],
                  Zs=Z[S], z_truth=float(Z.mean()))


def schoeler_weights(rep: CovRep) -> np.ndarray:
    """Schoeler et al. (2023)-style weights: L1-penalized logistic P(S | X), inverted."""
    clf = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.5, max_iter=500)
    clf.fit(rep.Xframe, rep.S.astype(int))
    return 1.0 / np.clip(clf.predict_proba(rep.Xs)[:, 1], 1e-4, 1 - 1e-4)


def study_schoeler(reps: int):
    methods = ("naive", "schoeler", "calib_all", "sch+prev", "oracle")
    n_out = 16
    settings = [
        ("socioeconomic", 1.4, 0.4),   # selection mostly X-driven
        ("balanced", 0.9, 0.9),
        ("disease", 0.4, 1.4),         # selection mostly disease/latent-driven
    ]
    rows = []
    for label, c_x, c_u in settings:
        runs = []
        for r in range(reps):
            rep = simulate_cov(3700 + r, n_out, c_x, c_u)
            w_sch = schoeler_weights(rep)
            weights = {
                "naive": np.ones(len(rep.Ys)),
                "schoeler": w_sch,
                "calib_all": entropy_balance(rep.Ys, rep.Kpop),
                "sch+prev": entropy_balance(rep.Ys, rep.Kpop, base_weights=w_sch),
                "oracle": 1.0 / rep.pis,
            }
            runs.append({m: (np.nan, abs(hajek(w, rep.Zs) - rep.z_truth), np.nan,
                             effective_sample_size(w) / len(w)) for m, w in weights.items()})
        rows.append((label, aggregate(runs, methods)))
    print("\n" + "=" * 72)
    print(f"STUDY D — Schoeler covariate model vs prevalence calibration  (N={n_out} means)")
    print("=" * 72)
    print("Selection = alpha + c_x*(X@b) + c_u*U; X (socioeconomic) independent of U.")
    print_idx_table("Held-out |E[Z] - truth| (Z loads on both X and U) - lower is better",
                    "sel. channel", rows, methods, idx=1, fmt=".4f")


def main():
    t0 = time.time()
    reps = 20
    print(f"Extensive selection-probability inference benchmark "
          f"(N_POP={N_POP:,}, {reps} reps/cell)")
    study_scenarios(reps)
    study_sweep_n(reps)
    study_sweep_k(reps)
    study_schoeler(reps)
    print("\nReading the tables — no method is uniformly best; the regime decides:")
    print(" - latent regime (all outcomes proxy one hidden driver): the analytic Lee-style")
    print("   weights (lee_cc) are startlingly good — averaging N simple case-control")
    print("   corrections reconstructs the latent driver with low variance, beating exact")
    print("   joint calibration, which chases sampling noise in each of the N margins.")
    print(" - case_control regime (a few correlated outcomes drive selection): lee_cc now")
    print("   OVER-corrects — it applies an independent correction for every outcome even")
    print("   though most only correlate with the true drivers — and is barely better than")
    print("   naive. A registry model or exact calibration, which can't push past the true")
    print("   margins, are near-exact here.")
    print(" - combined (registry model + calibration to the known means) is the robust")
    print("   choice: never catastrophic in any regime. When you don't know the mechanism,")
    print("   it is the safe default; lee_cc is a high-variance bet that pays off only when")
    print("   selection really is a latent factor cleanly proxied by all your outcomes.")
    print(" - Study B: calibration bias falls monotonically as the number of known")
    print("   prevalences (N) grows, but lee_cc is non-monotonic — it improves, then")
    print("   degrades once many weak, correlated outcomes each add an over-correction.")
    print(" - Study C: only the registry model (and combined) benefit from more frame-wide")
    print("   outcomes (k); lee_cc and calib_all ignore the frame and are flat in k. None")
    print("   reaches the oracle because a latent driver is only proxied, never seen.")
    print(" - Study D: the Schoeler covariate model fixes only the socioeconomic (X) channel")
    print("   of selection and is near-useless when selection is disease-driven; prevalence")
    print("   calibration fixes only the disease (U) channel. They are complementary, and")
    print("   sch+prev (covariate weights calibrated to the known means) handles both.")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

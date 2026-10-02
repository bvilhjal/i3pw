"""Guard for the numbers quoted in the prose docs and the report's LaTeX.

The figure drift guard in ``test_benchmarks.py`` proves the *generated* displays
match the frozen artifact. This file extends the same discipline to the
hand-written prose: every check derives its expected string from
``report/benchmark_results.tsv`` and asserts the documents still contain it. If
the artifact is ever re-frozen, these assertions fail until every quoted number
is resynchronized — the failure mode that shipped in 0.3.2, when the markdown
docs (and the report's B6 narrative) kept quoting a superseded freeze while
citing the current one.

The table is curated, not exhaustive: it covers the load-bearing claims (the
ones the README's recommendations rest on). When a new numeric claim is added to
the docs, add a line here.

Formatting conventions the derivations must respect: the markdown docs use the
unicode minus (U+2212) and en dash (U+2013); the LaTeX source uses ASCII signs
and ``--``/``\\%``. Rounding matches how each claim is written (3 decimals for
SD-unit biases, 1 for percentages the report spells to a decimal, 0 for the ones
it rounds).
"""

from __future__ import annotations

import math
import re

import pytest

from benchmarks.harness import REPO_ROOT, RESULTS_TSV, read_tsv

pytestmark = pytest.mark.skipif(
    not RESULTS_TSV.exists(), reason="benchmark artifact not present"
)

MINUS = "\u2212"   # the markdown docs' minus sign
ENDASH = "\u2013"  # ...and en dash

DOCS = {
    "studies": REPO_ROOT / "docs" / "studies.md",
    "guide": REPO_ROOT / "docs" / "guide.md",
    "theory": REPO_ROOT / "docs" / "theory.md",
    "readme": REPO_ROOT / "README.md",
    "tex": REPO_ROOT / "report" / "i3pw_report.tex",
}


@pytest.fixture(scope="module")
def text():
    """The documents, normalized for prose-level substring checks.

    LaTeX math wrappers and percent escapes are removed and whitespace runs are
    collapsed, so a needle matches the sentence as read rather than its
    typesetting (line wraps and ``\\( ... \\)`` are formatting, not content).
    """
    out = {}
    for name, path in DOCS.items():
        t = path.read_text(encoding="utf-8")
        if name == "tex":
            t = t.replace("\\(", "").replace("\\)", "").replace("\\%", "%")
        out[name] = re.sub(r"\s+", " ", t)
    return out


@pytest.fixture(scope="module")
def tsv():
    rows = read_tsv(RESULTS_TSV)
    index = {(r["benchmark"], r["condition"], r["estimator"], r["metric"]): r
             for r in rows}
    by_condition: dict[tuple[str, str], set] = {}
    for r in rows:
        by_condition.setdefault((r["benchmark"], r["condition"]), set()).add(r["estimator"])
    return index, by_condition


def mean(tsv, benchmark, condition, estimator, metric):
    index, _ = tsv
    return index[(benchmark, condition, estimator, metric)]["mean"]


def sole_estimator(tsv, benchmark, condition):
    """The estimator label the artifact records under one condition.

    Resolved from the artifact (not hardcoded) so this guard keeps working when
    a future freeze replaces the legacy uniform ``ipw+cal`` labels of B2/B8 with
    the accurate per-arm ones.
    """
    _, by_condition = tsv
    found = by_condition[(benchmark, condition)]
    assert len(found) == 1, (benchmark, condition, found)
    return next(iter(found))


def s3(v):   # signed, 3 decimals, ASCII (LaTeX)
    return f"{v:+.3f}"


def m3(v):   # signed, 3 decimals, unicode minus (markdown)
    return f"{v:+.3f}".replace("-", MINUS)


def u3(v):   # unsigned, 3 decimals
    return f"{v:.3f}"


def pct(v, nd=0):  # percentage with nd decimals
    return f"{v * 100:.{nd}f}%"




B1_LAWS = ("X only", "Y only", "X + Y", "X x Y", "severity")
B1_ESTIMATORS = ("naive", "ipw", "cal", "ipw+cal", "aipw", "oracle")


def b1(tsv, law, estimator, metric="trait_bias_sd"):
    return mean(tsv, "B1_selection_law", law, estimator, metric)


def b5(tsv, regime, procedure, metric):
    return mean(tsv, "B5_interval_coverage", regime, procedure, metric)


def b6(tsv, condition, metric):
    return mean(tsv, "B6_support", condition, "ipw+cal", metric)


# --------------------------------------------------------------------------------
# B1: the selection-law table and its derived claims
# --------------------------------------------------------------------------------

def test_studies_b1_table_cells(text, tsv):
    doc = text["studies"]
    for law in B1_LAWS:
        for est in B1_ESTIMATORS:
            assert m3(b1(tsv, law, est)) in doc, (law, est)


def test_tex_b1_prose_cells(text, tsv):
    doc = text["tex"]
    for law, est in [("X only", "naive"), ("X only", "cal"), ("Y only", "naive"),
                     ("Y only", "ipw"), ("Y only", "cal"), ("Y only", "oracle"),
                     ("Y only", "ipw+cal")]:
        assert s3(b1(tsv, law, est)) in doc, (law, est)


def test_base_model_harm_ratio(text, tsv):
    """README/studies' '19x worse' and the tex's 'nineteen times worse'."""
    ratio = abs(b1(tsv, "Y only", "ipw+cal")) / abs(b1(tsv, "Y only", "cal"))
    assert round(ratio) == 19, ratio
    assert f"{round(ratio)}\u00d7 worse" in text["readme"]
    assert f"{round(ratio)}\u00d7 worse" in text["studies"]
    assert "nineteen times worse" in text["tex"]


def test_oracle_bound_and_mixed_law_residuals(text, tsv):
    worst_oracle = max(abs(b1(tsv, law, "oracle")) for law in B1_LAWS)
    bound = u3(math.ceil(worst_oracle * 1000) / 1000)
    assert bound == "0.006"  # if this moves, both docs and the ceil-rounding move
    assert f"within {bound} of zero" in text["studies"]
    assert f"within {bound} SD" in text["tex"]
    mixed = [abs(b1(tsv, law, "ipw+cal"))
             for law in ("X + Y", "X x Y", "severity")]
    lo, hi = u3(min(mixed)), u3(max(mixed))
    assert f"{lo}{ENDASH}{hi}" in text["studies"]
    assert f"{lo}--{hi}" in text["tex"]


def test_alarm_not_ranking_smds(text, tsv):
    smd = lambda law, est: b1(tsv, law, est, "worst_held_out_smd")  # noqa: E731
    pairs = [(smd("Y only", "ipw+cal"), "0.146"), (smd("Y only", "cal"), "0.035"),
             (smd("X + Y", "ipw"), "0.023"), (smd("X + Y", "ipw+cal"), "0.106")]
    for value, expected in pairs:
        assert u3(value) == expected
        for name in ("studies", "guide", "tex"):
            assert expected in text[name], (name, expected)
    # the biases quoted beside them, and the guide/README's 3.5x ratio
    assert u3(abs(b1(tsv, "X + Y", "ipw"))) == "0.120"
    assert u3(abs(b1(tsv, "X + Y", "ipw+cal"))) == "0.034"
    ratio = abs(b1(tsv, "X + Y", "ipw")) / abs(b1(tsv, "X + Y", "ipw+cal"))
    assert f"{ratio:.1f}\u00d7 the bias" in text["guide"]
    assert f"{ratio:.1f}\u00d7 the bias" in text["readme"]
    assert u3(abs(b1(tsv, "Y only", "ipw+cal"))) == "0.065"
    assert u3(abs(b1(tsv, "Y only", "cal"))) == "0.003"


# --------------------------------------------------------------------------------
# B5: coverage
# --------------------------------------------------------------------------------

REGIMES = ("correct", "base + margin", "misspecified")
PROCEDURES = ("fixed-weight", "calibration-aware", "bootstrap")


def test_coverage_tables(text, tsv):
    for regime in REGIMES:
        cells = [u3(b5(tsv, regime, p, "coverage_95")) for p in PROCEDURES]
        width = u3(b5(tsv, regime, "calibration-aware", "interval_width_sd"))
        for name in ("studies", "guide"):
            for cell in (*cells, width):
                assert cell in text[name], (name, regime, cell)
    biases = [b5(tsv, r, "estimator", "trait_bias_sd") for r in REGIMES]
    for cell in (m3(biases[0]), m3(biases[1]), m3(biases[2])):
        assert cell in text["studies"], cell


def test_coverage_prose(text, tsv):
    cells = [u3(b5(tsv, "correct", p, "coverage_95")) for p in PROCEDURES]
    assert f"{cells[0]}, {cells[1]} and {cells[2]}" in text["tex"]
    for regime, phrase in (("base + margin", "0.62--0.64"), ("misspecified", "0.36--0.41")):
        covs = [b5(tsv, regime, p, "coverage_95") for p in PROCEDURES]
        assert f"{min(covs):.2f}--{max(covs):.2f}" == phrase
        assert phrase in text["tex"]
    widths = [b5(tsv, r, "calibration-aware", "interval_width_sd") for r in REGIMES]
    halves = [w / 2 for w in widths]
    assert f"{u3(widths[0])}, {u3(widths[1])} and {u3(widths[2])}" in text["tex"]
    assert f"{u3(halves[0])}, {u3(halves[1])} and {u3(halves[2])}" in text["tex"]
    assert f"{u3(halves[0])}, {u3(halves[1])} and {u3(halves[2])}" in text["guide"]
    biases = [abs(b5(tsv, r, "estimator", "trait_bias_sd")) for r in REGIMES]
    shares = [f"{100 * b / h:.0f}" for b, h in zip(biases, halves, strict=True)]
    assert f"{shares[0]}%, {shares[1]}% and {shares[2]}%" in text["tex"]
    assert f"{shares[1]}% of the half-width" in text["guide"]
    assert f"{shares[2]}% of it" in text["guide"]
    assert f"{u3(biases[1])} SD bias" in text["guide"]
    assert f"{u3(biases[2])} SD is" in text["guide"]
    # README's rounded ranges
    correct = [b5(tsv, "correct", p, "coverage_95") for p in PROCEDURES]
    middle = [b5(tsv, "base + margin", p, "coverage_95") for p in PROCEDURES]
    assert f"{min(correct):.2f}{ENDASH}{max(correct):.2f}" in text["readme"]
    assert f"**{min(middle):.2f}{ENDASH}{max(middle):.2f}**" in text["readme"]


# --------------------------------------------------------------------------------
# B4: case mix
# --------------------------------------------------------------------------------

def b4(tsv, condition, estimator, metric):
    return mean(tsv, "B4_case_mix", condition, estimator, metric)


def test_case_mix_claims(text, tsv):
    cond_s, cond_v = "stratum-differential", "within-case severity"
    within = [b4(tsv, cond_s, e, "worst_within_stratum_prevalence_error")
              for e in ("ipw+cal", "ipw+cal/v", "oracle")]
    assert u3(within[0]) == u3(within[1]) == "0.172"
    assert u3(within[2]) == "0.012"
    for name in ("studies", "guide", "readme", "tex"):
        assert "0.172" in text[name], name
    trait = [b4(tsv, cond_s, e, "trait_bias_sd")
             for e in ("ipw+cal", "ipw+cal/s", "ipw+cal/v", "oracle")]
    assert (m3(trait[0]), m3(trait[1]), m3(trait[3])) == (
        f"{MINUS}0.045", f"{MINUS}0.014", f"{MINUS}0.001")
    for name in ("studies", "guide", "theory"):
        for cell in (m3(trait[0]), m3(trait[1]), m3(trait[3])):
            assert cell in text[name], (name, cell)
    for cell in (s3(trait[0]), s3(trait[1])):
        assert cell in text["tex"], cell
    mix = [b4(tsv, cond_v, e, "case_mix_liability_error")
           for e in ("ipw+cal", "ipw+cal/s", "ipw+cal/v", "oracle")]
    assert (m3(mix[0]), m3(mix[1]), m3(mix[2])) == ("+0.060", "+0.067", "+0.016")
    for name in ("studies", "guide", "theory", "tex"):
        for cell in (("+0.060", "+0.067", "+0.016")):
            assert cell in text[name], (name, cell)
    ratio = mix[0] / mix[2]
    assert f"{ratio:.1f}\u00d7" in text["studies"]
    assert f"{ratio:.1f}\u00d7" in text["readme"]


# --------------------------------------------------------------------------------
# B6: support
# --------------------------------------------------------------------------------

def test_b6_support_claims(text, tsv):
    under = {k: f"under-recruited | K={k}" for k in ("0.02", "0.01", "0.005", "0.002")}
    disc = {k: b6(tsv, c, "bootstrap_discard_rate") for k, c in under.items()}
    solve = {k: b6(tsv, c, "solve_success_rate") for k, c in under.items()}
    assert pct(disc["0.02"], 1) == "0.4%"
    assert pct(disc["0.01"], 1) == "7.9%"
    assert f"{(1 - solve['0.01']) * 100:.0f}%" == "3%"  # 1 rep in 30
    assert f"{(1 - solve['0.005']) * 100:.0f}%" == "17%"
    assert f"{(1 - solve['0.002']) * 100:.0f}%" == "47%"
    assert f"{disc['0.005'] * 100:.0f}%" == "13%"
    assert f"{disc['0.002'] * 100:.0f}%" == "25%"
    # tex spells these with escaped percents in the source; the fixture unescapes
    for needle in (pct(disc["0.02"], 1), pct(disc["0.01"], 1),
                   f"{(1 - solve['0.005']) * 100:.0f}%",
                   f"{(1 - solve['0.002']) * 100:.0f}%",
                   f"{disc['0.005'] * 100:.0f}%"):
        assert needle in text["tex"], needle
    # markdown-side phrasings
    assert "(0.4%)" in text["studies"]
    assert "8% of replicates" in text["studies"]
    assert "17% of solves fail and 13%" in text["studies"]
    assert "47% and 25%" in text["studies"]
    assert "(1 rep in 30)" in text["studies"]
    # weight-concentration peaks and the over-recruited survival claim
    peak_under = max(b6(tsv, f"under-recruited | K={k}", "max_weight_share")
                     for k in ("0.2", "0.1", "0.05", "0.02", "0.01"))
    peak_over = max(b6(tsv, f"over-recruited | K={k}", "max_weight_share")
                    for k in ("0.2", "0.1", "0.05", "0.02", "0.01"))
    assert f"{peak_under:.1f}" == "21.5" and f"{peak_over:.1f}" == "11.5"
    assert f"{peak_under:.1f}" in text["tex"] and f"{peak_over:.1f}" in text["tex"]
    cases_over_rarest = b6(tsv, "over-recruited | K=0.001", "sampled_cases")
    assert int(cases_over_rarest) == 9
    assert "nine sampled cases" in text["tex"]
    assert "survives to nine sampled cases" in text["studies"]


# --------------------------------------------------------------------------------
# B2 and B3: the ladder and the register-error sweep
# --------------------------------------------------------------------------------

def b2(tsv, rung, metric):
    return mean(tsv, "B2_anchor_information", rung,
                sole_estimator(tsv, "B2_anchor_information", rung), metric)


def test_b2_ladder_claims(text, tsv):
    none_bias, first_bias = b2(tsv, "none", "trait_bias_sd"), b2(tsv, "K(Y1)", "trait_bias_sd")
    strat_bias = b2(tsv, "per stratum", "trait_bias_sd")
    assert (m3(none_bias), m3(first_bias), m3(strat_bias)) == (
        "+0.126", f"{MINUS}0.034", f"{MINUS}0.023")
    for name in ("studies",):
        for cell in ("+0.126", m3(first_bias), m3(strat_bias)):
            assert cell in text[name], (name, cell)
    for cell in (s3(none_bias), s3(first_bias), s3(strat_bias)):
        assert cell in text["tex"], cell
    ess_none = b2(tsv, "none", "kish_ess")
    assert f"{ess_none:.0f}" == "1756"
    assert "1756" in text["studies"] and "1756" in text["tex"]
    rung_ess = [b2(tsv, r, "kish_ess")
                for r in ("K(Y1)", "K(Y1), K(Y2)", "+ co-occurrence", "per stratum")]
    lo, hi = f"{min(rung_ess):.0f}", f"{max(rung_ess):.0f}"
    assert (lo, hi) == ("1500", "1520")
    assert f"{lo}--{hi}" in text["tex"]
    comorbid_none = b2(tsv, "none", "comorbidity_error")
    assert s3(comorbid_none) == "+0.019" and "+0.019" in text["tex"]


def b3(tsv, delta, metric, estimator="ipw+cal"):
    return mean(tsv, "B3_target_error", f"{delta:+.2f}", estimator, metric)


def test_b3_sweep_claims(text, tsv):
    lo, hi = b3(tsv, -0.30, "trait_bias_sd"), b3(tsv, 0.30, "trait_bias_sd")
    assert (m3(lo), m3(hi)) == (f"{MINUS}0.066", f"{MINUS}0.002")
    for name in ("studies",):
        assert f"{m3(lo)} to {m3(hi)}" in text[name]
    assert f"{s3(lo)} to {s3(hi)}" in text["tex"]
    slope = (hi - lo) / 6.0  # per +10 percentage points of relative error
    assert f"{slope:.3f}" == "0.011"
    assert "0.011 SD per 10 percentage points" in text["studies"]
    assert "0.011 SD per ten percentage points" in text["tex"]
    assert "0.011 SD per 10 percentage points" in text["readme"]
    naive = b1(tsv, "X + Y", "naive")
    assert u3(naive) == "0.479"
    for name in ("studies", "readme", "tex"):
        assert "0.479" in text[name], name
    prev_lo = b3(tsv, -0.30, "unanchored_prevalence_error_pct")
    prev_hi = b3(tsv, 0.30, "unanchored_prevalence_error_pct")
    assert f"+{prev_lo:.2f}%" in text["tex"] and f"+{prev_hi:.2f}%" in text["tex"]
    ess_lo = b3(tsv, -0.30, "kish_ess")
    ess_hi = b3(tsv, 0.30, "kish_ess")
    assert f"{ess_lo:.0f} to {ess_hi:.0f}" in text["tex"]


# --------------------------------------------------------------------------------
# B7 and B8: quoted only in the report
# --------------------------------------------------------------------------------

def b7(tsv, arm, ridge, metric, estimator=None):
    est = estimator or ("cal" if arm == "correct" else "ipw+cal")
    return mean(tsv, "B7_shrinkage", f"{arm} | ridge={ridge}", est, metric)


def test_b7_claims(text, tsv):
    doc = text["tex"]
    rmse0 = b7(tsv, "correct", "0", "trait_rmse_sd")
    rmse1 = b7(tsv, "correct", "1", "trait_rmse_sd")
    assert f"{rmse0:.4f}" == "0.0232" and f"{rmse1:.3f}" == "0.246"
    assert f"{rmse0:.4f}" in doc and f"{rmse1:.3f}" in doc
    bias0 = b7(tsv, "correct", "0", "trait_abs_bias_sd")
    bias1 = b7(tsv, "correct", "1", "trait_abs_bias_sd")
    assert f"{bias0:.3f}" == "0.003" and f"{bias1:.3f}" == "0.244"
    assert f"{bias0:.3f}" in doc and f"{bias1:.3f}" in doc
    for ridge, want_spread in (("0", "0.023"), ("1", "0.035")):
        rmse = b7(tsv, "correct", ridge, "trait_rmse_sd")
        signed = b7(tsv, "correct", ridge, "trait_bias_sd")
        spread = math.sqrt(max(rmse ** 2 - signed ** 2, 0.0))
        assert f"{spread:.3f}" == want_spread, (ridge, spread)
    assert f"{b7(tsv, 'base + margin', '0.03', 'trait_rmse_sd'):.3f}" == "0.041"
    assert f"{b7(tsv, 'base + margin', '0', 'trait_rmse_sd'):.3f}" == "0.054"
    assert f"{b7(tsv, 'misspecified', '0.3', 'trait_rmse_sd'):.3f}" == "0.055"
    assert f"{b7(tsv, 'misspecified', '0', 'trait_rmse_sd'):.3f}" == "0.106"
    assert f"{b7(tsv, 'misspecified', '0', 'trait_abs_bias_sd'):.3f}" == "0.083"
    assert f"{b7(tsv, 'misspecified', '0.3', 'trait_abs_bias_sd'):.3f}" == "0.020"
    for needle in ("0.041", "0.054", "0.055", "0.106", "0.083", "0.020"):
        assert needle in doc, needle
    ess0, ess1 = b7(tsv, "correct", "0", "kish_ess"), b7(tsv, "correct", "1", "kish_ess")
    assert f"{ess0:.0f}" == "1911" and f"{ess1:.0f}" == "2484"
    assert f"{ess0:.0f}" in doc and f"{ess1:.0f}" in doc
    ratio_03 = b7(tsv, "correct", "0.3", "trait_rmse_sd") / rmse0
    ratio_1 = rmse1 / rmse0
    assert 7.5 <= ratio_03 <= 8.5 and "eight times worse" in doc
    assert 9.5 <= ratio_1 <= 11.5 and "ten times worse" in doc


def b8(tsv, scenario, delta, metric="trait_bias_sd"):
    condition = f"{scenario} | delta={delta:+.2f}"
    return mean(tsv, "B8_k_error", condition,
                sole_estimator(tsv, "B8_k_error", condition), metric)


B8_SCENARIOS = ("pooled / X + Y", "stratified / X + Y", "pooled / X x Y",
                "pooled / stratum-differential", "stratified / stratum-differential",
                "pooled / severity", "severity / severity")


def test_b8_claims(text, tsv):
    doc = text["tex"]
    slopes = [(b8(tsv, s, 0.20) - b8(tsv, s, -0.20)) / 0.4 for s in B8_SCENARIOS]
    lo, hi = f"{min(slopes):.3f}", f"{max(slopes):.3f}"
    assert (lo, hi) == ("0.096", "0.133")
    assert f"+{lo}" in doc and f"+{hi}" in doc
    strat0 = b8(tsv, "stratified / stratum-differential", 0.0)
    pooled0 = b8(tsv, "pooled / stratum-differential", 0.0)
    assert f"{strat0:.3f}" == "-0.014" and f"{pooled0:.3f}" == "-0.045"
    assert "-0.014" in doc and "-0.045" in doc
    xxy0 = b8(tsv, "pooled / X x Y", 0.0)
    assert f"{xxy0:.3f}" == "-0.083" and "-0.083" in doc


# --------------------------------------------------------------------------------
# Counts, runtimes, and the stale strings that must never come back
# --------------------------------------------------------------------------------

def test_benchmark_counts_and_runtimes(text):
    assert "Nine benchmarks" in text["studies"]
    assert "nine benchmarks" in text["readme"]
    assert "~1 hour" in text["readme"]


@pytest.mark.parametrize("name, needles", [
    ("readme", ["11\u00d7 worse", "Seven benchmarks", "~17 min", "0.169 against"]),
    ("studies", ["11\u00d7 worse", "0.960", "3.4%", "0.169", "Seven benchmarks",
                 "+0.125 to"]),
    ("guide", ["0.960", "0.053 and 0.061", "3\u00d7 the bias", "0.169"]),
    ("theory", [f"{MINUS}0.044 to {MINUS}0.016", "+0.063 against +0.056"]),
    ("tex", ["3.4%", "twenty times worse", "within 0.001 SD", "18.4",
            "0.169", "order of magnitude in prevalence"]),
])
def test_superseded_numbers_stay_out(text, name, needles):
    """Regression guard for the 0.3.2 incident: values from the superseded first
    freeze (and the claims built on them) must not reappear while the current
    artifact is the one on disk."""
    for needle in needles:
        assert needle not in text[name], (name, needle)

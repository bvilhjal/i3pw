"""Tests for the benchmark suite.

The suite is not part of the installed package, but it produces the numbers the
report quotes, so its scaffolding needs the same guarantees as the package: a
simulated population whose channels do what they are documented to do, an estimator
zoo whose members are what their names say, and a figure generator that fails loudly
rather than plotting a number it could not find.

These are contract tests, not statistical ones. They use small populations and a
couple of seeds; the evidential claims live in the artifact, not here.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from benchmarks import estimators as E
from benchmarks import make_figures
from benchmarks.harness import Row, environment, proportion, read_tsv, rmse, summarize, write_tsv
from benchmarks.simulate import SELECTION_LAWS, Design, simulate

SMALL = dict(population_size=4000, n_features=6, expected_participants=800)


# --------------------------------------------------------------------------------
# The simulated population
# --------------------------------------------------------------------------------

def test_prevalence_and_participation_hit_their_targets():
    pop = simulate(Design(seed=0, prevalence=0.08, second_prevalence=0.03, **SMALL))
    assert pop.population_prevalence[0] == pytest.approx(0.08, abs=5e-4)
    assert pop.population_prevalence[1] == pytest.approx(0.03, abs=5e-4)
    # Participation is Bernoulli, so the realized count fluctuates around 800.
    assert pop.n_participants == pytest.approx(800, rel=0.15)
    assert np.all((pop.pi > 0) & (pop.pi < 1))


def test_outcome_channel_enriches_cases_and_covariate_channel_does_not():
    outcome_driven = simulate(Design(seed=1, delta_y=2.0, **SMALL))
    neutral = simulate(Design(seed=1, **SMALL))
    enriched = outcome_driven.Y[outcome_driven.mask, 0].mean()
    flat = neutral.Y[neutral.mask, 0].mean()
    assert enriched > flat * 1.5
    assert flat == pytest.approx(neutral.population_prevalence[0], abs=0.02)


def test_severity_channel_selects_the_more_ill_cases():
    mild = simulate(Design(seed=2, delta_y=1.7, **SMALL))
    severe = simulate(Design(seed=2, delta_y=1.7, delta_sev=1.2, **SMALL))
    share = [float(p.severe[p.mask][p.Y[p.mask, 0] == 1].mean()) for p in (mild, severe)]
    assert share[1] > share[0]


def test_severity_columns_partition_the_cases():
    pop = simulate(Design(seed=3, **SMALL))
    mild, sev = pop.severity_columns().T
    assert np.array_equal(mild + sev, pop.Y[:, 0])
    assert np.all(sev <= pop.Y[:, 0])


def test_stratum_prevalence_varies_and_shares_sum_to_one():
    pop = simulate(Design(seed=4, **SMALL))
    within = pop.within_stratum_prevalence()[:, 0]
    assert within.max() > 2 * within.min()
    assert pop.stratum_share.sum() == pytest.approx(1.0)


def test_every_documented_selection_law_runs_and_biases_the_sample():
    for law, channels in SELECTION_LAWS.items():
        pop = simulate(Design(seed=5, **channels, **SMALL))
        naive = pop.Z[pop.mask].mean()
        assert abs(naive - pop.trait_mean) / pop.trait_sd > 0.1, law


# --------------------------------------------------------------------------------
# The estimator zoo
# --------------------------------------------------------------------------------

def test_naive_is_uniform_and_oracle_is_inverse_pi():
    pop = simulate(Design(seed=6, delta_y=1.7, **SMALL))
    naive = E.fit_weighting(pop, "naive")
    assert np.allclose(naive.weights, naive.weights[0])
    oracle = E.fit_weighting(pop, "oracle")
    ratio = oracle.weights * pop.pi[pop.mask]
    assert np.allclose(ratio, ratio[0])


def test_calibration_reproduces_the_prevalence_it_was_given():
    pop = simulate(Design(seed=7, delta_y=1.7, **SMALL))
    wt = E.fit_weighting(pop, "cal")
    achieved = E.weighted_mean(pop.Y[pop.mask, 0], wt.weights)
    assert achieved == pytest.approx(pop.population_prevalence[0], abs=1e-8)


def test_stratified_calibration_matches_every_within_stratum_prevalence():
    pop = simulate(Design(seed=8, delta_y=1.7, delta_y_stratum=1.0, **SMALL))
    wt = E.fit_weighting(pop, "ipw+cal/s")
    mask = pop.mask
    levels = len(pop.stratum_share)
    dummies = (pop.stratum[mask][:, None] == np.arange(levels)[None, :]).astype(float)
    w = wt.weights[:, None]
    achieved = (w * dummies * pop.Y[mask, 0][:, None]).sum(0) / (w * dummies).sum(0)
    assert np.allclose(achieved, pop.within_stratum_prevalence()[:, 0], atol=1e-6)


def test_a_perturbed_target_is_the_one_that_is_matched():
    pop = simulate(Design(seed=9, delta_y=1.7, **SMALL))
    wrong = np.array([pop.population_prevalence[0] * 1.2])
    wt = E.fit_weighting(pop, "ipw+cal", targets=wrong)
    assert E.weighted_mean(pop.Y[pop.mask, 0], wt.weights) == pytest.approx(wrong[0], abs=1e-8)


def test_unreachable_target_is_reported_as_a_failed_solve():
    pop = simulate(Design(seed=10, prevalence=0.05, **SMALL))
    mask = pop.mask
    pop.Y[mask, 0] = 0  # no sampled cases: no reweighting reaches a positive prevalence
    wt = E.fit_weighting(pop, "cal")
    assert not wt.converged


def test_unknown_method_is_refused():
    pop = simulate(Design(seed=11, **SMALL))
    with pytest.raises(ValueError, match="unknown method"):
        E.fit_weighting(pop, "magic")


def test_metrics_are_zero_for_a_weighting_that_recovers_the_population():
    pop = simulate(Design(seed=12, delta_y=1.7, **SMALL))
    oracle = E.fit_weighting(pop, "oracle")
    naive = E.fit_weighting(pop, "naive")
    assert abs(E.trait_error(pop, oracle)) < abs(E.trait_error(pop, naive))
    assert oracle.ess <= naive.ess + 1e-9   # correcting always costs precision


# --------------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------------

def test_summarize_reports_spread_and_monte_carlo_error_separately():
    row = summarize("B", "c", "e", "m", [1.0, 2.0, 3.0])
    assert row.mean == pytest.approx(2.0)
    assert row.sd == pytest.approx(1.0)
    assert row.mcse == pytest.approx(1.0 / np.sqrt(3))
    assert row.n_reps == 3


def test_summarize_drops_non_finite_replications():
    row = summarize("B", "c", "e", "m", [1.0, float("nan"), 3.0, None])
    assert row.n_reps == 2
    assert row.mean == pytest.approx(2.0)


def test_proportion_uses_the_binomial_error_and_leaves_sd_empty():
    row = proportion("B", "c", "e", "m", [True, True, True, False])
    assert row.mean == pytest.approx(0.75)
    assert row.sd is None
    assert row.mcse == pytest.approx(np.sqrt(0.75 * 0.25 / 4))


def test_rmse_penalizes_spread_as_well_as_bias():
    assert rmse([1.0, -1.0]) == pytest.approx(1.0)
    assert rmse([0.0, 0.0]) == pytest.approx(0.0)


def test_tsv_round_trips_and_writes_na_for_missing_values(tmp_path):
    rows = [Row("B", "c", "e", "m", 1.5, None, None, 4, "note"),
            summarize("B", "c", "e", "m2", [1.0, 2.0])]
    path = write_tsv(rows, tmp_path / "out.tsv")
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("benchmark\tcondition")
    assert "\tNA\tNA\t" in text
    back = read_tsv(path)
    assert len(back) == 2
    assert back[0]["sd"] is None and back[0]["mean"] == pytest.approx(1.5)
    assert back[1]["n_reps"] == 2


def test_environment_records_the_versions_a_rerun_would_need():
    text = environment(quick=False, wall_seconds=1.0, n_rows=10)
    for key in ("i3pw_version", "numpy_version", "scipy_version",
                "scikit_learn_version", "python_version"):
        assert f"{key}=" in text


# --------------------------------------------------------------------------------
# The figure generator
# --------------------------------------------------------------------------------

def test_results_lookup_names_the_row_it_could_not_find(tmp_path):
    path = write_tsv([summarize("B1", "cond", "est", "metric", [1.0, 2.0])],
                     tmp_path / "r.tsv")
    res = make_figures.Results(path)
    assert res.mean("B1", "cond", "est", "metric") == pytest.approx(1.5)
    with pytest.raises(KeyError, match="no row for"):
        res.mean("B1", "cond", "est", "absent")


def test_conditions_keep_the_order_they_were_written_in(tmp_path):
    rows = [summarize("B1", c, "e", "m", [1.0]) for c in ("z", "a", "m")]
    res = make_figures.Results(write_tsv(rows, tmp_path / "r.tsv"))
    assert res.conditions("B1") == ["z", "a", "m"]


def test_offsets_are_symmetric_about_the_tick():
    assert make_figures.offsets(1) == [0.0]
    shifts = make_figures.offsets(4)
    assert shifts[0] == pytest.approx(-shifts[-1])
    assert sum(shifts) == pytest.approx(0.0)


def test_emitted_plots_carry_their_style_and_coordinates():
    out = make_figures.series("viz ipw", [(0, 1.0, 0.2)], errors=True, label="ipw")
    assert "viz ipw" in out and "(0,1) +- (0,0.2)" in out
    assert "\\addlegendentry{ipw}" in out
    assert "forget plot" in make_figures.series("viz cal", [(0, 1.0)])


def test_non_finite_coordinates_are_refused():
    with pytest.raises(ValueError, match="non-finite"):
        make_figures.series("viz ipw", [(0, float("nan"))])


@pytest.mark.skipif(not make_figures.RESULTS_TSV.exists(),
                    reason="benchmark artifact not present")
def test_every_report_display_builds_from_the_frozen_artifact():
    """Every table and figure the report inputs still finds all of its rows.

    This is the drift guard: a benchmark whose conditions were renamed, or a metric
    that stopped being written, fails here rather than in a LaTeX run.
    """
    res = make_figures.Results()
    for name, build in make_figures.FIGURES.items():
        body = build(res)
        expected = "\\begin{tabular}" if name.startswith("tab-") else "\\begin{tikzpicture}"
        assert expected in body, name
        # A missing summary formats as "nan"; the word boundaries keep the check off
        # ordinary prose such as "unanchored".
        assert not re.search(r"(?<![a-z])nan(?![a-z])", body.lower()), name


@pytest.mark.skipif(not make_figures.RESULTS_TSV.exists(),
                    reason="benchmark artifact not present")
def test_committed_displays_match_the_committed_artifact():
    """What is on disk in report/figures is what the artifact currently produces."""
    res = make_figures.Results()
    for name, build in make_figures.FIGURES.items():
        path = make_figures.FIGURE_DIR / name
        assert path.exists(), f"{name} has never been generated"
        assert path.read_text(encoding="utf-8") == make_figures.HEADER + build(res), (
            f"{name} is stale: rerun `python -m benchmarks.make_figures`"
        )

"""B9 --- what the shipped calls cost in wall-clock seconds, as a function of size.

The report's computational section has until now stated algorithmic orders only.
This benchmark times the calls a user actually makes -- one
:func:`i3pw.calibration_ipw` fit, one ``B=200`` bootstrap with the base model held
fixed (the default), and one :func:`i3pw.compute_base_weights` LASSO base fit --
over a 16x range of population size, and writes them to their own artifact. It is
a separate TSV on purpose: every statistical row in ``benchmark_results.tsv``
reproduces bit-exactly on the recorded environment, and wall-clock seconds never
do, so the two claims must not share a file.

The base-fit row is what prices ``refit_base=True``. A bootstrap that refits the
LASSO per replicate costs roughly the fixed-base bootstrap plus ``B`` base fits,
and a base fit is orders of magnitude dearer than a dual solve -- measuring the
component is feasible where measuring ``B=200`` refits end to end is not (a
single refit replicate costs seconds, so the full thing would run to tens of
minutes at the smallest size and hours at the largest). Read the row as the
per-replicate premium, multiply by ``B``.

Timings are only honest on a machine that is plugged in and not throttling, so
on macOS the module refuses to run on battery power or in Low Power Mode (the
same guard the family's benchmark runners use). Off macOS there is no ``pmset``
to ask and the check is skipped.
"""

from __future__ import annotations

import subprocess
import sys
import time
import warnings
from pathlib import Path

from benchmarks.harness import (
    TIMING_ENV_TXT,
    TIMING_TSV,
    Row,
    environment,
    summarize,
    write_tsv,
)
from i3pw import bootstrap_calibration_ipw, calibration_ipw, compute_base_weights, make_dataset

BENCHMARK = "B9_wall_clock"

SIZES = (5_000, 20_000, 80_000)
"""Population sizes; the ascertained sample is 20% of each, split 75/25 inside
the generator, so the mid row is the n=20k / sample 4k regime the profiling
question is usually asked about."""

SAMPLE_SHARE = 0.20
N_BOOT = 200

REPS: dict[int, int] = {5_000: 3, 20_000: 2, 80_000: 2}
"""Timed repetitions per size; fewer where one LASSO fit costs tens of seconds
(measured: ~65 s at N=20k, ~155 s at N=80k on the freeze environment, against
~0.2 s for all 200 bootstrap dual solves together)."""

CONFIG = dict(n_outcomes=2, predictors_per_outcome=10,
              target_population_prevalence=(0.10, 0.05))


def power_state() -> tuple[bool, str]:
    """``(ok, detail)``; ``ok`` is False on battery or in macOS Low Power Mode."""
    if sys.platform != "darwin":
        return True, "not darwin: no pmset to consult, power state unchecked"
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                          text=True, check=True).stdout
    if "AC Power" not in batt:
        return False, "drawing from battery"
    profile = subprocess.run(["pmset", "-g"], capture_output=True,
                             text=True, check=True).stdout
    for line in profile.splitlines():
        parts = line.split()
        if parts and parts[0] == "lowpowermode" and parts[-1] == "1":
            return False, "macOS Low Power Mode is on"
    return True, "on AC power"


def run(*, sizes=SIZES, n_boot: int = N_BOOT, n_reps: int | None = None,
        power: str = "check") -> list[Row]:
    """Time the shipped fit, bootstrap and base-fit calls at each size.

    ``n_reps`` overrides the per-size default in :data:`REPS` for every size.
    ``power='skip'`` bypasses the battery/Low-Power-Mode guard and exists for the
    contract tests, which are not evidence and must not be refused by hardware
    state.
    """
    if power == "check":
        ok, detail = power_state()
        if not ok:
            raise RuntimeError(
                f"{BENCHMARK} refuses to time on this machine: {detail}. Timings "
                "taken now would understate the cost and end up in a frozen "
                "artifact; plug in and rerun."
            )
    rows: list[Row] = []
    for n_pop in sizes:
        n_sample = int(round(n_pop * SAMPLE_SHARE))
        condition = f"N={n_pop}, n={n_sample}"
        reps = n_reps if n_reps is not None else REPS.get(n_pop, 3)
        timings: dict[str, list[float]] = {"fit": [], "boot": [], "base": []}
        for rep in range(reps):
            dataset = make_dataset(population_size=n_pop, sample_size=n_sample,
                                   seed=1000 + rep, **CONFIG)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                calibration_ipw(dataset)
                timings["fit"].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                bootstrap_calibration_ipw(dataset, n_boot=n_boot, refit_base=False)
                timings["boot"].append(time.perf_counter() - t0)

                X_train, _, s_train = dataset.split("train")
                X_test, _, s_test = dataset.split("test")
                t0 = time.perf_counter()
                compute_base_weights("lasso", "inverse", X_train, s_train,
                                     X_test[s_test == 1])
                timings["base"].append(time.perf_counter() - t0)
        rows.append(summarize(BENCHMARK, condition, "calibration_ipw", "seconds",
                              timings["fit"],
                              notes="one fit: LASSO base plus the dual solve"))
        rows.append(summarize(BENCHMARK, condition,
                              f"bootstrap_calibration_ipw B={n_boot}", "seconds",
                              timings["boot"],
                              notes="base weights held fixed (the default)"))
        rows.append(summarize(BENCHMARK, condition, "compute_base_weights", "seconds",
                              timings["base"],
                              notes="one LASSO base fit: the per-replicate premium "
                                    "when refit_base=True"))
    return rows


def write(rows: list[Row], *, quick: bool, wall_seconds: float) -> Path:
    """Write the timing artifact and its provenance. Overwrites: a whole run."""
    if quick:
        path = TIMING_TSV.with_name("timing_results_quick.tsv")
        env = TIMING_ENV_TXT.with_name("timing_environment_quick.txt")
    else:
        path, env = TIMING_TSV, TIMING_ENV_TXT
    write_tsv(rows, path)
    env.write_text(environment(quick=quick, wall_seconds=wall_seconds, n_rows=len(rows),
                               artifact=f"report/{path.name}"), encoding="utf-8")
    return path


def main() -> None:
    t0 = time.perf_counter()
    rows = run()
    path = write(rows, quick=False, wall_seconds=time.perf_counter() - t0)
    print(f"wrote {len(rows)} rows to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()

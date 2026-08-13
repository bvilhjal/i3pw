"""Run every benchmark and write the artifact the report and figures read.

    python -m benchmarks.run_all            # the full freeze (tens of minutes)
    python -m benchmarks.run_all --quick    # a few replications, for a smoke test
    python -m benchmarks.run_all B1 B5      # only the named benchmarks

Every seed is fixed, so a full run on the environment recorded in
``report/benchmark_environment.txt`` reproduces ``report/benchmark_results.tsv``
exactly. A ``--quick`` run is for checking that the code executes; its Monte Carlo
error is far too large to quote, and it writes to a separate file so it cannot be
mistaken for the freeze.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from benchmarks import (
    b1_selection_laws,
    b2_anchor_information,
    b3_target_error,
    b4_case_mix,
    b5_interval_coverage,
    b6_support,
    b7_shrinkage,
)
from benchmarks.harness import ENVIRONMENT_TXT, RESULTS_TSV, environment, write_tsv

# (key, module, full-run kwargs, quick-run kwargs)
BENCHMARKS = (
    ("B1", b1_selection_laws, dict(n_reps=40), dict(n_reps=3)),
    ("B2", b2_anchor_information, dict(n_reps=40), dict(n_reps=3)),
    ("B3", b3_target_error, dict(n_reps=40), dict(n_reps=3)),
    ("B4", b4_case_mix, dict(n_reps=40), dict(n_reps=3)),
    ("B5", b5_interval_coverage, dict(n_reps=300, n_boot=400),
     dict(n_reps=4, n_boot=40)),
    ("B6", b6_support, dict(n_reps=30, n_boot=200), dict(n_reps=2, n_boot=20)),
    ("B7", b7_shrinkage, dict(n_reps=40), dict(n_reps=3)),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("only", nargs="*", help="benchmark keys to run (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="few replications; a smoke test, not evidence")
    parser.add_argument("--out", type=Path, default=None,
                        help="destination TSV (default: report/benchmark_results.tsv)")
    args = parser.parse_args(argv)

    selected = [b for b in BENCHMARKS if not args.only or b[0] in args.only]
    if not selected:
        parser.error(f"no benchmark matched {args.only}; keys are "
                     f"{[b[0] for b in BENCHMARKS]}")

    out = args.out or (RESULTS_TSV.with_name("benchmark_results_quick.tsv")
                       if args.quick else RESULTS_TSV)
    start = time.time()
    rows = []
    for key, module, full_kwargs, quick_kwargs in selected:
        print(f"[{key}] {module.BENCHMARK}", file=sys.stderr)
        rows.extend(module.run(**(quick_kwargs if args.quick else full_kwargs)))

    wall = time.time() - start
    write_tsv(rows, out)
    print(f"\nwrote {len(rows)} rows to {out} in {wall:.0f}s", file=sys.stderr)

    if not args.quick and not args.only:
        ENVIRONMENT_TXT.write_text(
            environment(quick=False, wall_seconds=wall, n_rows=len(rows)), encoding="utf-8"
        )
        print(f"wrote provenance to {ENVIRONMENT_TXT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

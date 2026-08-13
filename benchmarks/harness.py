"""Shared plumbing: one result row shape, one summary rule, one provenance record.

Every benchmark writes rows of the same nine columns into one TSV, so the report,
the figures and the documentation all read the same artifact and cannot drift from
each other. The figure generator refuses to plot a row it cannot find, which is the
point: a number in the PDF exists because it is in the artifact.

Two summary rules, because two kinds of quantity are reported:

``summarize``    a continuous per-replication quantity (a bias, an error, an ESS).
                 ``sd`` is the spread across replications -- the thing an
                 analyst would see rerunning on another population -- and ``mcse``
                 is ``sd / sqrt(R)``, the precision of the reported mean itself.
``proportion``   a per-replication indicator (did the interval cover? did the solve
                 fail?). ``sd`` is left NA and ``mcse`` is the binomial
                 ``sqrt(p(1-p)/R)``, which is what a coverage claim must be read
                 against: 200 replications cannot distinguish 0.95 from 0.93.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_TSV = REPO_ROOT / "report" / "benchmark_results.tsv"
ENVIRONMENT_TXT = REPO_ROOT / "report" / "benchmark_environment.txt"

COLUMNS = (
    "benchmark", "condition", "estimator", "metric",
    "mean", "sd", "mcse", "n_reps", "notes",
)


@dataclass
class Row:
    """One reported number, with the spread and the Monte Carlo error around it."""

    benchmark: str
    condition: str
    estimator: str
    metric: str
    mean: float
    sd: float | None
    mcse: float | None
    n_reps: int
    notes: str = ""

    def as_tsv(self) -> str:
        d = asdict(self)
        return "\t".join(_format(d[c]) for c in COLUMNS)


def _format(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def summarize(
    benchmark: str, condition: str, estimator: str, metric: str,
    values, *, notes: str = "",
) -> Row:
    """Mean, across-replication SD and Monte Carlo SE of a continuous quantity."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Row(benchmark, condition, estimator, metric,
                   float("nan"), None, None, 0, notes)
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    mcse = sd / np.sqrt(v.size) if v.size > 1 else float("nan")
    return Row(benchmark, condition, estimator, metric,
               float(v.mean()), sd, float(mcse), int(v.size), notes)


def proportion(
    benchmark: str, condition: str, estimator: str, metric: str,
    indicators, *, notes: str = "",
) -> Row:
    """Rate of a per-replication indicator, with its binomial Monte Carlo error."""
    v = np.asarray([bool(x) for x in indicators], dtype=float)
    if v.size == 0:
        return Row(benchmark, condition, estimator, metric,
                   float("nan"), None, None, 0, notes)
    p = float(v.mean())
    return Row(benchmark, condition, estimator, metric,
               p, None, float(np.sqrt(max(p * (1 - p), 0.0) / v.size)), int(v.size), notes)


def rmse(values) -> float:
    """Root mean squared error of per-replication signed errors."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v**2))) if v.size else float("nan")


def write_tsv(rows: list[Row], path: Path = RESULTS_TSV) -> Path:
    """Write all rows, header first. Overwrites: the artifact is a whole run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(row.as_tsv() for row in rows)
    path.write_text("\t".join(COLUMNS) + "\n" + body + "\n", encoding="utf-8")
    return path


def read_tsv(path: Path = RESULTS_TSV) -> list[dict]:
    """Read a results TSV back as dicts, numbers parsed, ``NA`` as ``None``."""
    # splitlines rather than strip().split(): a row whose trailing `notes` field is
    # empty ends in a tab, and stripping the file would eat that last column on the
    # final line, leaving it one field short.
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    header = lines[0].split("\t")
    out = []
    for line in lines[1:]:
        record = dict(zip(header, line.split("\t"), strict=True))
        for key in ("mean", "sd", "mcse"):
            record[key] = None if record[key] == "NA" else float(record[key])
        record["n_reps"] = int(record["n_reps"])
        out.append(record)
    return out


def environment(*, quick: bool, wall_seconds: float, n_rows: int) -> str:
    """Provenance for the artifact: versions, platform, and how it was produced."""
    import scipy
    import sklearn

    import i3pw

    stamp = time.strftime("%Y-%m-%d")
    return "\n".join([
        "artifact=report/benchmark_results.tsv",
        "generator=benchmarks/run_all.py",
        f"run_date={stamp}",
        f"mode={'quick' if quick else 'full'}",
        f"rows={n_rows}",
        f"wall_seconds={wall_seconds:.1f}",
        f"i3pw_version={i3pw.__version__}",
        f"python_version={platform.python_version()}",
        f"numpy_version={np.__version__}",
        f"scipy_version={scipy.__version__}",
        f"scikit_learn_version={sklearn.__version__}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        "note=every seed is fixed; rerunning the generator on this environment "
        "reproduces the table exactly",
    ]) + "\n"


class Progress:
    """Minimal progress reporting, so a long sweep is not a silent terminal."""

    def __init__(self, label: str, total: int) -> None:
        self.label, self.total, self.done = label, total, 0
        self.start = time.time()
        print(f"  {label}: 0/{total}", end="", flush=True, file=sys.stderr)

    def step(self, n: int = 1) -> None:
        self.done += n
        print(f"\r  {self.label}: {self.done}/{self.total}"
              f"  ({time.time() - self.start:.0f}s)", end="", flush=True, file=sys.stderr)

    def close(self) -> None:
        print(f"\r  {self.label}: {self.done}/{self.total}"
              f"  done in {time.time() - self.start:.0f}s", file=sys.stderr)

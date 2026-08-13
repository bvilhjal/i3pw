# Benchmark suite

The evidence base for [the report](../report/i3pw_report.pdf). Seven benchmarks,
one per axis of the validation matrix the report asks for, all writing rows into
one artifact — [`report/benchmark_results.tsv`](../report/benchmark_results.tsv) —
which the report's tables and figures both read. Nothing in the PDF is retyped from
a terminal.

This is **not part of the installed package**. `i3pw` ships the estimator; this
directory is the machinery that tries to break it.

```bash
python -m benchmarks.run_all          # the full freeze (~20 min)
python -m benchmarks.run_all --quick  # smoke test; too noisy to quote
python -m benchmarks.run_all B5 B7    # just those two
python -m benchmarks.make_figures     # regenerate report/figures/fig-*.tex
```

Seeds are fixed throughout, so a full run on the environment recorded in
[`report/benchmark_environment.txt`](../report/benchmark_environment.txt)
reproduces the artifact exactly.

## What each one asks

| | question | what can fail |
| --- | --- | --- |
| **B1** `selection_laws` | which estimator recovers a held-out estimand, as the recruitment mechanism changes from covariate-driven to outcome-driven to neither | bias against the oracle weights `1/π`, held-out balance, ESS |
| **B2** `anchor_information` | how much each additional register quantity buys, from none to per-stratum prevalences | bias on a common held-out trait; ESS as the price |
| **B3** `target_error` | what a mis-stated register prevalence costs, from −30% to +30% | transfer of the error to unanchored quantities |
| **B4** `case_mix` | whether a matched prevalence implies the right *kind* of case | case mix and trait bias, both held out in every arm |
| **B5** `interval_coverage` | do the intervals cover — fixed-weight, calibration-aware, and bootstrap, under three specifications | coverage against nominal 0.95, with binomial Monte Carlo error |
| **B6** `support` | how rare the anchored disease can be before the solve, then the bootstrap, give out | solve failure, replicate discards, weight concentration |
| **B7** `shrinkage` | whether relaxing an exact constraint ever pays | RMSE against the exact solve, bias and spread separately |

## How it is put together

```
simulate.py    one population rich enough for every axis: a covariate channel, an
               outcome channel, their interaction, within-case severity,
               stratum-differential ascertainment, comorbid recruitment — and the
               true inclusion probabilities, so an oracle row is possible.
estimators.py  the estimator zoo and the metrics, defined once so that a row
               labelled `ipw+cal` means the same thing in every benchmark.
harness.py     the nine-column row, the two summary rules (continuous quantities
               get an across-replication SD; indicators get a binomial Monte Carlo
               error), the TSV writer and the provenance record.
b1..b7         one file per benchmark, each with a module docstring stating what
               it is asking and why the answer is not already known.
make_figures.py  reads the artifact, writes report/figures/fig-*.tex.
run_all.py     runs everything and writes the artifact and its environment file.
```

Two rules hold everywhere, and they are the reason the numbers are worth reading:

- **A constrained quantity is never reported as a result.** Calibration reproduces
  its own targets whatever the truth. Where such a number appears in a table it is
  labelled an identity, and it is never plotted.
- **Every headline metric is held out.** The estimand throughout is the population
  mean of a continuous trait that is observed only in participants and is never a
  calibration constraint, alongside a second disease whose prevalence is never
  supplied. Those are quantities the estimator can get wrong.

Contract tests live in [`tests/test_benchmarks.py`](../tests/test_benchmarks.py):
that the simulated channels do what their names say, that each estimator is what it
claims, and that the figure generator raises rather than plotting a number it could
not find.

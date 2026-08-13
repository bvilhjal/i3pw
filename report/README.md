# Technical report

`i3pw_report.pdf` (also copied to `output/pdf/`) is a methods note:
estimand, identifying density-ratio family, and the simulation claims
that survive their design.

## The two numeric artifacts

| file | what it holds | frozen at |
| --- | --- | --- |
| `validation_results.tsv` | the `examples/` scripts — the illustrations the README and `docs/studies.md` quote | i3pw 0.3.0, 11 August 2026 |
| `benchmark_results.tsv` | the `benchmarks/` suite — seven benchmarks over the report's validation matrix | i3pw 0.3.1, `benchmark_environment.txt` |

Every numeric display in the PDF is copied from one of them. Regenerate the
artifact, or keep citing its freeze, before changing a reported number.

```bash
python -m benchmarks.run_all        # rewrites benchmark_results.tsv (~20 min)
python -m benchmarks.make_figures   # rewrites report/figures/fig-*.tex
```

## Figures

`figures/fig-*.tex` are **generated** from the artifacts by
`benchmarks/make_figures.py` and must not be edited by hand — a figure that
disagrees with the table beside it is the failure mode the generator exists to
prevent. `figures/i3pw-viz.tex` is hand-written: it holds the colours, axis
chrome and per-estimator styles the generated files reference, so that an
estimator keeps its hue and marker across every figure in the document.

## Rebuild

Requires [Tectonic](https://tectonic-typesetting.github.io/) or `pdflatex`:

```bash
cd report
tectonic -X compile i3pw_report.tex
cp i3pw_report.pdf ../output/pdf/i3pw_report.pdf
```

The PDF is documentation, not a runtime dependency of the installed
package.

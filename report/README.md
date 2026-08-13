# Technical report

`i3pw_report.pdf` (also copied to `output/pdf/`) is a methods note:
estimand, identifying density-ratio family, and the simulation claims
that survive their design.

Numeric displays are copied from `validation_results.tsv`, frozen at
i3pw 0.3.0 on 11 August 2026. The present package version may be later.
Regenerate the TSV, or keep citing that freeze, before changing a
reported number.

Rebuild (requires [Tectonic](https://tectonic-typesetting.github.io/)
or `pdflatex`):

```bash
cd report
tectonic -X compile i3pw_report.tex
cp i3pw_report.pdf ../output/pdf/i3pw_report.pdf
```

The PDF is documentation, not a runtime dependency of the installed
package.

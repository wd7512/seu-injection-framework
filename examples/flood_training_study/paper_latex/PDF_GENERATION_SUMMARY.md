# PDF Generation Summary

## Status: CURRENT (March 2026 Rebuild)

The LaTeX source (`main.tex`), all 3 figures, and the compiled PDF have been updated with corrected experimental data from the March 2026 result refresh. The paper has been condensed to a clean, concise proof-of-concept format.

## Changes Since December 2025 Build

- Paper rewritten as concise proof-of-concept (reduced from ~8 pages to ~5-6)
- Figures reduced from 6 to 3 (removed cost-benefit, loss trajectories, training dynamics)
- All ROI metrics removed entirely
- All numerical claims updated to match re-run experimental data
- Old claims (6.5-14.2% improvement, 15.9x ROI, 0.41% cost) replaced with real data
- New findings: dataset-dependent effects, non-monotonic response, dropout 15.1%, bit-1 dominance
- Title changed to "Flood Level Training for SEU Robustness: A Proof-of-Concept Study"
- LaTeX build artifacts removed from git tracking

## Compilation Process

Full LaTeX compilation cycle:

1. `pdflatex main.tex` (first pass)
2. `bibtex main` (bibliography processing)
3. `pdflatex main.tex` (second pass - resolve citations)
4. `pdflatex main.tex` (third pass - final cross-references)

## Figures Embedded

3 figures at 300 DPI:

1. **Figure 1**: Robustness vs flood level (3-panel, one per dataset)
2. **Figure 2**: Training loss vs flood level (verifies flooding is active)
3. **Figure 3**: Heatmap of all 36 configurations (dataset x dropout x flood level)

## Paper Statistics

- **Length**: ~5-6 pages (including figures and references)
- **Sections**: 6 main sections + abstract
- **Figures**: 3 (all publication-quality, 300 DPI)
- **Tables**: 1 (main results summary)
- **References**: 6 (all verified)

## Regenerating PDF

```bash
cd examples/flood_training_study/paper_latex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Regenerating Figures

```bash
cd examples/flood_training_study
python generate_figures.py
```

---

**Original Generation Date**: 2025-12-12
**Current Rebuild Date**: 2026-03-15
**LaTeX Distribution**: MiKTeX
**Compiler**: pdfLaTeX
**Status**: Current

# Flood Level Training Paper (LaTeX)

## Title

**Flood Level Training for SEU Robustness: A Proof-of-Concept Study**

## Paper Structure

- `main.tex` - Complete paper in single LaTeX file
- `bibliography.bib` - Bibliography with all cited references
- `neurips_2024.sty` - NeurIPS style file
- `figures/` - 3 publication-quality figures (PNG, 300 DPI)

---

## Compilation

### Using pdflatex (Recommended)

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk

```bash
latexmk -pdf main.tex
latexmk -c  # clean intermediate files
```

### Using Overleaf

1. Create a new project on [Overleaf](https://www.overleaf.com)
2. Upload `main.tex`, `bibliography.bib`, and `neurips_2024.sty`
3. Upload all files from `figures/` directory
4. Set compiler to `pdfLaTeX`
5. Click "Recompile"

---

## Required LaTeX Packages

Uses the `neurips_2024` document class (preprint mode). Required packages:

- `inputenc`, `fontenc` - Character encoding
- `hyperref`, `url` - Hyperlinks and URLs
- `booktabs` - Professional tables
- `amsfonts`, `amsmath` - Mathematical symbols
- `nicefrac`, `microtype` - Typography
- `xcolor` - Colors
- `graphicx`, `subcaption` - Figures and subfigures

All packages are standard in modern LaTeX distributions (TeX Live, MiKTeX).

---

## Figures

### Figure 1: Robustness vs Flood Level

**File**: `figures/fig1_robustness_vs_flood.png`
- 3-panel line plot (one per dataset) showing mean accuracy drop vs flood level, comparing with/without dropout

### Figure 2: Training Loss vs Flood Level

**File**: `figures/fig2_training_loss.png`
- Bar chart of final training loss vs flood level, verifying that flooding actively constrains training

### Figure 3: Results Heatmap

**File**: `figures/fig3_heatmap.png`
- Heatmap of mean accuracy drop across all 36 configurations (3 datasets x 2 dropout x 6 flood levels)

---

## Key Results (Table 1)

- Standard training (b=0.0): 1.94% accuracy drop under SEU injection
- Optimal flooding (b=0.15): 1.75% accuracy drop (10.0% relative improvement, 0.50% accuracy cost)
- Dropout alone: 15.1% robustness improvement with negligible accuracy cost
- Effect is dataset-dependent: blobs benefits most (~49%), circles shows no benefit

---

## File Structure

```
paper_latex/
├── main.tex                          # Main paper
├── main.pdf                          # Compiled PDF
├── bibliography.bib                  # Citations
├── neurips_2024.sty                  # Style file
├── figures/
│   ├── fig1_robustness_vs_flood.png  # 3-panel robustness plot
│   ├── fig2_training_loss.png        # Training loss bar chart
│   └── fig3_heatmap.png              # 36-config heatmap
├── PDF_GENERATION_SUMMARY.md         # Build status
└── README.md                         # This file
```

---

## For Camera-Ready Version

- [ ] Add actual author names and affiliations
- [ ] Add GitHub repository link
- [ ] Run spell-check and grammar review
- [ ] Verify all DOIs and URLs
- [ ] Format check against venue requirements

---

**Last Updated**: 2026-03-15
**Status**: Current

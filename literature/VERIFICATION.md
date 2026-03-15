# Verification Protocol

## Verification Levels

| Badge | Meaning | Requirement |
|-------|---------|-------------|
| `arXiv` | Preprint only | URL accessible |
| `DOI` | Peer-reviewed | DOI resolves |

## Adding Papers

Each paper entry MUST include verification links:

### For arXiv papers:
- **Primary**: https://arxiv.org/abs/[ID]
- **PDF**: https://arxiv.org/pdf/[ID].pdf

### For DOI papers:
- **Primary**: https://doi.org/[DOI]
- **Venue**: IEEE Xplore / journal homepage

### For IEEE papers:
- **Primary**: https://ieeexplore.ieee.org/document/[ID]

## Process

1. Find primary verification URL (arXiv/DOI/IEEE)
2. Test URL accessibility
3. Add to paper entry under "Verification Links"
4. Add badge to bibliography.yaml

## Badge Display

Use in README index:
- `(arXiv)` for arXiv-only papers
- `(DOI)` for peer-reviewed papers

# Flood Level Training for SEU Robustness

**Research Study**: Comprehensive empirical investigation of flood level training's impact on neural network robustness to Single Event Upsets (SEUs).

**Authors**: SEU Injection Framework Research Team\
**Date**: December 2025

______________________________________________________________________

## Executive Summary

### Research Question

**Does flood level training improve neural network robustness to SEUs?**

Flood level training (Ishida et al., 2020) is a regularization technique that prevents models from achieving arbitrarily low training loss:

```python
L_flood = |L(θ) - b| + b
```

Where `b` is the flood level. We investigated whether this improves robustness to radiation-induced bit flips.

### Key Findings

**Comprehensive experiments** (36 configurations: 3 datasets × 6 flood levels × 2 dropout settings):

| Metric                       | Result                                   |
| ---------------------------- | ---------------------------------------- |
| **Robustness Improvement**   | Up to 10.0% avg (b=0.15), ~49% best config (blobs+dropout) |
| **Optimal Configuration**    | b=0.15 with dropout (20.0× ROI)          |
| **Accuracy Cost**            | 0.50% at optimal setting                 |
| **Dropout Alone**            | 15.1% robustness improvement (0.10% cost) |
| **Dominant Vulnerability**   | Bit 1 (exponent MSB) accounts for nearly all SEU impact |

**Important**: Flooding effectiveness is dataset-dependent — the flood level must exceed the model's natural training loss convergence point to be active. For tasks with high natural training loss (e.g., circles at ~0.43), flooding has no effect.

**Recommendation**: Use dropout (0.2) as a baseline. Add flood training (b=0.15) after verifying the flood level exceeds natural training loss. Deploy in safety-critical radiation environments.

______________________________________________________________________

## Research Paper Structure

Navigate through the sections:

### 📄 Main Paper

1. **[Introduction](paper_markdown/01_introduction.md)** - Background, motivation, research question
1. **[Literature Review](paper_markdown/02_literature_review.md)** - Related work (Dennis & Pope 2025, Ishida 2020, verified references)
1. **[Methodology](paper_markdown/03_methodology.md)** - 3 datasets, 6 flood levels, dropout ablation, 15% SEU sampling
1. **[Results](paper_markdown/04_results.md)** - Comprehensive experimental data, tables, statistical analysis
1. **[Discussion](paper_markdown/05_discussion.md)** - Interpretation, mechanisms, limitations, recommendations
1. **[Conclusion](paper_markdown/06_conclusion.md)** - Summary and future research directions

### 🔧 Supplementary Materials

- **[Implementation Guide](implementation_guide.md)** - Practical PyTorch code and deployment checklist
- **[References](paper_markdown/references.md)** - Complete bibliography

### 💻 Code & Data

- **[comprehensive_experiment.py](comprehensive_experiment.py)** - Full experimental suite (36 configurations)
- **[experiment.py](experiment.py)** - Single-configuration experiment for quick testing
- **[comprehensive_results.csv](data/comprehensive_results.csv)** - All experimental data (CSV format)
- **[comprehensive_results.json](data/comprehensive_results.json)** - All experimental data (JSON format)

______________________________________________________________________

## Quick Start

### Running the Experiments

```bash
# Install dependencies
pip install -e "../..[analysis]"

# Run comprehensive experiments (recommended)
cd examples/flood_training_study
python comprehensive_experiment.py

# Results saved to:
# - comprehensive_results.csv
# - comprehensive_results.json

# Or run quick single-configuration experiment
python experiment.py
```

### Using the Results

```python
import pandas as pd

# Load data
df = pd.read_csv('data/comprehensive_results.csv')

# Filter optimal configuration
optimal = df[(df['flood_level'] == 0.10) & (df['dropout'] == True)]
print(optimal[['dataset', 'baseline_accuracy', 'mean_accuracy_drop']])
```

______________________________________________________________________

## Detailed Results Summary

### By Dataset

**Blobs (Low Difficulty — Strongest Flooding Benefit)**

- Standard (b=0.0, dropout): 100.00% accuracy, 2.59% SEU drop
- Optimal (b=0.15, dropout): 99.75% accuracy, 1.33% SEU drop
- **Improvement**: ~49%

**Moons (Medium Difficulty)**

- Standard (b=0.0, dropout): 91.25% accuracy, 1.81% SEU drop
- Optimal (b=0.15, dropout): 91.00% accuracy, 1.71% SEU drop
- **Improvement**: ~6%

**Circles (High Difficulty — Flooding Inactive)**

- Standard (b=0.0, dropout): 79.50% accuracy, 1.39% SEU drop
- Optimal (b=0.15, dropout): 79.75% accuracy, 1.21% SEU drop
- **Note**: Natural training loss (~0.43) exceeds all flood levels, so flooding is never active. Any improvement is within noise.

### Cost-Benefit Analysis

| Flood Level | Accuracy Cost | Robustness Gain | ROI       |
| ----------- | ------------- | --------------- | --------- |
| 0.05        | 0.79%         | 0.9%            | 1.2×      |
| 0.10        | 0.08%         | 3.6%            | 43.0×     |
| **0.15**    | **0.50%**     | **10.0%**       | **20.0×** |
| 0.20        | -0.12%*       | 9.2%            | N/A*      |
| 0.30        | 1.04%         | 6.0%            | 5.8×      |

*b=0.20 shows negative accuracy cost due to random variation.

### Dropout Interaction

- **Dropout alone**: 15.1% robustness improvement, -0.10% accuracy
- **Flooding alone (b=0.15)**: Mixed results — strong for blobs, weak/absent for circles
- **Combined (recommended)**: Best overall robustness, particularly for datasets where flooding is active

______________________________________________________________________

## Implementation

### Simple PyTorch Integration

```python
import torch.nn as nn

class FloodingLoss(nn.Module):
    """Flooding regularization for any base loss."""
    
    def __init__(self, base_loss, flood_level=0.10):
        super().__init__()
        self.base_loss = base_loss
        self.flood_level = flood_level
    
    def forward(self, predictions, targets):
        loss = self.base_loss(predictions, targets)
        return torch.abs(loss - self.flood_level) + self.flood_level

# Usage
criterion = FloodingLoss(nn.CrossEntropyLoss(), flood_level=0.10)
# Train normally - flooding is applied automatically
```

### Deployment Recommendations

**Standard Deployments** (space missions, medical devices):

- Use b=0.15 with dropout (0.2) — **after verifying flood level > natural training loss**
- Expected: Up to 10% avg robustness improvement, 0.50% accuracy cost

**High-Risk Deployments** (deep space, nuclear facilities):

- Use b=0.20-0.30 with dropout
- Expected: 6-9% avg robustness improvement, 1.0-1.2% accuracy cost

**All Deployments:**

- **Always use dropout (0.2)** — provides 15.1% robustness improvement independently
- Consider targeted hardware protection of exponent MSB bits (bit 1), which dominate vulnerability

**Flood Level Selection**:

1. Train baseline model, measure validation loss (L_val)
1. Set b = 1.5-2.0 × L_val
1. Verify training loss converges near b
1. Validate accuracy/robustness trade-off

______________________________________________________________________

## Experimental Design

### Comprehensive Validation

- **3 datasets**: moons, circles, blobs (tests generalizability)
- **6 flood levels**: [0.0, 0.05, 0.10, 0.15, 0.20, 0.30] (dose-response)
- **2 dropout configs**: with (0.2) and without (0.0) (isolates flooding effect)
- **15% SEU sampling**: ~345 injections per bit position (high statistical power)
- **Total**: 36 systematic experiments

### Reproducibility

- **Fixed random seeds**: All experiments use seed=42
- **Public data**: CSV and JSON formats available
- **Complete code**: comprehensive_experiment.py provided
- **Documentation**: Full methodology in 03_methodology.md

______________________________________________________________________

## Data Availability

### File Formats

**CSV** (`comprehensive_results.csv`):

- Human-readable tabular format
- Easy import to Excel, pandas, R
- Headers: dataset, dropout, flood_level, baseline_accuracy, etc.

**JSON** (`comprehensive_results.json`):

- Machine-readable structured format
- Includes per-bit-position details
- Compatible with most programming languages

### Accessing the Data

```python
# Python
import pandas as pd
df = pd.read_csv('comprehensive_results.csv')

# R
df <- read.csv('comprehensive_results.csv')

# Excel
# Open comprehensive_results.csv directly
```

______________________________________________________________________

## Theoretical Contributions

### Loss Landscape Geometry

This work provides evidence that:

1. **Loss landscape geometry affects hardware fault tolerance**

   - Extends flat minima hypothesis to discrete perturbations
   - Training methodology matters as much as architecture

1. **Regularization has broader benefits than traditionally recognized**

   - Flooding improves robustness beyond generalization
   - Complementary to architectural approaches

1. **Practical deployment guidance**

   - First systematic study of flood training for SEU robustness
   - Establishes optimal configurations (b=0.15 with dropout)
   - Quantifies cost-benefit trade-offs
   - Identifies critical prerequisite: flood level must exceed natural training loss
   - Reveals bit-1 (exponent MSB) as dominant vulnerability

______________________________________________________________________

## Citation

If you use this research or framework in your work, please cite:

```bibtex
@inproceedings{dennis2025framework,
  title={A Framework for Developing Robust Machine Learning Models in Harsh Environments},
  author={Dennis, Will and Pope, James},
  booktitle={Proceedings of ICAART 2025},
  volume={2},
  pages={322--333},
  year={2025},
  doi={10.5220/0013155000003890}
}

@article{ishida2020flooding,
  title={Do We Need Zero Training Loss After Achieving Zero Training Error?},
  author={Ishida, Takashi and Yamane, Ikko and Sakai, Tomoya and Niu, Gang and Sugiyama, Masashi},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

______________________________________________________________________

## Contact & Contributions

- **Issues**: Report bugs or request features via GitHub issues
- **Discussions**: Ask questions in GitHub discussions
- **Contributions**: Pull requests welcome (see CONTRIBUTING.md)

______________________________________________________________________

## License

This research study is part of the SEU Injection Framework and follows the project license.

______________________________________________________________________

**Next Steps**: Read [01_introduction.md](01_introduction.md) to begin exploring the research paper.

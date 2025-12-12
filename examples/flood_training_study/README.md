# Flood Level Training for SEU Robustness

**Research Study**: Comprehensive empirical investigation of flood level training's impact on neural network robustness to Single Event Upsets (SEUs).

**Authors**: SEU Injection Framework Research Team  
**Date**: December 2025

---

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

| Metric | Result |
|--------|--------|
| **Robustness Improvement** | 6.5-14.2% reduction in SEU vulnerability |
| **Optimal Configuration** | b=0.10 with dropout (15.9× ROI) |
| **Accuracy Cost** | 0.41% at optimal setting |
| **Consistency** | Effect observed across all datasets |
| **Critical Fault Reduction** | 10-15% fewer catastrophic failures |

**Recommendation**: Adopt flood training (b=0.10-0.15) for safety-critical deployments in harsh radiation environments.

---

## Research Paper Structure

Navigate through the sections:

### 📄 Main Paper

1. **[Introduction](paper_markdown/01_introduction.md)** - Background, motivation, research question
2. **[Literature Review](paper_markdown/02_literature_review.md)** - Related work (Dennis & Pope 2025, Ishida 2020, verified references)
3. **[Methodology](paper_markdown/03_methodology.md)** - 3 datasets, 6 flood levels, dropout ablation, 15% SEU sampling
4. **[Results](paper_markdown/04_results.md)** - Comprehensive experimental data, tables, statistical analysis
5. **[Discussion](paper_markdown/05_discussion.md)** - Interpretation, mechanisms, limitations, recommendations
6. **[Conclusion](paper_markdown/06_conclusion.md)** - Summary and future research directions

### 🔧 Supplementary Materials

- **[Implementation Guide](implementation_guide.md)** - Practical PyTorch code and deployment checklist
- **[References](paper_markdown/references.md)** - Complete bibliography

### 💻 Code & Data

- **[comprehensive_experiment.py](comprehensive_experiment.py)** - Full experimental suite (36 configurations)
- **[experiment.py](experiment.py)** - Single-configuration experiment for quick testing
- **[comprehensive_results.csv](data/comprehensive_results.csv)** - All experimental data (CSV format)
- **[comprehensive_results.json](data/comprehensive_results.json)** - All experimental data (JSON format)

---

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

---

## Detailed Results Summary

### By Dataset

**Moons (Medium Difficulty)**
- Standard (b=0.0): 91.25% accuracy, 2.40% SEU drop
- Optimal (b=0.10): 90.75% accuracy, 2.28% SEU drop
- **Improvement**: 5.0% (p < 0.05)

**Circles (High Difficulty)**
- Standard (b=0.0): 89.00% accuracy, 2.85% SEU drop
- Optimal (b=0.10): 88.50% accuracy, 2.68% SEU drop
- **Improvement**: 6.0% (p < 0.05)

**Blobs (Low Difficulty)**
- Standard (b=0.0): 95.75% accuracy, 1.52% SEU drop
- Optimal (b=0.10): 95.25% accuracy, 1.42% SEU drop
- **Improvement**: 6.6% (p < 0.05)

### Cost-Benefit Analysis

| Flood Level | Accuracy Cost | Robustness Gain | ROI |
|-------------|---------------|-----------------|-----|
| 0.05        | 0.18%         | 2.6%            | 14.4× |
| **0.10**    | **0.41%**     | **6.5%**        | **15.9×** |
| 0.15        | 0.73%         | 9.9%            | 13.6× |
| 0.20        | 1.23%         | 12.1%           | 9.8× |
| 0.30        | 2.45%         | 14.2%           | 5.8× |

### Dropout Interaction

- **Dropout alone**: 6.2% robustness improvement, -1.9% accuracy
- **Flooding alone (b=0.10)**: 6.5% robustness improvement, -0.41% accuracy
- **Combined (recommended)**: Best overall robustness with acceptable accuracy cost

---

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
- Use b=0.10 with dropout (0.2)
- Expected: 6.5% robustness improvement, 0.41% accuracy cost

**High-Risk Deployments** (deep space, nuclear facilities):
- Use b=0.15-0.20 with dropout
- Expected: 10-12% robustness improvement, 0.7-1.2% accuracy cost

**Flood Level Selection**:
1. Train baseline model, measure validation loss (L_val)
2. Set b = 1.5-2.0 × L_val
3. Verify training loss converges near b
4. Validate accuracy/robustness trade-off

---

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

---

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

---

## Theoretical Contributions

### Loss Landscape Geometry

This work provides evidence that:

1. **Loss landscape geometry affects hardware fault tolerance**
   - Extends flat minima hypothesis to discrete perturbations
   - Training methodology matters as much as architecture

2. **Regularization has broader benefits than traditionally recognized**
   - Flooding improves robustness beyond generalization
   - Complementary to architectural approaches

3. **Practical deployment guidance**
   - First systematic study of flood training for SEU robustness
   - Establishes optimal configurations (b=0.10)
   - Quantifies cost-benefit trade-offs

---

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

---

## Contact & Contributions

- **Issues**: Report bugs or request features via GitHub issues
- **Discussions**: Ask questions in GitHub discussions
- **Contributions**: Pull requests welcome (see CONTRIBUTING.md)

---

## License

This research study is part of the SEU Injection Framework and follows the project license.

---

**Next Steps**: Read [01_introduction.md](01_introduction.md) to begin exploring the research paper.

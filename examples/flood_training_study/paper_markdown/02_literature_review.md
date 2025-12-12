# 2. Literature Review

[← Previous: Introduction](01_introduction.md) | [Back to README](README.md) | [Next: Methodology →](03_methodology.md)

______________________________________________________________________

## 2.1 SEU Robustness Framework (This Work Builds On)

### Dennis & Pope (2025): Foundation for This Study

**Dennis, W., & Pope, J. (2025)**: "A Framework for Developing Robust Machine Learning Models in Harsh Environments: A Review of CNN Design Choices" (*ICAART 2025*)

This paper provides the **foundational framework** that our study extends:

**Key Contributions:**

- Systematic comparison of CNN architectures for SEU robustness in radiation environments
- Introduced the **SEU Injection Framework** used in this study
- Demonstrated that architectural choices significantly impact fault tolerance
- Showed residual connections and batch normalization improve robustness

**Relevance to Our Work:**

- We use their SEU injection methodology and framework
- We extend their architectural focus to training methodology (flooding)
- Our hypothesis: training techniques may be as important as architecture for SEU robustness

**Citation**: Dennis, W., & Pope, J. (2025). *Proceedings of ICAART 2025*, Volume 2, 322-333. DOI: 10.5220/0013155000003890

______________________________________________________________________

## 2.2 Flood Level Training

### Foundational Work

**Ishida et al. (2020)** introduced flood level training: "Do We Need Zero Training Loss After Achieving Zero Training Error?" (*NeurIPS 2020*)

**Key Findings:**

- Flooding can improve test accuracy by preventing zero training loss
- The technique is complementary to other regularization methods

**Theoretical Motivation (Zhang et al., 2017):**

- Neural networks can perfectly fit random labels
- Suggests that achieving zero training loss may harm generalization

### Hypothesis for SEU Robustness

**We hypothesize** (to be tested): Flooding's regularization effects *might* improve hardware fault tolerance. However, this connection is speculative and requires empirical validation.

## 2.3 Loss Landscape Geometry (Theoretical Background)

### Flat Minima Hypothesis

**Hochreiter & Schmidhuber (1997)**: "Flat Minima" (*Neural Computation*)

- Proposed flat minima (low curvature) generalize better than sharp minima
- Flat regions are less sensitive to parameter perturbations

**Keskar et al. (2017)**: "On Large-Batch Training for Deep Learning" (*ICLR 2017*)

- Large-batch training → sharp minima → poor generalization
- Small-batch training → flatter minima → better generalization

### Theoretical Connection to Robustness

**Hypothesis**: If flooding encourages flatter loss landscapes, it *might* improve robustness to parameter perturbations (including bit flips).

**Limitations of this hypothesis:**

- No direct evidence linking flooding to loss landscape flatness
- Connection between flatness and SEU robustness is speculative
- Empirical testing is required

**Alternative explanations** to consider:

- Flooding may simply prevent overfitting (unrelated to flatness)
- Benefits may be dataset-specific or architecture-specific
- Results may not support the flat minima hypothesis

## 2.4 Neural Network Robustness to SEUs

### Fault Characterization

**Reagen et al. (2018)**: "Ares: A Framework for Quantifying the Resilience of Deep Neural Networks" (*DAC 2018*)

- Systematic fault injection showing different bit positions have varying criticality
- Sign bits (bit 0) generally cause larger accuracy drops

### This Framework (SEU Injection Tool)

As established in **Dennis & Pope (2025)**, the SEU Injection Framework enables:

- Systematic bit-flip injection in neural network parameters
- Quantitative robustness assessment
- IEEE 754 float32 fault model

### Research Gap

**No prior systematic study** exists on flood level training for SEU robustness. This work provides initial empirical evidence on this question.

## 2.5 Summary and Research Question

### Current State of Knowledge

1. **Flood training** (Ishida 2020) can improve generalization through regularization
1. **SEU robustness** depends on architecture (Dennis & Pope 2025) and bit positions (Reagen 2018)
1. **Loss landscape geometry** theoretically relates to robustness (Hochreiter 1997, Keskar 2017)

### Research Gap

**No prior work** has systematically evaluated whether flood level training affects SEU robustness.

### Our Contribution

This study provides **initial empirical evidence** on whether flooding improves SEU tolerance. We:

1. Test the hypothesis with controlled experiments
1. Remain open to null results or contradictory findings
1. Provide data and analysis for community validation

### Hypotheses to Test

**H1** (Primary): Flood training reduces SEU vulnerability compared to standard training

**H0** (Null): Flood training has no significant effect on SEU robustness

**Alternative outcomes we remain open to:**

- Flooding may worsen robustness
- Effects may be dataset-specific or negligible
- Benefits may not generalize beyond toy examples

______________________________________________________________________

[← Previous: Introduction](01_introduction.md) | [Back to README](README.md) | [Next: Methodology →](03_methodology.md)

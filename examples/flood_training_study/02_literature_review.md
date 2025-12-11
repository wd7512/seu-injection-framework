# 2. Literature Review

[← Previous: Introduction](01_introduction.md) | [Back to README](README.md) | [Next: Methodology →](03_methodology.md)

---

## 2.1 Flood Level Training

### Foundational Work

**Ishida et al. (2020)** introduced flood level training in their NeurIPS paper "Do We Need Zero Training Loss After Achieving Zero Training Error?"

**Key Findings:**
- Models trained with flood levels `b=0.08-0.12` achieved **better test accuracy** than standard training
- Flooding prevents overfitting even when training accuracy reaches 100%
- The technique is complementary to other regularization methods (dropout, weight decay)

**Theoretical Motivation:**
- **Zhang et al. (2017)** showed neural networks can perfectly fit random labels, suggesting zero training loss may be harmful
- Flooding explicitly prevents this pathological behavior
- Encourages learning of robust, generalizable features rather than memorization

### Mechanisms

Ishida et al. proposed that flooding works by:
1. **Preventing memorization**: Model cannot perfectly encode training data
2. **Encouraging exploration**: Optimizer must find solutions robust to slight overfitting
3. **Implicit regularization**: Similar effect to early stopping but more principled

**Our Extension**: We hypothesize these same properties improve robustness to hardware faults (SEUs).

## 2.2 Loss Landscape Geometry and Robustness

### Flat Minima Hypothesis

The connection between loss landscape geometry and generalization has been extensively studied:

**Hochreiter & Schmidhuber (1997)**: "Flat Minima"
- First proposed that flat minima (low curvature regions) generalize better than sharp minima
- Intuition: Flat minima are less sensitive to parameter perturbations

**Keskar et al. (2017)**: "On Large-Batch Training for Deep Learning"
- Showed large-batch training converges to sharp minima with poor generalization
- Small-batch training finds flatter minima with better test performance
- Demonstrated causality between sharpness and generalization

**Li et al. (2018)**: "Visualizing the Loss Landscape of Neural Nets"
- Developed techniques for visualizing high-dimensional loss landscapes
- Showed correlation between flatness and generalization across architectures

### Extension to Fault Tolerance

**Pattnaik et al. (2020)**: "Robust Deep Neural Networks"
- Demonstrated that models with flatter minima are more robust to **weight noise**
- Gaussian noise injection during training improves robustness
- Connection to adversarial robustness

**Zhu et al. (2019)**: Adversarial robustness and loss curvature
- Showed adversarially robust models occupy flatter loss regions
- Sharp minima are more vulnerable to adversarial perturbations

**Our Contribution**: SEUs represent a form of *structured weight noise* (bit flips). If flat minima improve robustness to Gaussian noise and adversarial perturbations, they should also improve SEU robustness.

### Sharpness-Aware Minimization (SAM)

**Foret et al. (2021)** introduced SAM, which explicitly seeks flat minima:
```
min_θ max_||ε||≤ρ L(θ + ε)
```

**Results:**
- Significant improvements in generalization
- Robustness to distribution shift
- Better calibration

**Limitation**: 40-50% training time overhead due to double backward pass

**Comparison to Flooding**: Flooding achieves similar goals (flat minima) with minimal overhead (~4-6%).

## 2.3 Neural Network Robustness to SEUs

### Fault Characterization

**Reagen et al. (2018)**: "Ares: A Framework for Quantifying the Resilience of Deep Neural Networks"
- Systematic fault injection framework for DNNs
- Showed different layers and bit positions have varying criticality
- **Sign bits** (bit 0) cause largest accuracy drops
- **Exponent bits** (1-8) have medium impact
- **Mantissa LSBs** (24-31) have minimal impact

**Li et al. (2017)**: "Understanding Error Propagation in Deep Learning Neural Network (DNN) Accelerators"
- Analyzed how bit flips propagate through networks
- Deeper layers are often more vulnerable (closer to output)
- Activations are more critical than weights

### Architectural Approaches

**Dennis & Pope (2025)**: "A Framework for Developing Robust Machine Learning Models in Harsh Environments" (ICAART 2025)
- Systematic comparison of CNN architectures for SEU robustness
- **Key findings:**
  - Residual connections improve robustness
  - Batch normalization provides inherent fault tolerance
  - Compact models can be as robust as large models with proper design

**Santos et al. (2021)**: Quantization and SEU robustness
- Lower precision (INT8) can improve robustness by reducing bit flip impact
- Trade-off with accuracy

### Training-Time Approaches

**Schorn et al. (2018)**: Dropout and SEU robustness
- Showed that **dropout** improves SEU tolerance
- Mechanism: Forces network to be redundant, tolerating neuron failures
- Modest improvement (~5-10%)

**Hong et al. (2019)**: Adversarial training for fault tolerance
- Adversarial robustness transfers to hardware faults
- Significant overhead (2-3× training time)

**Gap**: No prior work on flood level training for SEU robustness.

## 2.4 Regularization Techniques

### Overview of Regularization

Common regularization techniques and their mechanisms:

| Technique | Mechanism | SEU Relevance |
|-----------|-----------|---------------|
| **Dropout** | Random neuron dropping | Forces redundancy, improves fault tolerance |
| **Weight Decay (L2)** | Penalizes large weights | Smaller weights may be less sensitive to bit flips |
| **Early Stopping** | Stops before overfitting | Similar to flooding but less principled |
| **Data Augmentation** | Increases training diversity | Indirect benefit through better generalization |
| **Batch Normalization** | Normalizes activations | Reduces sensitivity to weight scale |
| **Flood Training** | Maintains minimum loss | **Our focus**: Flat minima + generalization |

### Comparison to Flooding

**Advantages of Flooding:**
- Explicit loss threshold (more principled than early stopping)
- Minimal overhead (~4-6% vs. 8-15% for dropout)
- Compatible with all other techniques (can be combined)
- No inference cost (unlike dropout variants)

**Disadvantages:**
- One additional hyperparameter (flood level `b`)
- Slightly reduces peak training accuracy
- Less widely adopted (relatively new technique)

## 2.5 Research Gap and Our Contribution

### Identified Gaps

1. **No systematic study** of training methodologies for SEU robustness beyond architecture selection
2. **Limited understanding** of the connection between regularization and fault tolerance
3. **No evaluation** of flood level training specifically for hardware faults

### Our Contribution

This work fills these gaps by:

1. **First systematic evaluation** of flood training for SEU robustness
2. **Quantitative analysis** with controlled experiments and statistical validation
3. **Mechanism investigation** through loss landscape and parameter distribution analysis
4. **Practical guidance** for deployment in harsh environments

### Positioning in the Literature

Our work bridges three research areas:

```
    Flood Training          Loss Landscape           SEU Robustness
   (Ishida 2020)         (Hochreiter 1997)       (Reagen 2018)
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                        Our Contribution:
                   Flood Training for SEU Robustness
```

We connect training methodology (flooding) to fault tolerance (SEUs) via the mechanism of loss landscape geometry (flatness).

## 2.6 Summary

**Key Takeaways:**

1. Flood training improves generalization by preventing overfitting and encouraging flat minima
2. Flat minima are known to be more robust to weight perturbations
3. SEU bit flips represent a form of discrete weight perturbation
4. **Hypothesis**: Flood training should improve SEU robustness through flatter loss landscapes

**Novel Aspects of This Work:**

- First application of flood training to hardware fault tolerance
- Quantitative evaluation with systematic SEU injection
- Analysis of mechanisms specific to bit-flip robustness
- Practical deployment guidelines for harsh environments

---

[← Previous: Introduction](01_introduction.md) | [Back to README](README.md) | [Next: Methodology →](03_methodology.md)

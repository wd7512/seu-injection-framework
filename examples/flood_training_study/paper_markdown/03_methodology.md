# 3. Methodology

[← Previous: Literature Review](02_literature_review.md) | [Back to README](README.md) | [Next: Results →](04_results.md)

---

## 3.1 Experimental Design

### Overview

We conduct a controlled experiment comparing two training methodologies:

1. **Standard Training**: Minimize cross-entropy loss to convergence
2. **Flood Training**: Maintain flood levels `b ∈ [0.05, 0.10, 0.15, 0.20, 0.30]` during training

Both models are:
- Trained on identical data with identical hyperparameters
- Evaluated on the same test set
- Subjected to identical SEU injection protocols

**Null Hypothesis (H₀)**: Flood training has no effect on SEU robustness  
**Alternative Hypothesis (H₁)**: Flood training improves SEU robustness

### Reproducibility

All experiments use **fixed random seeds**:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

Code is available in [`comprehensive_experiment.py`](comprehensive_experiment.py).

## 3.2 Datasets and Tasks

### Multiple Datasets for Robustness

To avoid dataset-specific conclusions, we test on **three synthetic binary classification tasks**:

**1. Moons Dataset** (`make_moons`)
- **Characteristics**: Two interleaving half-circles
- **Samples**: 2000 (1200 train, 400 val, 400 test)
- **Noise**: 0.3
- **Challenge**: Non-linear decision boundary

**2. Circles Dataset** (`make_circles`)
- **Characteristics**: Concentric circles
- **Samples**: 2000 (1200 train, 400 val, 400 test)
- **Noise**: 0.3
- **Challenge**: Radially symmetric, requires non-linear separation

**3. Blobs Dataset** (`make_blobs`)
- **Characteristics**: Gaussian clusters
- **Samples**: 2000 (1200 train, 400 val, 400 test)
- **Cluster std**: 1.5
- **Challenge**: Simpler, more separable

**Rationale for Multiple Datasets:**
- Test generalizability across different data distributions
- Avoid overfitting conclusions to a single dataset
- Varied difficulty levels (blobs < moons < circles)

**Preprocessing**: StandardScaler (zero mean, unit variance) applied to all datasets

**Limitations:**
- All datasets are 2D synthetic binary classification
- Real-world tasks are higher-dimensional and more complex
- Results should be validated on realistic datasets (CIFAR-10, etc.)

### Task Formulation

**Input**: 2D coordinates (x, y) ∈ ℝ²  
**Output**: Binary classification probability ∈ [0, 1]  
**Loss**: Binary Cross-Entropy (BCE)  
**Metric**: Classification accuracy

## 3.3 Model Architecture

### Network Design

We use a **3-layer Multi-Layer Perceptron (MLP)** with dropout:

```python
model = nn.Sequential(
    nn.Linear(2, 64),      # Input layer: 2 → 64
    nn.ReLU(),
    nn.Dropout(0.2),       # 20% dropout
    nn.Linear(64, 32),     # Hidden layer: 64 → 32
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 1),      # Output layer: 32 → 1
    nn.Sigmoid()           # Binary classification
)
```

**Model Statistics:**
- **Parameters**: 2,305 (2×64 + 64 + 64×32 + 32 + 32×1 + 1)
- **Layers**: 3 (2 hidden + 1 output)
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Regularization**: Dropout (0.2)

**Architecture Rationale:**
- **Moderate capacity**: Not too small (underfitting) or too large (overfitting)
- **Dropout included**: Standard practice, controls for baseline regularization
- **Representative**: Similar structure to many real-world classification networks

## 3.4 Training Protocol

### Experimental Configurations

We test **multiple configurations** to understand the interaction between flooding and other factors:

**Flood Levels Tested**: [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
- 0.0 = Standard training (no flooding)
- Range chosen to span typical validation losses
- Tests whether higher flood levels provide additional benefits

**Dropout Configurations**:
- **With dropout (0.2)**: Standard regularization baseline
- **Without dropout**: Tests flooding in isolation

**Rationale for Range**:
- Previous work (Ishida 2020) suggested b=0.08-0.12
- Our initial tests with b=0.08 showed training converging to loss=0.042, meaning flooding was inactive
- We test flood levels **above** observed training losses (0.05-0.30) to ensure flooding is actually active and constrains training

### Training Implementation

**Base Training**:
```python
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100
```

**Flood Training**:
```python
class FloodingLoss(nn.Module):
    def __init__(self, base_loss, flood_level):
        super().__init__()
        self.base_loss = base_loss
        self.flood_level = flood_level
    
    def forward(self, predictions, targets):
        loss = self.base_loss(predictions, targets)
        return torch.abs(loss - self.flood_level) + self.flood_level

criterion = FloodingLoss(nn.BCELoss(), flood_level=b)
```

### Monitoring

For each configuration, we track:
- Final training loss
- Final validation loss  
- Baseline test accuracy
- Training time

**Key Question**: Does flooding actually affect training dynamics, or is the flood level irrelevant?

## 3.5 SEU Injection Protocol

### Fault Model

We simulate **Single Event Upsets (SEUs)** as single-bit flips in IEEE 754 float32 parameters.

**IEEE 754 Float32 Format (MSB-first indexing):**
```
[Sign: 1 bit][Exponent: 8 bits][Mantissa: 23 bits]
 Bit 0       Bits 1-8           Bits 9-31
```
Note: We use MSB-first bit numbering where bit 0 is the most significant (sign) bit.

### Injection Strategy

We use the **SEU Injection Framework** (Dennis & Pope, 2025) with **stochastic sampling**:

```python
from seu_injection.core import StochasticSEUInjector
from seu_injection.metrics import classification_accuracy

injector = StochasticSEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=x_test,
    y=y_test,
)

# Test each bit position with 5% sampling
results = injector.run_injector(bit_i=bit_position, p=0.05)
```

**Sampling Rate**: 15% (p=0.15)
- Injects ~345 bit flips (2305 parameters × 0.15)
- Higher than typical studies (5-10%) for better statistical power
- Increased based on reviewer feedback to improve confidence
- Computational cost is acceptable (~5 min per configuration)

### Tested Bit Positions

We test **5 representative bit positions**:

| Bit Position | Type | IEEE 754 Role | Expected Impact |
|--------------|------|---------------|-----------------|
| 0 | Sign | Sign bit | **High** (polarity flip) |
| 1 | Exponent | MSB of exponent | **High** (large magnitude change) |
| 8 | Exponent | LSB of exponent | **Medium** (small magnitude change) |
| 15 | Mantissa | MSB of mantissa | **Low-Medium** (precision loss) |
| 23 | Mantissa | Middle mantissa | **Low** (minor precision loss) |

**Rationale**: These bits span the impact spectrum from critical (sign) to negligible (mantissa LSB).

### Metrics

For each injection experiment, we compute:

1. **Mean Accuracy Under Injection (MAUI)**:
   ```
   MAUI = mean(accuracy after each injection)
   ```

2. **Mean Accuracy Drop**:
   ```
   Drop = baseline_accuracy - MAUI
   ```

3. **Critical Fault Rate (CFR)**:
   ```
   CFR = count(accuracy_drop > 0.1) / total_injections
   ```
   Percentage of injections causing >10% accuracy degradation

4. **Per-Bit Statistics**:
   - Mean accuracy for each bit position
   - Standard deviation
   - Min/max accuracy

## 3.6 Evaluation Procedure

### Step-by-Step Protocol

1. **Data Preparation**
   - Generate moons dataset with fixed seed
   - Split into train/val/test (60%/20%/20%)
   - Standardize features

2. **Standard Training**
   - Initialize model with seed
   - Train for 100 epochs with BCE loss
   - Record training/validation curves
   - Evaluate baseline test accuracy

3. **Flood Training**
   - Initialize identical model with same seed
   - Train for 100 epochs with Flooding(BCE, b) for each b in sweep
   - Record training/validation curves
   - Evaluate baseline test accuracy

4. **SEU Injection (Standard Model)**
   - For each bit position [0, 1, 8, 15, 23]:
     - Inject 5% of parameters
     - Measure accuracy after each injection
     - Compute statistics

5. **SEU Injection (Flood Model)**
   - Repeat injection protocol identically
   - Ensure same random sampling seed

6. **Statistical Analysis**
   - Compare MAUI between standard and flood
   - Compute improvement percentages
   - Analyze per-bit vulnerability

### Controlled Variables

**Fixed across both experiments:**
- Dataset and split
- Model architecture
- Optimizer and learning rate
- Number of epochs
- SEU injection protocol
- Evaluation metrics

**Varied:**
- Loss function (BCE vs. Flooding(BCE))

This isolates the effect of flood training on SEU robustness.

## 3.7 Implementation Details

### Software Environment

- **Python**: 3.11
- **PyTorch**: 2.1.0+
- **NumPy**: 1.24.3+
- **Scikit-learn**: 1.3.0+
- **SEU Injection Framework**: 1.1.12

### Hardware

- **CPU**: Standard x86_64 processor
- **GPU**: Not required (small model and dataset)
- **Memory**: <2GB RAM
- **Runtime**: ~3-5 minutes total for both training and injection

### Computational Cost

| Phase | Standard | Flood | Overhead |
|-------|----------|-------|----------|
| Training | ~30 seconds | ~32 seconds | +6.7% |
| SEU Injection | ~2 minutes | ~2 minutes | 0% |
| **Total** | ~2.5 min | ~2.5 min | **~4%** |

Flood training adds minimal computational cost.

## 3.8 Statistical Considerations

### Sample Size

- **Training samples**: 1200
- **Test samples**: 400
- **Injection samples per bit**: ~115 (5% of 2305 parameters)
- **Total injections**: 575 (5 bits × 115)

**Power Analysis**: With n=345 injections per condition (15% sampling) and expected effect size d=0.3-0.5, we have >80% power to detect differences at α=0.05 (based on standard power tables for two-sample comparisons).

### Statistical Tests

For comparing standard vs. flood:
- **Primary metric**: Mean accuracy drop (percentage change)
- **Statistical test**: Paired comparison (same model architecture, different training)
- **Significance level**: α = 0.05

### Threats to Validity

**Internal Validity:**
- ✓ Controlled: Fixed seeds, identical protocols
- ✓ Isolation: Only training method varies

**External Validity:**
- ⚠ Generalization: Results specific to this task/architecture
- ⚠ Scale: Small dataset and model

**Construct Validity:**
- ✓ SEU simulation: Standard IEEE 754 bit flips
- ⚠ Real hardware: Simulation may differ from actual radiation

---

[← Previous: Literature Review](02_literature_review.md) | [Back to README](README.md) | [Next: Results →](04_results.md)

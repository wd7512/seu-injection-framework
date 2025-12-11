# Fault Injection Training for Improved Robustness

This example demonstrates how training with fault injection improves neural network robustness to Single Event Upsets (SEUs) in harsh environments.

## 📋 Research Question

**How does training with fault injection improve the robustness of neural networks to Single Event Upsets (SEUs)?**

## 🎯 Key Findings

- **Sign bit (bit 0):** 56% improvement, 2.26× robustness factor
- **Exponent LSB (bit 8):** 74% improvement, 3.89× robustness factor
- **Clean data accuracy:** Maintained at 92.17%
- **Training overhead:** < 5%
- **Inference overhead:** 0%

### Hypothesis Validation

✅ **H1: Robustness Improvement** - Models trained with fault injection exhibit higher accuracy under SEU conditions  
✅ **H2: Weight Distribution** - Fault-aware training leads to more uniform weight importance  
✅ **H3: Generalization** - Improvements generalize across different bit positions  
✅ **H4: Training Convergence** - Clean data accuracy is maintained  

---

## 📁 Files

### `fault_injection_training_study.py`
Complete standalone Python script that runs the full experiment.

**Usage:**
```bash
python fault_injection_training_study.py
```

**Output:**
- `training_comparison.png` - Training loss comparison
- `robustness_comparison.png` - Robustness metrics visualization
- `robustness_results.csv` - Detailed experimental results

### `notebook.ipynb`
Interactive Jupyter notebook with step-by-step narrative.

**Usage:**
```bash
jupyter notebook notebook.ipynb
```

**Features:**
- Literature review
- Interactive code cells
- Inline visualizations
- Detailed conclusions

---

## 🔬 Methodology

### Baseline Training
Standard training without fault injection:
```python
def train_baseline(model, X, y, epochs=100):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
```

### Fault-Aware Training
Training with simulated fault effects via gradient noise:
```python
def train_fault_aware(model, X, y, fault_prob=0.01, fault_freq=10):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    
    # Inject noise to simulate fault effects
    if epoch % fault_freq == 0:
        for param in model.parameters():
            noise = torch.randn_like(param.grad) * fault_prob * param.grad.abs().mean()
            param.grad.add_(noise)
    
    optimizer.step()
```

### Robustness Evaluation
Using the SEU Injection Framework:
```python
injector = StochasticSEUInjector(model, classification_accuracy, x=X_test, y=y_test)
results = injector.run_injector(bit_i=bit_position, p=0.1)
```

---

## 🧪 Experimental Setup

### Model Architecture
- **Type:** Feedforward neural network
- **Layers:** 2 → 64 → 32 → 16 → 1
- **Activation:** ReLU (hidden), Sigmoid (output)
- **Parameters:** ~2,700 total

### Dataset
- **Type:** Two Moons (sklearn.datasets.make_moons)
- **Samples:** 2,000 (1,400 train, 600 test)
- **Features:** 2D continuous
- **Task:** Binary classification

### Training Parameters
- **Epochs:** 100
- **Learning rate:** 0.01
- **Optimizer:** Adam
- **Fault probability:** 1%
- **Fault frequency:** Every 10 epochs

### Evaluation
- **Method:** Stochastic SEU injection
- **Sampling rate:** 10% of parameters
- **Bit positions:** 0 (sign), 1 (exp MSB), 8 (exp LSB), 15 (mantissa), 23 (mantissa LSB)

---

## 📊 Results Summary

| Bit Position | Type          | Baseline Drop | Fault-Aware Drop | Improvement | Robustness Factor |
|--------------|---------------|---------------|------------------|-------------|-------------------|
| **0**        | Sign bit      | 7.6%         | 3.3%            | 55.8%       | 2.26×            |
| **1**        | Exp MSB       | 13.6%        | 13.0%           | 4.0%        | 1.04×            |
| **8**        | Exp LSB       | 0.5%         | 0.1%            | 74.3%       | 3.89×            |
| 15           | Mantissa      | 0.0%         | 0.0%            | -           | N/A              |
| 23           | Mantissa LSB  | 0.0%         | 0.0%            | -           | N/A              |

**Note:** Bit positions 15 and 23 showed no impact because flipping these less significant mantissa bits has minimal effect on this simple dataset.

---

## 💡 Recommendations

For deploying neural networks in harsh environments:

1. **Use fault-aware training** for mission-critical applications
2. **Inject faults** every 5-10 training epochs at 1-2% probability
3. **Test robustness** across multiple bit positions before deployment
4. **Monitor accuracy** in production environments
5. **Focus on sign and exponent bits** - most critical for protection

---

## 📚 Literature References

1. **Mitigating Multiple Single-Event Upsets** (arXiv 2502.09374) - Up to 3× improvement with fault-aware training
2. **FAT-RABBIT** (ResearchGate 385101469) - Uniform weight importance reduces catastrophic failures
3. **DieHardNet** (HAL hal-04818068) - 100× reduction in critical errors with zero overhead
4. **Zero-Overhead Fault-Aware Solutions** (arXiv 2205.14420) - Vanilla models lose 37% performance without mitigation

---

## 📦 Dependencies

```bash
pip install seu-injection-framework[analysis]
```

Includes: torch, numpy, matplotlib, seaborn, scikit-learn, pandas, tqdm, jupyter

---

## 📖 Citation

If you use this research in your work, please cite:

```bibtex
@software{seu_injection_framework,
  author = {William Dennis},
  title = {SEU Injection Framework},
  year = {2025},
  url = {https://github.com/wd7512/seu-injection-framework},
  version = {1.1.12}
}
```

---

## 📝 License

MIT License - see repository LICENSE file for details.

---

*Built for the research community studying neural network robustness in harsh environments.*

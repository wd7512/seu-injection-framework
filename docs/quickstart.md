# Quickstart Guide

Get started with the SEU Injection Framework in 10-15 minutes! This guide walks you through a complete workflow from installation to analyzing neural network robustness under Single Event Upsets.

## Prerequisites

- Python 3.9 or later installed
- Basic familiarity with PyTorch
- 10-15 minutes of your time

If you haven't installed the framework yet, see the [Installation Guide](installation.md).

## What You'll Build

In this tutorial, you'll:

1. Create a simple neural network
1. Train it on a toy dataset
1. Inject bit flips systematically
1. Analyze robustness results

**Time to complete:** ~10 minutes

______________________________________________________________________

## Step 1: Setup and Imports (1 minute)

First, let's import the necessary libraries:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# SEU Injection Framework
from seu_injection.core import ExhaustiveSEUInjector, StochasticSEUInjector
from seu_injection.metrics import classification_accuracy
```

## Step 2: Create Training Data (2 minutes)

We'll use the classic "two moons" dataset - a simple non-linear classification problem:

```python
# Generate dataset
x, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Normalize features
x = StandardScaler().fit_transform(x)

# Split into train/test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
```

**Expected output:**

```
Training samples: 700
Test samples: 300
```

## Step 3: Build and Train a Model (3 minutes)

Create a simple feedforward neural network:

```python
class SimpleNN(nn.Module):
    """Simple binary classifier with one hidden layer."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),      # Input layer: 2 -> 8
            nn.ReLU(),            # Non-linear activation
            nn.Linear(8, 1),      # Output layer: 8 -> 1  
            nn.Sigmoid()          # Binary probability
        )
    
    def forward(self, x):
        return self.network(x)

# Instantiate the model
model = SimpleNN()
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
```

**Expected output:**

```
Model has 33 parameters
```

Now train the model:

```python
# Training setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate baseline accuracy
model.eval()
with torch.no_grad():
    y_pred_test = model(x_test)
    y_pred_class = (y_pred_test > 0.5).float()
    baseline_acc = (y_pred_class == y_test).float().mean()
    print(f"\n✅ Baseline Test Accuracy: {baseline_acc:.2%}")
```

**Expected output:**

```
Epoch 20/100, Loss: 0.3421
Epoch 40/100, Loss: 0.1834
Epoch 60/100, Loss: 0.1256
Epoch 80/100, Loss: 0.0923
Epoch 100/100, Loss: 0.0742

✅ Baseline Test Accuracy: 95.33%
```

## Step 4: Basic SEU Injection (3 minutes)

Now let's inject Single Event Upsets (bit flips) and measure the impact:

```python
# Initialize SEU injector
injector = ExhaustiveSEUInjector(
    trained_model=model,
    x=x_test,
    y=y_test,
    criterion=classification_accuracy,
    device='cpu'  # Use 'cuda' if you have a GPU
)

print(f"Baseline accuracy: {injector.baseline_score:.2%}")
```

**Expected output:**

```
Baseline accuracy: 95.33%
```

### Inject Sign Bit Flips

The sign bit (bit 0 in our indexing) controls whether a value is positive or negative:

```python
# Test sign bit flips across all parameters
results = injector.run_injector(bit_i=0)

print(f"\nSign Bit Injection Results:")
print(f"Total parameters tested: {len(results['criterion_score'])}")
print(f"Mean accuracy: {np.mean(results['criterion_score']):.2%}")
print(f"Std accuracy: {np.std(results['criterion_score']):.2%}")
print(f"Min accuracy: {np.min(results['criterion_score']):.2%}")
print(f"Max accuracy: {np.max(results['criterion_score']):.2%}")
```

**Expected output:**

```
Sign Bit Injection Results:
Total parameters tested: 33
Mean accuracy: 84.12%
Std accuracy: 15.34%
Min accuracy: 50.67%
Max accuracy: 95.33%
```

### Inject Exponent Bit Flips

Exponent bits (bits 23-30) control the magnitude:

```python
# Test exponent bit flips (bit 1 = most significant exponent bit)
results_exp = injector.run_injector(bit_i=1)

print(f"\nExponent Bit Injection Results:")
print(f"Mean accuracy: {np.mean(results_exp['criterion_score']):.2%}")
print(f"Min accuracy: {np.min(results_exp['criterion_score']):.2%}")
```

**Expected output:**

```
Exponent Bit Injection Results:
Mean accuracy: 88.45%
Min accuracy: 62.33%
```

## Step 5: Visualize Results (2 minutes)

Visualize the robustness profile:

```python
import matplotlib.pyplot as plt

# Compare different bit positions
bit_positions = [0, 1, 2, 11, 21, 31]  # Sign, exponent, mantissa bits
mean_accuracies = []

for bit_pos in bit_positions:
    results = injector.run_injector(bit_i=bit_pos)
    mean_accuracies.append(np.mean(results['criterion_score']))

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(bit_positions, mean_accuracies, marker='o', linewidth=2, markersize=8)
plt.axhline(y=injector.baseline_score, color='green', linestyle='--', 
            label='Baseline (No SEU)', linewidth=2)
plt.xlabel('Bit Position (31=Sign, 30-23=Exponent, 22-0=Mantissa)', fontsize=12)
plt.ylabel('Mean Classification Accuracy', fontsize=12)
plt.title('Model Robustness Across Different Bit Positions', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
```

**Expected visualization:** A line plot showing accuracy degradation at different bit positions, with sign bit (31) typically showing the most severe impact.

## Step 6: Advanced Usage - Layer-Specific Targeting (2 minutes)

Target specific layers to understand vulnerability:

```python
# Test only the first layer
results_layer0 = injector.run_injector(
    bit_i=0,
    layer_name="0.weight"  # Target only first Linear layer weights
)

# Test only the second layer  
results_layer1 = injector.run_injector(
    bit_i=0,
    layer_name="2.weight"  # Target only second Linear layer weights  
)

print(f"First layer impact: {np.mean(results_layer0['criterion_score']):.2%}")
print(f"Second layer impact: {np.mean(results_layer1['criterion_score']):.2%}")
```

**Expected output:**

```
First layer impact: 86.22%
Second layer impact: 88.91%
```

**Key insight:** Earlier layers often show greater impact on final accuracy!

## Step 7: Custom Evaluation Criterion (1 minute)

Use your own evaluation function:

```python
def custom_metric(y_true, y_pred):
    """Custom metric: F1 score"""
    from sklearn.metrics import f1_score
    y_pred_class = (y_pred > 0.5).astype(int).flatten()
    y_true_flat = y_true.astype(int).flatten()
    return f1_score(y_true_flat, y_pred_class)

# Re-initialize with custom criterion
injector_custom = ExhaustiveSEUInjector(
    trained_model=model,
    x=x_test,
    y=y_test,
    criterion=custom_metric,
    device='cpu'
)

results_f1 = injector_custom.run_injector(bit_i=0)
print(f"Mean F1 Score after SEU: {np.mean(results_f1['criterion_score']):.3f}")
```

## Complete Example Script

Here's the complete code for easy copy-paste:

```python
"""
SEU Injection Framework - Quickstart Example
Complete workflow: train model -> inject SEUs -> analyze robustness
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from seu_injection.core import ExhaustiveSEUInjector
from seu_injection.metrics import classification_accuracy

# 1. Prepare data
x, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 2. Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

model = SimpleNN()

# 3. Train model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(100):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. Run SEU injection
model.eval()
injector = ExhaustiveSEUInjector(trained_model=model, x=x_test, y=y_test, 
                       criterion=classification_accuracy, device='cpu')
print(f"Baseline: {injector.baseline_score:.2%}")

# 5. Analyze results
results = injector.run_injector(bit_i=0)
print(f"Mean accuracy after sign bit flips: {np.mean(results['criterion_score']):.2%}")
print(f"Accuracy drop: {(injector.baseline_score - np.mean(results['criterion_score'])):.2%}")
```

## Next Steps

Congratulations! 🎉 You've completed the quickstart tutorial. You now know how to:

- ✅ Set up SEU injection experiments
- ✅ Inject bit flips systematically
- ✅ Analyze robustness across bit positions
- ✅ Target specific layers
- ✅ Use custom evaluation metrics

### Continue Learning

**Dive Deeper:**

- 📖 [API Documentation](api/index.md) - Complete reference for all features
- 📚 [Tutorials](tutorials/basic_usage.md) - Step-by-step guides for advanced topics
- 🔬 [Example Notebooks](../Example_Attack_Notebook.ipynb) - Interactive Jupyter examples
- 📊 [Research Examples](examples/notebooks/) - Real-world research applications

**Advanced Topics:**

- Stochastic sampling for large models
- Batch processing with DataLoaders
- GPU acceleration for faster experiments
- Multi-metric evaluation strategies
- Fault injection strategies for CNNs and RNNs

**Research Applications:**

- Study architectural robustness (skip connections, batch norm, etc.)
- Develop radiation-hardened models
- Benchmark fault tolerance across model families
- Analyze vulnerability patterns in different layers

### Common Use Cases

**1. Quick Robustness Check**

```python
# Test all critical bits quickly
critical_bits = [0, 1, 2]  # Sign + top exponent bits
for bit in critical_bits:
    results = injector.run_injector(bit_i=bit)
    print(f"Bit {bit}: {np.mean(results['criterion_score']):.2%}")
```

**2. Layer Vulnerability Analysis**

```python
# Identify most vulnerable layer - check layer names first
for layer_name, _ in model.named_parameters():
    if 'weight' in layer_name:
        results = injector.run_injector(bit_i=0, layer_name=layer_name)
        print(f"Layer {layer_name}: {np.mean(results['criterion_score']):.2%}")
```

**3. Comprehensive Robustness Profile**

```python
# Test all bits (warning: time-consuming for large models)
all_results = []
for bit in range(32):
    results = injector.run_injector(bit_i=bit)
    all_results.append(np.mean(results['criterion_score']))
    
# Find most vulnerable bits
import numpy as np
vulnerable_bits = np.argsort(all_results)[:5]
print(f"Most vulnerable bits: {vulnerable_bits}")
```

## Troubleshooting

### Issue: "No module named pytest" when running examples

**Problem:** Missing development dependencies.
**Solution:**

```bash
# Install all dependencies including testing tools
uv sync --all-extras
```

### Issue: "No module named 'testing'" import errors

**Problem:** Using older version without proper package structure.
**Solution:**

```bash
# Switch to the latest development branch
git checkout ai_refactor
git pull origin ai_refactor
uv sync --all-extras
```

### Issue: Individual test files fail with coverage errors

**Problem:** Coverage requirements too strict for single test files.
**Solution:**

```bash
# Run tests without coverage requirements
uv run pytest tests/test_injector.py --no-cov
```

### Issue: Low baseline accuracy

**Solution:** Train longer or adjust hyperparameters (learning rate, epochs, architecture).

### Issue: All SEUs show 0% accuracy

**Problem:** Model might be too sensitive or criterion incorrect.
**Solution:** Check that criterion matches task (classification vs regression), verify data format.

### Issue: SEU injection very slow

**Solution:**

- Use GPU: `device='cuda'`
- Use stochastic sampling: `injector.run_stochastic_seu(bit_i=31, p=0.01)`
- Reduce test set size

### Issue: Results seem random

**Solution:** Ensure model is in `eval()` mode, check baseline score is reasonable, verify data preprocessing.

## Getting Help

- 📝 **Documentation:** [Full API Reference](api/index.md)
- 🐛 **Issues:** [GitHub Issues](https://github.com/wd7512/seu-injection-framework/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/wd7512/seu-injection-framework/discussions)
- 📧 **Contact:** See repository maintainers

______________________________________________________________________

**Last Updated:** November 2025\
**Estimated Time:** 10-15 minutes\
**Difficulty:** Beginner\
**Version:** 1.1.10

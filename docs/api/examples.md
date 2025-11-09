# API Examples

Practical code examples demonstrating the SEU Injection Framework API.

---

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Data Input Methods](#data-input-methods)
3. [Injection Strategies](#injection-strategies)
4. [Custom Metrics](#custom-metrics)
5. [Visualization Examples](#visualization-examples)
6. [Advanced Patterns](#advanced-patterns)

---

## Quick Start Examples

### Minimal Working Example

The simplest possible SEU injection:

```python
import torch
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy

# Your trained model
model = torch.load('model.pth')

# Test data
X_test = torch.randn(100, 10)
y_test = torch.randint(0, 2, (100,))

# Run SEU injection
injector = SEUInjector(model, x=X_test, y=y_test, 
                       criterion=classification_accuracy)
results = injector.run_seu(bit_i=31)

print(f"Baseline: {injector.baseline_score:.2%}")
print(f"After SEU: {results['criterion_score'].mean():.2%}")
```

### Complete Workflow

End-to-end example with training:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy

# 1. Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 2. Define and train model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# 3. Run SEU analysis
model.eval()
injector = SEUInjector(
    trained_model=model,
    x=X_test,
    y=y_test,
    criterion=classification_accuracy,
    device='cpu'
)

# Test sign bit vulnerability
results = injector.run_seu(bit_i=31)

print(f"✅ Training complete!")
print(f"📊 Baseline accuracy: {injector.baseline_score:.2%}")
print(f"⚠️  Mean accuracy after sign bit flip: {results['criterion_score'].mean():.2%}")
print(f"📉 Worst case accuracy: {results['criterion_score'].min():.2%}")
```

---

## Data Input Methods

### Method 1: NumPy Arrays

```python
import numpy as np
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy

# NumPy data
X_np = np.random.randn(100, 10).astype(np.float32)
y_np = np.random.randint(0, 2, (100,)).astype(np.float32)

# Framework automatically converts to tensors
injector = SEUInjector(
    trained_model=model,
    x=X_np,
    y=y_np,
    criterion=classification_accuracy
)

results = injector.run_seu(bit_i=31)
```

### Method 2: PyTorch Tensors

```python
import torch
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy

# PyTorch tensors
X_torch = torch.randn(100, 10)
y_torch = torch.randint(0, 2, (100,))

injector = SEUInjector(
    trained_model=model,
    x=X_torch,
    y=y_torch,
    criterion=classification_accuracy,
    device='cuda'  # GPU acceleration
)

results = injector.run_seu(bit_i=31)
```

### Method 3: DataLoader

```python
from torch.utils.data import DataLoader, TensorDataset
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy

# Create dataset
dataset = TensorDataset(X_torch, y_torch)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Use DataLoader directly
injector = SEUInjector(
    trained_model=model,
    data_loader=loader,
    criterion=classification_accuracy,
    device='cuda'
)

results = injector.run_seu(bit_i=31)
```

**When to use each method:**
- **NumPy**: For scikit-learn workflows
- **Tensors**: For small-medium datasets that fit in memory
- **DataLoader**: For large datasets requiring batch processing

---

## Injection Strategies

### Exhaustive Testing - All Parameters

Test every parameter in the model:

```python
# Test sign bit on all parameters
results = injector.run_seu(bit_i=31)

print(f"Tested {len(results['tensor_location'])} parameters")
print(f"Mean accuracy: {results['criterion_score'].mean():.2%}")
```

### Layer-Specific Targeting

Focus on specific layers:

```python
# List all layer names
for name, param in model.named_parameters():
    print(f"Layer: {name}, Shape: {param.shape}")

# Test only first layer
results_layer1 = injector.run_seu(bit_i=31, layer_name='fc.0.weight')

# Test only bias parameters
for name, param in model.named_parameters():
    if 'bias' in name:
        results = injector.run_seu(bit_i=31, layer_name=name)
        print(f"{name}: {results['criterion_score'].mean():.2%}")
```

### Multi-Bit Analysis

Test multiple bit positions:

```python
import pandas as pd

# Test multiple bits
bit_results = []
for bit in [31, 30, 29, 23, 15, 0]:  # Sample key positions
    results = injector.run_seu(bit_i=bit)
    bit_results.append({
        'bit': bit,
        'mean': results['criterion_score'].mean(),
        'std': results['criterion_score'].std(),
        'min': results['criterion_score'].min()
    })

df = pd.DataFrame(bit_results)
print(df)
```

### Critical Bit Identification

Find the most vulnerable bits:

```python
# Test all 32 bits
all_bits = {}
for bit in range(32):
    results = injector.run_seu(bit_i=bit)
    all_bits[bit] = results['criterion_score'].mean()

# Sort by impact (lowest accuracy = highest impact)
sorted_bits = sorted(all_bits.items(), key=lambda x: x[1])

print("Top 5 most critical bits:")
for bit, acc in sorted_bits[:5]:
    impact = injector.baseline_score - acc
    print(f"Bit {bit}: {acc:.2%} (drop of {impact:.2%})")
```

---

## Custom Metrics

### F1 Score

```python
from sklearn.metrics import f1_score
import torch

def f1_score_criterion(model, x, y, device):
    """Custom F1 score criterion."""
    model.eval()
    with torch.no_grad():
        predictions = model(x.to(device)).cpu().numpy()
        y_pred_class = (predictions > 0.5).astype(int).flatten()
        y_true = y.cpu().numpy().astype(int).flatten()
    return f1_score(y_true, y_pred_class)

# Use custom criterion
injector = SEUInjector(
    trained_model=model,
    x=X_test,
    y=y_test,
    criterion=f1_score_criterion
)

results = injector.run_seu(bit_i=31)
print(f"Baseline F1: {injector.baseline_score:.3f}")
print(f"Mean F1 after SEU: {results['criterion_score'].mean():.3f}")
```

### Precision and Recall

```python
from sklearn.metrics import precision_score, recall_score

def precision_criterion(model, x, y, device):
    """Precision metric."""
    model.eval()
    with torch.no_grad():
        predictions = model(x.to(device)).cpu().numpy()
        y_pred_class = (predictions > 0.5).astype(int).flatten()
        y_true = y.cpu().numpy().astype(int).flatten()
    return precision_score(y_true, y_pred_class, zero_division=0)

def recall_criterion(model, x, y, device):
    """Recall metric."""
    model.eval()
    with torch.no_grad():
        predictions = model(x.to(device)).cpu().numpy()
        y_pred_class = (predictions > 0.5).astype(int).flatten()
        y_true = y.cpu().numpy().astype(int).flatten()
    return recall_score(y_true, y_pred_class, zero_division=0)

# Test with different metrics
injector_prec = SEUInjector(model, x=X_test, y=y_test, criterion=precision_criterion)
injector_rec = SEUInjector(model, x=X_test, y=y_test, criterion=recall_criterion)

results_prec = injector_prec.run_seu(bit_i=31)
results_rec = injector_rec.run_seu(bit_i=31)

print(f"Precision: {results_prec['criterion_score'].mean():.3f}")
print(f"Recall: {results_rec['criterion_score'].mean():.3f}")
```

### Multi-Class Top-K Accuracy

```python
def top5_accuracy(model, x, y, device):
    """Top-5 accuracy for multi-class classification."""
    model.eval()
    with torch.no_grad():
        predictions = model(x.to(device))
        _, top5 = predictions.topk(5, dim=1)
        y_expanded = y.to(device).view(-1, 1).expand_as(top5)
        correct = top5.eq(y_expanded).any(dim=1).sum().item()
    return correct / len(y)

# Use for multi-class models
injector = SEUInjector(model, x=X_test, y=y_test, criterion=top5_accuracy)
results = injector.run_seu(bit_i=31)
```

### Custom Loss Function

```python
import torch.nn.functional as F

def cross_entropy_criterion(model, x, y, device):
    """Negative cross-entropy (higher is better)."""
    model.eval()
    with torch.no_grad():
        predictions = model(x.to(device))
        loss = F.cross_entropy(predictions, y.long().to(device))
    return -loss.item()

injector = SEUInjector(model, x=X_test, y=y_test, criterion=cross_entropy_criterion)
results = injector.run_seu(bit_i=31)
```

---

## Visualization Examples

### Basic Accuracy Plot

```python
import matplotlib.pyplot as plt

# Run SEU injection
results = injector.run_seu(bit_i=31)

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(results['criterion_score'], bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=injector.baseline_score, color='green', linestyle='--', 
            linewidth=2, label='Baseline')
plt.xlabel('Accuracy After SEU')
plt.ylabel('Number of Parameters')
plt.title('Distribution of Accuracies After Sign Bit Flip')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Bit Position Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

# Test multiple bit positions
bits_to_test = [31, 30, 29, 28, 23, 15, 7, 0]
mean_accuracies = []
std_accuracies = []

for bit in bits_to_test:
    results = injector.run_seu(bit_i=bit)
    mean_accuracies.append(results['criterion_score'].mean())
    std_accuracies.append(results['criterion_score'].std())

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(bits_to_test))

ax.bar(x_pos, mean_accuracies, yerr=std_accuracies, 
       align='center', alpha=0.7, capsize=10, color='steelblue')
ax.axhline(y=injector.baseline_score, color='green', linestyle='--', 
           linewidth=2, label='Baseline')
ax.set_xticks(x_pos)
ax.set_xticklabels(bits_to_test)
ax.set_xlabel('Bit Position', fontsize=12)
ax.set_ylabel('Mean Accuracy', fontsize=12)
ax.set_title('Model Robustness Across Bit Positions', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
```

### Layer Vulnerability Heatmap

```python
import seaborn as sns
import pandas as pd

# Test each layer
layer_names = []
layer_accuracies = []

for name, param in model.named_parameters():
    if param.requires_grad and 'weight' in name:
        results = injector.run_seu(bit_i=31, layer_name=name)
        layer_names.append(name)
        layer_accuracies.append(results['criterion_score'].mean())

# Create DataFrame
df_layers = pd.DataFrame({
    'Layer': layer_names,
    'Accuracy': layer_accuracies
})

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.barh(df_layers['Layer'], df_layers['Accuracy'], color='coral')
plt.axvline(x=injector.baseline_score, color='green', linestyle='--', 
            linewidth=2, label='Baseline')
plt.xlabel('Mean Accuracy After SEU')
plt.ylabel('Layer')
plt.title('Layer-Specific Vulnerability to Sign Bit Flips')
plt.legend()
plt.tight_layout()
plt.show()
```

### Complete Robustness Profile

```python
# Test all bits and create comprehensive visualization
bit_data = []
for bit in range(32):
    results = injector.run_seu(bit_i=bit)
    bit_data.append({
        'bit': bit,
        'mean': results['criterion_score'].mean(),
        'min': results['criterion_score'].min(),
        'max': results['criterion_score'].max(),
        'std': results['criterion_score'].std()
    })

df_bits = pd.DataFrame(bit_data)

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Mean ± Std
axes[0].fill_between(df_bits['bit'], 
                      df_bits['mean'] - df_bits['std'],
                      df_bits['mean'] + df_bits['std'],
                      alpha=0.3, label='±1 Std Dev')
axes[0].plot(df_bits['bit'], df_bits['mean'], 'o-', linewidth=2, label='Mean')
axes[0].axhline(y=injector.baseline_score, color='green', linestyle='--', 
                linewidth=2, label='Baseline')
axes[0].set_xlabel('Bit Position')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Mean Accuracy ± Std Dev Across Bit Positions')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Min/Max Range
axes[1].fill_between(df_bits['bit'], df_bits['min'], df_bits['max'],
                      alpha=0.3, color='orange', label='Min-Max Range')
axes[1].plot(df_bits['bit'], df_bits['mean'], 'o-', color='blue', 
             linewidth=2, label='Mean')
axes[1].axhline(y=injector.baseline_score, color='green', linestyle='--', 
                linewidth=2, label='Baseline')
axes[1].set_xlabel('Bit Position')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Range (Min-Max) Across Bit Positions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Advanced Patterns

### Pattern 1: Batch Processing with Progress Bar

```python
from tqdm import tqdm

# Test multiple configurations with progress tracking
configurations = [
    {'bit': 31, 'layer': None},
    {'bit': 30, 'layer': None},
    {'bit': 31, 'layer': 'fc.0.weight'},
    {'bit': 31, 'layer': 'fc.2.weight'},
]

results_summary = []
for config in tqdm(configurations, desc="Running SEU injections"):
    results = injector.run_seu(bit_i=config['bit'], layer_name=config['layer'])
    results_summary.append({
        'bit': config['bit'],
        'layer': config['layer'] or 'all',
        'mean_acc': results['criterion_score'].mean(),
        'std_acc': results['criterion_score'].std()
    })

df_summary = pd.DataFrame(results_summary)
print(df_summary)
```

### Pattern 2: GPU Accelerated Large-Scale Analysis

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Prepare large dataset
large_X = torch.randn(10000, 128)
large_y = torch.randint(0, 10, (10000,))
dataset = TensorDataset(large_X, large_y)
loader = DataLoader(dataset, batch_size=512, num_workers=4, pin_memory=True)

# GPU-accelerated injection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_gpu = model.to(device)

injector_gpu = SEUInjector(
    trained_model=model_gpu,
    data_loader=loader,
    criterion=classification_accuracy,
    device=device
)

# Fast exhaustive testing
import time
start = time.time()
results = injector_gpu.run_seu(bit_i=31)
elapsed = time.time() - start

print(f"Tested {len(results['tensor_location'])} parameters in {elapsed:.2f}s")
print(f"Throughput: {len(results['tensor_location'])/elapsed:.1f} injections/sec")
```

### Pattern 3: Statistical Significance Testing

```python
from scipy import stats
import numpy as np

# Test if performance degradation is statistically significant
results = injector.run_seu(bit_i=31)
baseline = injector.baseline_score

# Calculate effect size (Cohen's d)
mean_diff = baseline - results['criterion_score'].mean()
pooled_std = np.sqrt((0 + results['criterion_score'].std()**2) / 2)
cohens_d = mean_diff / pooled_std

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(results['criterion_score'], baseline)

print(f"Baseline: {baseline:.4f}")
print(f"Mean after SEU: {results['criterion_score'].mean():.4f}")
print(f"Cohen's d: {cohens_d:.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("✅ Degradation is statistically significant (p < 0.05)")
else:
    print("⚠️  Degradation is NOT statistically significant")
```

### Pattern 4: Comparing Multiple Models

```python
# Compare robustness of different architectures
models = {
    'Small': small_model,
    'Medium': medium_model,
    'Large': large_model
}

comparison_results = []
for name, mdl in models.items():
    inj = SEUInjector(mdl, x=X_test, y=y_test, criterion=classification_accuracy)
    res = inj.run_seu(bit_i=31)
    comparison_results.append({
        'Model': name,
        'Baseline': inj.baseline_score,
        'Mean After SEU': res['criterion_score'].mean(),
        'Drop': inj.baseline_score - res['criterion_score'].mean(),
        'Worst Case': res['criterion_score'].min()
    })

df_comparison = pd.DataFrame(comparison_results)
print(df_comparison.to_string(index=False))

# Visualize comparison
df_comparison.plot(x='Model', y=['Baseline', 'Mean After SEU'], 
                   kind='bar', figsize=(10, 6))
plt.ylabel('Accuracy')
plt.title('Model Robustness Comparison')
plt.xticks(rotation=0)
plt.legend(['Baseline', 'After Sign Bit Flip'])
plt.tight_layout()
plt.show()
```

### Pattern 5: Save Results for Later Analysis

```python
import json
import pickle

# Run comprehensive analysis
all_results = {}
for bit in range(32):
    results = injector.run_seu(bit_i=bit)
    all_results[f'bit_{bit}'] = {
        'mean': float(results['criterion_score'].mean()),
        'std': float(results['criterion_score'].std()),
        'min': float(results['criterion_score'].min()),
        'max': float(results['criterion_score'].max()),
        'scores': results['criterion_score'].tolist()
    }

# Save as JSON
with open('seu_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Save raw results as pickle
with open('seu_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# Load later
with open('seu_results.json', 'r') as f:
    loaded_results = json.load(f)

print(f"Loaded results for {len(loaded_results)} bit positions")
```

---

## Real-World Use Case Examples

### Use Case 1: Space Mission Deployment

```python
# Simulate radiation environment for space deployment
import numpy as np

# Critical layers for space mission
critical_layers = ['backbone.conv1.weight', 'head.fc.weight']

# Test critical bits
space_mission_report = []
for layer in critical_layers:
    for bit in [31, 30, 29]:  # Sign + top exponent bits
        results = injector.run_seu(bit_i=bit, layer_name=layer)
        worst_case = results['criterion_score'].min()
        
        space_mission_report.append({
            'Layer': layer,
            'Bit': bit,
            'Worst Case Acc': worst_case,
            'Status': 'PASS' if worst_case > 0.7 else 'FAIL'
        })

df_report = pd.DataFrame(space_mission_report)
print("\n📡 Space Mission Readiness Report:")
print(df_report.to_string(index=False))

# Check if mission-critical threshold met
mission_ready = all(df_report['Status'] == 'PASS')
print(f"\n{'✅ MISSION READY' if mission_ready else '❌ MISSION NOT READY'}")
```

### Use Case 2: Automotive Safety Validation

```python
# Automotive safety requires 99.99% reliability
safety_threshold = 0.9999

# Test all parameters
results = injector.run_seu(bit_i=31)

# Calculate failure rate
failures = (results['criterion_score'] < safety_threshold).sum()
failure_rate = failures / len(results['criterion_score'])

print(f"🚗 Automotive Safety Analysis:")
print(f"Total parameters tested: {len(results['criterion_score'])}")
print(f"Safety threshold: {safety_threshold:.2%}")
print(f"Failures: {failures}")
print(f"Failure rate: {failure_rate:.4%}")
print(f"Status: {'✅ PASS' if failure_rate < 0.0001 else '❌ FAIL'}")
```

### Use Case 3: Nuclear Environment Hardening

```python
# Nuclear environments require testing against multiple radiation types
# Simulate different radiation-induced bit patterns

radiation_patterns = {
    'Alpha particles': [31, 30],  # High-order bits
    'Beta particles': [15, 14, 13],  # Mid-range bits
    'Gamma rays': list(range(23, 32)),  # Exponent bits
}

nuclear_results = []
for radiation_type, bits in radiation_patterns.items():
    acc_values = []
    for bit in bits:
        results = injector.run_seu(bit_i=bit)
        acc_values.append(results['criterion_score'].mean())
    
    nuclear_results.append({
        'Radiation Type': radiation_type,
        'Mean Accuracy': np.mean(acc_values),
        'Min Accuracy': np.min(acc_values),
        'Bits Tested': len(bits)
    })

df_nuclear = pd.DataFrame(nuclear_results)
print("\n☢️  Nuclear Environment Robustness Report:")
print(df_nuclear.to_string(index=False))
```

---

**Last Updated:** November 2025  
**Version:** 1.0.0

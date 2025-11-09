# SEU Injection Framework - Basic Usage Tutorial

This comprehensive tutorial guides you through the essential workflow of using the SEU Injection Framework to analyze neural network robustness under radiation-induced Single Event Upsets (SEUs). You'll learn how to set up experiments, run fault injection campaigns, and analyze results effectively.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Quick Start Example](#quick-start-example)  
3. [Understanding SEU Injection](#understanding-seu-injection)
4. [Basic Workflow](#basic-workflow)
5. [Comprehensive Example](#comprehensive-example)
6. [Result Analysis](#result-analysis)
7. [Advanced Usage Patterns](#advanced-usage-patterns)
8. [Performance Optimization](#performance-optimization)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Installation and Setup

### Prerequisites

Before using the SEU Injection Framework, ensure you have the following dependencies:

```python
# Required packages
torch >= 1.12.0      # PyTorch for neural networks
numpy >= 1.21.0      # Numerical computing
scikit-learn >= 1.0  # Machine learning metrics
tqdm                 # Progress bars
```

### Installation

Install the framework using pip (once published) or from source:

```bash
# From PyPI (future release)
pip install seu-injection-framework

# From source (current development)
git clone https://github.com/your-repo/seu-injection-framework.git
cd seu-injection-framework
pip install -e .
```

### Verify Installation

```python
import torch
from seu_injection import SEUInjector
from seu_injection.metrics.accuracy import classification_accuracy
from seu_injection.bitops.float32 import bitflip_float32_fast

print("SEU Injection Framework successfully imported!")
```

## Quick Start Example

Here's a minimal example to get you started immediately:

```python
import torch
import torch.nn as nn
from seu_injection import SEUInjector
from seu_injection.metrics.accuracy import classification_accuracy

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2),
    nn.Softmax(dim=1)
)

# Generate sample data
x_test = torch.randn(100, 10)
y_test = torch.randint(0, 2, (100,))

# Define evaluation criterion
def accuracy_criterion(model, x, y, device):
    return classification_accuracy(model, x, y, device)

# Create SEU injector
injector = SEUInjector(
    trained_model=model,
    criterion=accuracy_criterion,
    x=x_test,
    y=y_test
)

# Get baseline performance
baseline_accuracy = injector.get_criterion_score()
print(f"Baseline accuracy: {baseline_accuracy:.3f}")

# Run fault injection campaign
results = injector.run_stochastic_seu(bit_i=15, p=0.01)
print(f"Injected {len(results['tensor_location'])} faults")

# Analyze impact
accuracy_drops = [baseline_accuracy - score for score in results['criterion_score']]
critical_faults = sum(1 for drop in accuracy_drops if drop > 0.1)
print(f"Critical faults (>10% accuracy drop): {critical_faults}")
```

## Understanding SEU Injection

### What are Single Event Upsets (SEUs)?

Single Event Upsets are radiation-induced bit flips in computer memory that can corrupt neural network parameters during inference or training. The SEU Injection Framework simulates these events by systematically flipping bits in model parameters and measuring the impact on model performance.

### IEEE 754 Float32 Bit Layout

Understanding the IEEE 754 representation is crucial for effective SEU analysis:

```
Bit Position: 0    1-8        9-31
Component:   Sign  Exponent   Mantissa
Impact:      ±     ×2^±127    Precision
```

Different bit positions have dramatically different impacts:

- **Bit 0 (Sign)**: Changes positive ↔ negative values
- **Bits 1-8 (Exponent)**: Can cause massive magnitude changes (×2^±127)  
- **Bits 9-31 (Mantissa)**: Cause precision changes, impact decreases toward LSB

### Injection Strategies

The framework provides two main injection strategies:

1. **Systematic (`run_seu`)**: Exhaustive injection across all parameters
2. **Stochastic (`run_stochastic_seu`)**: Probabilistic sampling for large models

## Basic Workflow

The typical SEU analysis workflow consists of five key steps:

### Step 1: Model Preparation

```python
import torch
import torch.nn as nn

# Load your pre-trained model
model = torch.load('my_trained_model.pth')
model.eval()  # Set to evaluation mode

# Verify model is in float32 (required for bit manipulation)
for name, param in model.named_parameters():
    if param.dtype != torch.float32:
        print(f"Warning: {name} is {param.dtype}, converting to float32")
        param.data = param.data.float()
```

### Step 2: Data Preparation

```python
# Option A: Use tensors for small datasets
x_test = torch.load('test_inputs.pt')
y_test = torch.load('test_labels.pt')

# Option B: Use DataLoader for large datasets  
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
```

### Step 3: Criterion Definition

```python
# Define how to measure model performance
def accuracy_criterion(model, x, y, device):
    return classification_accuracy(model, x, y, device, batch_size=256)

# Alternative: Custom criterion function
def custom_loss_criterion(model, x, y, device):
    model.eval()
    with torch.no_grad():
        outputs = model(x.to(device))
        loss = nn.CrossEntropyLoss()(outputs, y.to(device))
        return -loss.item()  # Return negative loss (higher = better)
```

### Step 4: SEUInjector Setup

```python
# Create injector with tensor data
injector = SEUInjector(
    trained_model=model,
    criterion=accuracy_criterion,
    x=x_test,
    y=y_test,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Alternative: Create injector with DataLoader
injector_loader = SEUInjector(
    trained_model=model,
    criterion=accuracy_criterion,
    data_loader=test_loader,
    device='cuda'
)

# Verify baseline performance
baseline = injector.get_criterion_score()
print(f"Baseline performance: {baseline:.4f}")
```

### Step 5: Fault Injection Campaign

```python
# Systematic injection (small models)
systematic_results = injector.run_seu(bit_i=15)

# Stochastic injection (large models)  
stochastic_results = injector.run_stochastic_seu(bit_i=0, p=0.001)

# Layer-specific analysis
layer_results = injector.run_seu(bit_i=1, layer_name='classifier.weight')
```

## Comprehensive Example

Here's a complete end-to-end example using a realistic CNN model:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from seu_injection import SEUInjector
from seu_injection.metrics.accuracy import classification_accuracy

class SimpleCNN(nn.Module):
    """Simple CNN for demonstration purposes."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Create and "train" model (using random weights for demo)
model = SimpleCNN(num_classes=10)
model.eval()

# Generate synthetic MNIST-like data
batch_size = 1000
x_test = torch.randn(batch_size, 1, 28, 28)
y_test = torch.randint(0, 10, (batch_size,))

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

# Define comprehensive accuracy criterion
def robust_accuracy_criterion(model, x, y, device):
    """Robust accuracy criterion with error handling."""
    try:
        return classification_accuracy(model, x, y, device, batch_size=128)
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.0  # Return worst-case accuracy on error

# Create SEU injector
injector = SEUInjector(
    trained_model=model,
    criterion=robust_accuracy_criterion,
    x=x_test,
    y=y_test,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Establish baseline
baseline_accuracy = injector.get_criterion_score()
print(f"Baseline accuracy: {baseline_accuracy:.4f}")

# Comprehensive bit position analysis
bit_analysis = {}

for bit_pos in [0, 1, 8, 15, 23, 31]:  # Representative bit positions
    print(f"\nAnalyzing bit position {bit_pos}...")
    
    # Run stochastic injection
    results = injector.run_stochastic_seu(bit_i=bit_pos, p=0.001)
    
    # Calculate statistics
    scores = results['criterion_score']
    accuracy_drops = [baseline_accuracy - score for score in scores]
    
    bit_analysis[bit_pos] = {
        'num_injections': len(scores),
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'mean_drop': np.mean(accuracy_drops),
        'max_drop': max(accuracy_drops) if accuracy_drops else 0,
        'critical_faults': sum(1 for drop in accuracy_drops if drop > 0.1),
        'severe_faults': sum(1 for drop in accuracy_drops if drop > 0.5)
    }
    
    stats = bit_analysis[bit_pos]
    print(f"  Injections: {stats['num_injections']}")
    print(f"  Mean accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    print(f"  Mean drop: {stats['mean_drop']:.4f}")
    print(f"  Max drop: {stats['max_drop']:.4f}")
    print(f"  Critical faults (>10% drop): {stats['critical_faults']}")
    print(f"  Severe faults (>50% drop): {stats['severe_faults']}")

# Layer-wise vulnerability analysis
print("\nLayer-wise vulnerability analysis...")

layer_analysis = {}
target_layers = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']

for layer_name in target_layers:
    print(f"\nAnalyzing layer: {layer_name}")
    
    try:
        # Focus on exponent bit (high impact)
        layer_results = injector.run_stochastic_seu(
            bit_i=1, p=0.01, layer_name=layer_name
        )
        
        scores = layer_results['criterion_score']
        drops = [baseline_accuracy - score for score in scores]
        
        layer_analysis[layer_name] = {
            'injections': len(scores),
            'mean_drop': np.mean(drops),
            'critical_rate': sum(1 for drop in drops if drop > 0.1) / len(drops)
        }
        
        stats = layer_analysis[layer_name]
        print(f"  Injections: {stats['injections']}")
        print(f"  Mean accuracy drop: {stats['mean_drop']:.4f}")
        print(f"  Critical fault rate: {stats['critical_rate']:.3f}")
        
    except Exception as e:
        print(f"  Error analyzing {layer_name}: {e}")

print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)

# Bit position vulnerability ranking
print("\nBit Position Vulnerability Ranking:")
sorted_bits = sorted(bit_analysis.items(), 
                    key=lambda x: x[1]['mean_drop'], reverse=True)

for i, (bit_pos, stats) in enumerate(sorted_bits, 1):
    print(f"{i}. Bit {bit_pos:2d}: {stats['mean_drop']:.4f} avg drop, "
          f"{stats['critical_faults']:3d} critical faults")

# Layer vulnerability ranking  
print("\nLayer Vulnerability Ranking:")
sorted_layers = sorted(layer_analysis.items(),
                      key=lambda x: x[1]['mean_drop'], reverse=True)

for i, (layer_name, stats) in enumerate(sorted_layers, 1):
    print(f"{i}. {layer_name:15s}: {stats['mean_drop']:.4f} avg drop, "
          f"{stats['critical_rate']:.3f} critical rate")

print(f"\nBaseline accuracy: {baseline_accuracy:.4f}")
print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Result Analysis

### Understanding Injection Results

The injection methods return detailed results dictionaries:

```python
# Example result structure
results = {
    'tensor_location': [0, 1, 5, 10, ...],      # Parameter indices
    'criterion_score': [0.95, 0.94, 0.12, ...], # Performance after injection
    'layer_name': ['conv1.weight', ...],         # Layer names
    'value_before': [0.1234, ...],              # Original values
    'value_after': [-0.1234, ...]               # Values after bit flip
}
```

### Statistical Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_injection_results(results, baseline_accuracy, title="SEU Analysis"):
    """Comprehensive analysis of injection results."""
    
    scores = np.array(results['criterion_score'])
    drops = baseline_accuracy - scores
    
    # Basic statistics
    stats = {
        'total_injections': len(scores),
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'min_accuracy': np.min(scores),
        'max_accuracy': np.max(scores),
        'mean_drop': np.mean(drops),
        'max_drop': np.max(drops),
        'critical_faults': np.sum(drops > 0.1),
        'severe_faults': np.sum(drops > 0.5),
        'catastrophic_faults': np.sum(drops > 0.9)
    }
    
    # Print summary
    print(f"\n{title} Results Summary:")
    print(f"  Total injections: {stats['total_injections']}")
    print(f"  Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    print(f"  Range: [{stats['min_accuracy']:.4f}, {stats['max_accuracy']:.4f}]")
    print(f"  Mean drop: {stats['mean_drop']:.4f}")
    print(f"  Max drop: {stats['max_drop']:.4f}")
    print(f"  Critical faults (>10% drop): {stats['critical_faults']} "
          f"({stats['critical_faults']/stats['total_injections']:.2%})")
    print(f"  Severe faults (>50% drop): {stats['severe_faults']} "
          f"({stats['severe_faults']/stats['total_injections']:.2%})")
    
    return stats

# Layer-wise analysis
def analyze_by_layer(results, baseline_accuracy):
    """Analyze vulnerability by layer."""
    
    layer_stats = {}
    
    for layer_name in set(results['layer_name']):
        # Filter results for this layer
        layer_mask = np.array(results['layer_name']) == layer_name
        layer_scores = np.array(results['criterion_score'])[layer_mask]
        layer_drops = baseline_accuracy - layer_scores
        
        layer_stats[layer_name] = {
            'count': len(layer_scores),
            'mean_drop': np.mean(layer_drops),
            'max_drop': np.max(layer_drops),
            'critical_rate': np.sum(layer_drops > 0.1) / len(layer_drops)
        }
    
    # Sort by vulnerability
    sorted_layers = sorted(layer_stats.items(), 
                          key=lambda x: x[1]['mean_drop'], reverse=True)
    
    print("\nLayer Vulnerability Analysis:")
    for layer_name, stats in sorted_layers:
        print(f"  {layer_name:20s}: {stats['mean_drop']:.4f} avg drop, "
              f"{stats['critical_rate']:.3f} critical rate")
    
    return layer_stats

# Use the analysis functions
results = injector.run_stochastic_seu(bit_i=15, p=0.01)
stats = analyze_injection_results(results, baseline_accuracy)
layer_stats = analyze_by_layer(results, baseline_accuracy)
```

### Visualization

```python
import matplotlib.pyplot as plt

def plot_injection_results(results, baseline_accuracy, bit_position):
    """Create comprehensive plots of injection results."""
    
    scores = np.array(results['criterion_score'])
    drops = baseline_accuracy - scores
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'SEU Injection Analysis - Bit Position {bit_position}')
    
    # Accuracy distribution
    axes[0, 0].hist(scores, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(baseline_accuracy, color='red', linestyle='--', 
                       label=f'Baseline: {baseline_accuracy:.3f}')
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Accuracy Distribution')
    axes[0, 0].legend()
    
    # Accuracy drop distribution
    axes[0, 1].hist(drops, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0.1, color='orange', linestyle='--', label='Critical (10%)')
    axes[0, 1].axvline(0.5, color='red', linestyle='--', label='Severe (50%)')
    axes[0, 1].set_xlabel('Accuracy Drop')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Accuracy Drop Distribution')
    axes[0, 1].legend()
    
    # Accuracy vs injection order
    axes[1, 0].plot(scores, alpha=0.7)
    axes[1, 0].axhline(baseline_accuracy, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Injection Number')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy vs Injection Order')
    
    # Layer-wise vulnerability (if multiple layers)
    layer_names = list(set(results['layer_name']))
    if len(layer_names) > 1:
        layer_drops = []
        for layer in layer_names:
            layer_mask = np.array(results['layer_name']) == layer
            layer_drop = np.mean(drops[layer_mask])
            layer_drops.append(layer_drop)
        
        axes[1, 1].bar(range(len(layer_names)), layer_drops)
        axes[1, 1].set_xticks(range(len(layer_names)))
        axes[1, 1].set_xticklabels([name.split('.')[-1] for name in layer_names], 
                                   rotation=45)
        axes[1, 1].set_ylabel('Mean Accuracy Drop')
        axes[1, 1].set_title('Layer Vulnerability')
    else:
        axes[1, 1].text(0.5, 0.5, 'Single Layer\nAnalysis', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Create plots
results = injector.run_stochastic_seu(bit_i=15, p=0.01)
fig = plot_injection_results(results, baseline_accuracy, bit_position=15)
```

## Advanced Usage Patterns

### Multi-Bit Position Comparison

```python
def compare_bit_positions(injector, baseline, bit_positions, p=0.001):
    """Compare vulnerability across multiple bit positions."""
    
    comparison = {}
    
    for bit_pos in bit_positions:
        print(f"Testing bit position {bit_pos}...")
        results = injector.run_stochastic_seu(bit_i=bit_pos, p=p)
        
        scores = np.array(results['criterion_score'])
        drops = baseline - scores
        
        comparison[bit_pos] = {
            'mean_drop': np.mean(drops),
            'max_drop': np.max(drops),
            'critical_rate': np.sum(drops > 0.1) / len(drops),
            'severe_rate': np.sum(drops > 0.5) / len(drops)
        }
    
    return comparison

# Compare different bit types
bit_positions = [0, 1, 8, 15, 23, 31]  # Sign, exp MSB, exp LSB, mantissa positions
comparison = compare_bit_positions(injector, baseline_accuracy, bit_positions)

# Analyze results
print("\nBit Position Vulnerability Comparison:")
for bit_pos in bit_positions:
    stats = comparison[bit_pos]
    bit_type = "Sign" if bit_pos == 0 else "Exponent" if 1 <= bit_pos <= 8 else "Mantissa"
    print(f"Bit {bit_pos:2d} ({bit_type:8s}): {stats['mean_drop']:.4f} avg drop, "
          f"{stats['critical_rate']:.3f} critical rate")
```

### Batch Processing for Large Models

```python
def batch_injection_analysis(model, data_loader, bit_positions, device='cuda'):
    """Perform SEU analysis on large models using DataLoader."""
    
    def loader_criterion(model, data_loader, device):
        return classification_accuracy_loader(model, data_loader, device)
    
    # Create injector with DataLoader
    injector = SEUInjector(
        trained_model=model,
        criterion=loader_criterion,
        data_loader=data_loader,
        device=device
    )
    
    baseline = injector.get_criterion_score()
    print(f"Baseline accuracy: {baseline:.4f}")
    
    results = {}
    
    for bit_pos in bit_positions:
        print(f"\nProcessing bit position {bit_pos}...")
        
        # Use lower probability for large models
        injection_results = injector.run_stochastic_seu(bit_i=bit_pos, p=0.0001)
        
        scores = np.array(injection_results['criterion_score'])
        drops = baseline - scores
        
        results[bit_pos] = {
            'injections': len(scores),
            'mean_drop': np.mean(drops),
            'critical_faults': np.sum(drops > 0.1)
        }
        
        print(f"  Completed {len(scores)} injections")
        print(f"  Mean drop: {np.mean(drops):.4f}")
        print(f"  Critical faults: {np.sum(drops > 0.1)}")
    
    return results, baseline

# Example usage for large model
# large_results, large_baseline = batch_injection_analysis(
#     large_model, large_data_loader, [0, 1, 8, 15, 31]
# )
```

### Custom Criterion Functions

```python
# Loss-based criterion
def cross_entropy_criterion(model, x, y, device):
    """Use cross-entropy loss as performance metric."""
    model.eval()
    with torch.no_grad():
        outputs = model(x.to(device))
        loss = nn.CrossEntropyLoss()(outputs, y.to(device))
        return -loss.item()  # Negative loss (higher = better)

# Top-k accuracy criterion
def top_k_accuracy_criterion(model, x, y, device, k=5):
    """Use top-k accuracy instead of top-1."""
    model.eval()
    with torch.no_grad():
        outputs = model(x.to(device))
        _, predicted = outputs.topk(k, 1, True, True)
        predicted = predicted.t()
        correct = predicted.eq(y.to(device).view(1, -1).expand_as(predicted))
        top_k_acc = correct[:k].sum().float() / y.size(0)
        return top_k_acc.item()

# Confidence-based criterion  
def confidence_criterion(model, x, y, device):
    """Measure prediction confidence."""
    model.eval()
    with torch.no_grad():
        outputs = model(x.to(device))
        probabilities = F.softmax(outputs, dim=1)
        max_probs = torch.max(probabilities, dim=1)[0]
        mean_confidence = torch.mean(max_probs)
        return mean_confidence.item()

# Use custom criterion
confidence_injector = SEUInjector(
    trained_model=model,
    criterion=confidence_criterion,
    x=x_test,
    y=y_test
)
```

## Performance Optimization

### Memory Management

```python
# For large models, use smaller batch sizes
def memory_efficient_injection(model, x_test, y_test, device='cuda'):
    """Memory-efficient SEU injection for large models."""
    
    # Use smaller batch size for evaluation
    def efficient_criterion(model, x, y, device):
        return classification_accuracy(model, x, y, device, batch_size=64)
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    injector = SEUInjector(
        trained_model=model,
        criterion=efficient_criterion,
        x=x_test,
        y=y_test,
        device=device
    )
    
    # Use stochastic injection with very low probability
    results = injector.run_stochastic_seu(bit_i=15, p=0.0001)
    
    return results

# Monitor memory usage
def monitor_memory_usage():
    """Monitor GPU memory usage during injection."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

# Use throughout analysis
monitor_memory_usage()
results = memory_efficient_injection(model, x_test, y_test)
monitor_memory_usage()
```

### Parallel Processing

```python
# For systematic analysis across multiple bit positions
from multiprocessing import Pool
import pickle

def analyze_single_bit(args):
    """Analyze single bit position (for multiprocessing)."""
    bit_pos, model_path, data_path, p = args
    
    # Load model and data in worker process
    model = torch.load(model_path, map_location='cpu')
    data = torch.load(data_path)
    x_test, y_test = data['x'], data['y']
    
    # Create injector
    injector = SEUInjector(
        trained_model=model,
        criterion=lambda m, x, y, d: classification_accuracy(m, x, y, d),
        x=x_test,
        y=y_test,
        device='cpu'  # Use CPU for parallel processing
    )
    
    # Run analysis
    baseline = injector.get_criterion_score()
    results = injector.run_stochastic_seu(bit_i=bit_pos, p=p)
    
    # Return summary statistics
    scores = np.array(results['criterion_score'])
    drops = baseline - scores
    
    return {
        'bit_position': bit_pos,
        'baseline': baseline,
        'mean_drop': np.mean(drops),
        'critical_faults': np.sum(drops > 0.1)
    }

def parallel_bit_analysis(model, x_test, y_test, bit_positions, p=0.001):
    """Parallel analysis across bit positions."""
    
    # Save model and data temporarily
    model_path = 'temp_model.pth'
    data_path = 'temp_data.pth'
    
    torch.save(model, model_path)
    torch.save({'x': x_test, 'y': y_test}, data_path)
    
    # Prepare arguments for parallel processing
    args = [(bit_pos, model_path, data_path, p) for bit_pos in bit_positions]
    
    # Run parallel analysis
    with Pool(processes=min(len(bit_positions), 4)) as pool:
        results = pool.map(analyze_single_bit, args)
    
    # Clean up temporary files
    import os
    os.remove(model_path)
    os.remove(data_path)
    
    return results

# Example usage (be cautious with memory)
# parallel_results = parallel_bit_analysis(
#     model, x_test, y_test, [0, 1, 8, 15, 23, 31]
# )
```

## Best Practices

### Experimental Design

1. **Start Small**: Begin with systematic injection on small models or layers
2. **Use Stochastic for Scale**: Switch to probabilistic injection for large models
3. **Focus on Critical Bits**: Prioritize exponent bits (1-8) for high-impact analysis
4. **Layer-wise Analysis**: Compare vulnerability across different layer types
5. **Multiple Runs**: Use different random seeds for statistical confidence

### Model Preparation

```python
def prepare_model_for_seu(model):
    """Prepare model for SEU injection analysis."""
    
    # Set to evaluation mode
    model.eval()
    
    # Verify float32 precision
    non_float32_params = []
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            non_float32_params.append(name)
    
    if non_float32_params:
        print(f"Warning: Non-float32 parameters found: {non_float32_params}")
        print("Consider converting to float32 for proper bit manipulation")
    
    # Check for frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if trainable_params != total_params:
        print(f"Note: {total_params - trainable_params} frozen parameters detected")
    
    print(f"Model ready: {total_params:,} total parameters")
    
    return model

# Use before analysis
model = prepare_model_for_seu(model)
```

### Data Management

```python
def prepare_evaluation_data(x, y, validation_split=0.2, device='cuda'):
    """Prepare evaluation data with proper splits."""
    
    # Create validation split if needed
    if validation_split > 0:
        n_val = int(len(x) * validation_split)
        indices = torch.randperm(len(x))
        
        x_val, x_test = x[indices[:n_val]], x[indices[n_val:]]
        y_val, y_test = y[indices[:n_val]], y[indices[n_val:]]
        
        print(f"Split data: {len(x_val)} validation, {len(x_test)} test samples")
        return (x_val, y_val), (x_test, y_test)
    
    return (x, y)

# Prepare data with validation split
(x_val, y_val), (x_test, y_test) = prepare_evaluation_data(x_data, y_data)
```

### Result Storage

```python
import json
import datetime

def save_injection_results(results, metadata, filename=None):
    """Save injection results with metadata."""
    
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"seu_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, list):
            serializable_results[key] = value
        else:
            serializable_results[key] = str(value)
    
    # Combine with metadata
    output_data = {
        'metadata': metadata,
        'results': serializable_results,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename

# Example usage
metadata = {
    'model_name': 'SimpleCNN',
    'bit_position': 15,
    'injection_probability': 0.01,
    'baseline_accuracy': baseline_accuracy,
    'total_parameters': sum(p.numel() for p in model.parameters()),
    'device': str(injector.device)
}

results_file = save_injection_results(results, metadata)
```

## Troubleshooting

### Common Issues and Solutions

#### Memory Errors

```python
# Problem: CUDA out of memory during injection
# Solution: Reduce batch size or use CPU

def handle_memory_error(injector, bit_i, p):
    """Handle memory errors gracefully."""
    try:
        return injector.run_stochastic_seu(bit_i=bit_i, p=p)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU memory error, switching to CPU...")
            torch.cuda.empty_cache()
            
            # Move to CPU
            injector.device = torch.device('cpu')
            injector.model = injector.model.cpu()
            injector.X = injector.X.cpu()
            injector.y = injector.y.cpu()
            
            return injector.run_stochastic_seu(bit_i=bit_i, p=p)
        else:
            raise e

# Use robust injection
results = handle_memory_error(injector, bit_i=15, p=0.01)
```

#### Model Compatibility Issues

```python
def validate_model_compatibility(model):
    """Validate model compatibility with SEU injection."""
    
    issues = []
    
    # Check for unsupported layers
    unsupported_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if module.training:
                unsupported_layers.append(name)
    
    if unsupported_layers:
        issues.append(f"Normalization layers in training mode: {unsupported_layers}")
    
    # Check for complex number parameters
    complex_params = []
    for name, param in model.named_parameters():
        if param.dtype in [torch.complex64, torch.complex128]:
            complex_params.append(name)
    
    if complex_params:
        issues.append(f"Complex parameters not supported: {complex_params}")
    
    # Check for very small parameters
    tiny_params = []
    for name, param in model.named_parameters():
        if param.numel() < 10:
            tiny_params.append(f"{name}: {param.numel()}")
    
    if tiny_params:
        issues.append(f"Very small parameter tensors: {tiny_params}")
    
    if issues:
        print("Model compatibility issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Model is compatible with SEU injection")
    
    return len(issues) == 0

# Validate before analysis
is_compatible = validate_model_compatibility(model)
```

#### Performance Issues

```python
def optimize_injection_performance(model, x_test, y_test):
    """Optimize injection performance for large models."""
    
    total_params = sum(p.numel() for p in model.parameters())
    
    if total_params > 1_000_000:  # Large model
        print(f"Large model detected ({total_params:,} parameters)")
        print("Recommendations:")
        print("  - Use stochastic injection with p < 0.001")
        print("  - Consider layer-specific analysis")
        print("  - Use DataLoader for large datasets")
        print("  - Reduce evaluation batch size")
        
        # Suggested configuration
        suggested_p = min(0.001, 1000 / total_params)
        print(f"  - Suggested probability: {suggested_p:.6f}")
        
        return {
            'use_stochastic': True,
            'probability': suggested_p,
            'batch_size': 64,
            'use_dataloader': len(x_test) > 10000
        }
    
    else:  # Small model
        print(f"Small model ({total_params:,} parameters)")
        print("Can use systematic injection for complete analysis")
        
        return {
            'use_stochastic': False,
            'batch_size': 256,
            'use_dataloader': False
        }

# Get optimization suggestions
config = optimize_injection_performance(model, x_test, y_test)
```

### Debugging Injection Issues

```python
def debug_injection_step(injector, bit_i=15, debug_param_idx=0):
    """Debug individual injection steps."""
    
    print(f"Debugging injection at bit position {bit_i}")
    
    # Get model state before injection
    param_names = list(injector.model.named_parameters())
    target_name, target_param = param_names[debug_param_idx]
    
    print(f"Target parameter: {target_name}")
    print(f"Shape: {target_param.shape}")
    print(f"Device: {target_param.device}")
    print(f"Dtype: {target_param.dtype}")
    
    # Get baseline
    baseline = injector.get_criterion_score()
    print(f"Baseline score: {baseline:.6f}")
    
    # Manual injection for debugging
    original_value = target_param.data.flat[0].item()
    print(f"Original value: {original_value}")
    
    # Apply bit flip
    from seu_injection.bitops.float32 import bitflip_float32_fast
    flipped_value = bitflip_float32_fast(original_value, bit_i)
    print(f"Flipped value: {flipped_value}")
    print(f"Difference: {abs(flipped_value - original_value)}")
    
    # Inject and measure
    target_param.data.flat[0] = flipped_value
    corrupted_score = injector.get_criterion_score()
    print(f"Corrupted score: {corrupted_score:.6f}")
    print(f"Score drop: {baseline - corrupted_score:.6f}")
    
    # Restore
    target_param.data.flat[0] = original_value
    restored_score = injector.get_criterion_score()
    print(f"Restored score: {restored_score:.6f}")
    print(f"Restoration error: {abs(baseline - restored_score):.8f}")

# Debug if injection results seem wrong
debug_injection_step(injector, bit_i=15)
```

---

This tutorial provides a comprehensive foundation for using the SEU Injection Framework effectively. As you gain experience, you can customize the approaches for your specific use cases and research requirements. Remember to start with simple experiments and gradually increase complexity as you become familiar with the framework's capabilities and limitations.

For additional help and advanced examples, consult the API documentation and example notebooks in the repository.
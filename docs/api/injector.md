# SEUInjector Class Reference

## Overview

The `SEUInjector` class is the core component of the SEU Injection Framework, providing comprehensive fault injection capabilities for PyTorch neural networks. It enables systematic and stochastic Single Event Upset (SEU) simulation through precise IEEE 754 bit manipulation.

## Quick Start

```python
from seu_injection import SEUInjector
from seu_injection.metrics.accuracy import classification_accuracy

# Create injector
injector = SEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=test_data,
    y=test_labels
)

# Run fault injection
results = injector.run_seu(bit_i=15)
baseline = injector.baseline_score
```

## Class Definition

```python
class SEUInjector:
    """
    Single Event Upset (SEU) injector for PyTorch neural networks.
    
    Provides systematic and stochastic fault injection capabilities
    to study neural network robustness under radiation-induced bit flips.
    """
```

## Constructor

### `__init__(trained_model, criterion, device=None, x=None, y=None, data_loader=None)`

Initialize the SEU injector with model, data, and evaluation criterion.

**Parameters:**
- `trained_model` (torch.nn.Module): PyTorch neural network model
- `criterion` (callable): Performance evaluation function  
- `device` (Optional[Union[str, torch.device]]): Computing device
- `x` (Optional[torch.Tensor]): Input data tensor
- `y` (Optional[torch.Tensor]): Target labels tensor  
- `data_loader` (Optional[DataLoader]): PyTorch DataLoader

**Example:**
```python
# Tensor-based setup
injector = SEUInjector(
    trained_model=model,
    criterion=accuracy_criterion,
    x=test_inputs,
    y=test_labels,
    device='cuda'
)

# DataLoader-based setup
injector = SEUInjector(
    trained_model=model, 
    criterion=accuracy_criterion,
    data_loader=test_loader
)
```

## Methods

### `get_criterion_score() -> float`

Evaluate current model performance using the configured criterion.

**Returns:**
- `float`: Current model performance score

**Example:**
```python
baseline = injector.get_criterion_score()
print(f"Baseline accuracy: {baseline:.3f}")
```

### `run_seu(bit_i, layer_name=None) -> dict`

Perform systematic exhaustive SEU injection across model parameters.

**Parameters:**
- `bit_i` (int): Bit position to flip (0-31, where 0 is MSB)
- `layer_name` (Optional[str]): Target layer name (None for all layers)

**Returns:**
- `dict`: Injection results containing:
  - `tensor_location`: Parameter indices
  - `criterion_score`: Performance after each injection
  - `layer_name`: Layer names
  - `value_before`: Original parameter values
  - `value_after`: Values after bit flip

**Example:**
```python
# Systematic injection across all parameters
results = injector.run_seu(bit_i=15)

# Layer-specific injection
classifier_results = injector.run_seu(bit_i=0, layer_name='classifier.weight')

# Analyze results
scores = results['criterion_score']
accuracy_drops = [baseline - score for score in scores]
critical_faults = sum(1 for drop in accuracy_drops if drop > 0.1)
```

### `run_stochastic_seu(bit_i, p, layer_name=None) -> dict`

Perform probabilistic stochastic SEU injection for large-scale analysis.

**Parameters:**
- `bit_i` (int): Bit position to flip (0-31, where 0 is MSB)
- `p` (float): Probability of injection for each parameter [0.0, 1.0]
- `layer_name` (Optional[str]): Target layer name (None for all layers)

**Returns:**
- `dict`: Injection results (same format as `run_seu`)

**Example:**
```python
# Stochastic injection with 0.1% probability
results = injector.run_stochastic_seu(bit_i=15, p=0.001)

# Layer-specific stochastic injection
layer_results = injector.run_stochastic_seu(
    bit_i=1, p=0.01, layer_name='conv1.weight'
)

# Statistical analysis
import numpy as np
scores = np.array(results['criterion_score'])
mean_accuracy = np.mean(scores)
std_accuracy = np.std(scores)
print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
```

## Attributes

### `model` (torch.nn.Module)
The PyTorch model under test, automatically moved to target device.

### `criterion` (callable)  
Evaluation function for measuring model performance.

### `device` (torch.device)
Computing device (CPU/CUDA) used for evaluation.

### `baseline_score` (float)
Model performance without fault injection, computed during initialization.

## Usage Patterns

### Basic Fault Injection Campaign

```python
import torch
from seu_injection import SEUInjector
from seu_injection.metrics.accuracy import classification_accuracy

# Setup model and data
model = torch.load('trained_model.pth')
x_test = torch.load('test_data.pt')
y_test = torch.load('test_labels.pt')

# Create injector
injector = SEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=x_test,
    y=y_test
)

# Multi-bit analysis
bit_positions = [0, 1, 8, 15, 23, 31]  # Representative positions
results_by_bit = {}

for bit_pos in bit_positions:
    print(f"Analyzing bit position {bit_pos}...")
    results = injector.run_stochastic_seu(bit_i=bit_pos, p=0.001)
    
    # Calculate statistics
    scores = results['criterion_score']
    accuracy_drops = [injector.baseline_score - score for score in scores]
    
    results_by_bit[bit_pos] = {
        'injections': len(scores),
        'mean_accuracy': np.mean(scores),
        'critical_faults': sum(1 for drop in accuracy_drops if drop > 0.1)
    }

# Compare vulnerability across bit positions
for bit_pos, stats in results_by_bit.items():
    bit_type = "Sign" if bit_pos == 0 else "Exponent" if 1 <= bit_pos <= 8 else "Mantissa"
    print(f"Bit {bit_pos:2d} ({bit_type}): {stats['critical_faults']} critical faults")
```

### Layer-wise Vulnerability Analysis

```python
# Analyze vulnerability across different layer types
target_layers = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']
layer_vulnerability = {}

for layer_name in target_layers:
    print(f"Analyzing layer: {layer_name}")
    
    # Focus on high-impact exponent bit
    results = injector.run_stochastic_seu(bit_i=1, p=0.01, layer_name=layer_name)
    
    scores = results['criterion_score']
    drops = [injector.baseline_score - score for score in scores]
    
    layer_vulnerability[layer_name] = {
        'mean_drop': np.mean(drops),
        'critical_rate': sum(1 for drop in drops if drop > 0.1) / len(drops)
    }

# Rank layers by vulnerability
sorted_layers = sorted(layer_vulnerability.items(), 
                      key=lambda x: x[1]['mean_drop'], reverse=True)

print("\nLayer Vulnerability Ranking:")
for i, (layer_name, stats) in enumerate(sorted_layers, 1):
    print(f"{i}. {layer_name}: {stats['mean_drop']:.4f} avg drop, "
          f"{stats['critical_rate']:.3f} critical rate")
```

### Large Model Analysis

```python
from torch.utils.data import DataLoader, TensorDataset

# For large models, use DataLoader and low probability
def large_model_analysis(model, test_data, test_labels):
    
    # Create DataLoader for memory efficiency
    dataset = TensorDataset(test_data, test_labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Create injector with DataLoader
    injector = SEUInjector(
        trained_model=model,
        criterion=classification_accuracy_loader,
        data_loader=loader,
        device='cuda'
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Baseline accuracy: {injector.baseline_score:.4f}")
    
    # Use very low probability for large models
    results = injector.run_stochastic_seu(bit_i=15, p=0.0001)
    
    print(f"Completed {len(results['tensor_location'])} injections")
    
    return results

# Example usage
# large_results = large_model_analysis(large_model, x_test, y_test)
```

## Performance Considerations

### Computational Complexity

**Systematic Injection (`run_seu`)**:
- Time: O(n) where n = number of parameters
- Memory: O(1) additional memory
- Recommended for: Models with <1M parameters

**Stochastic Injection (`run_stochastic_seu`)**:
- Time: O(p×n) where p = probability, n = parameters  
- Memory: O(1) additional memory
- Recommended for: Large models, statistical analysis

### Memory Optimization

```python
# For memory-constrained environments
def memory_efficient_injector(model, x, y):
    
    # Use smaller batch sizes
    def efficient_criterion(model, x, y, device):
        return classification_accuracy(model, x, y, device, batch_size=32)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    injector = SEUInjector(
        trained_model=model,
        criterion=efficient_criterion,
        x=x, y=y,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return injector
```

## Error Handling

### Common Issues and Solutions

```python
# Handle device mismatches
try:
    injector = SEUInjector(model, criterion, x=x_test, y=y_test, device='cuda')
except RuntimeError as e:
    if "CUDA" in str(e):
        print("CUDA not available, falling back to CPU")
        injector = SEUInjector(model, criterion, x=x_test, y=y_test, device='cpu')

# Handle memory errors during injection
def robust_injection(injector, bit_i, p):
    try:
        return injector.run_stochastic_seu(bit_i=bit_i, p=p)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Memory error, reducing probability")
            return injector.run_stochastic_seu(bit_i=bit_i, p=p/10)
        raise

# Validate model compatibility
def validate_model(model):
    # Check for float32 parameters
    non_float32 = [name for name, param in model.named_parameters() 
                   if param.dtype != torch.float32]
    if non_float32:
        print(f"Warning: Non-float32 parameters: {non_float32}")
    
    # Check model is in eval mode
    if model.training:
        print("Warning: Model should be in eval mode")
        model.eval()
    
    return model

model = validate_model(model)
```

## Integration Examples

### Custom Criterion Functions

```python
# Loss-based criterion
def cross_entropy_criterion(model, x, y, device):
    model.eval()
    with torch.no_grad():
        outputs = model(x.to(device))
        loss = nn.CrossEntropyLoss()(outputs, y.to(device))
        return -loss.item()  # Negative for "higher is better"

# Confidence-based criterion
def confidence_criterion(model, x, y, device):
    model.eval()
    with torch.no_grad():
        outputs = model(x.to(device))
        probabilities = F.softmax(outputs, dim=1)
        max_probs = torch.max(probabilities, dim=1)[0]
        return torch.mean(max_probs).item()

# Use custom criterion
confidence_injector = SEUInjector(
    trained_model=model,
    criterion=confidence_criterion,
    x=x_test, y=y_test
)
```

### Result Analysis Integration

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_visualize(injector, bit_positions):
    """Comprehensive analysis with visualization."""
    
    results_summary = []
    
    for bit_pos in bit_positions:
        results = injector.run_stochastic_seu(bit_i=bit_pos, p=0.001)
        
        scores = np.array(results['criterion_score'])
        drops = injector.baseline_score - scores
        
        results_summary.append({
            'bit_position': bit_pos,
            'bit_type': "Sign" if bit_pos == 0 else "Exponent" if 1 <= bit_pos <= 8 else "Mantissa",
            'injections': len(scores),
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'mean_drop': np.mean(drops),
            'max_drop': np.max(drops),
            'critical_faults': np.sum(drops > 0.1),
            'critical_rate': np.sum(drops > 0.1) / len(drops)
        })
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results_summary)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean accuracy drop by bit position
    axes[0, 0].bar(df['bit_position'], df['mean_drop'])
    axes[0, 0].set_xlabel('Bit Position')
    axes[0, 0].set_ylabel('Mean Accuracy Drop')
    axes[0, 0].set_title('Vulnerability by Bit Position')
    
    # Critical fault rate by bit type
    bit_type_stats = df.groupby('bit_type')['critical_rate'].mean()
    axes[0, 1].bar(bit_type_stats.index, bit_type_stats.values)
    axes[0, 1].set_ylabel('Critical Fault Rate')
    axes[0, 1].set_title('Vulnerability by Bit Type')
    
    plt.tight_layout()
    plt.show()
    
    return df

# Example usage
# analysis_df = analyze_and_visualize(injector, [0, 1, 8, 15, 23, 31])
```

## See Also

- [`bitops/float32.md`](./bitops/float32.md) - IEEE 754 bit manipulation functions
- [`metrics/accuracy.md`](./metrics/accuracy.md) - Evaluation criterion functions
- [`criterion.md`](./criterion.md) - Criterion class reference
- [`../tutorials/basic_usage.md`](../tutorials/basic_usage.md) - Comprehensive tutorial

## Version History

- **v1.0**: Initial SEUInjector implementation
- **v1.1**: Added stochastic injection support
- **v1.2**: Enhanced DataLoader integration
- **v1.3**: Performance optimizations and memory efficiency improvements
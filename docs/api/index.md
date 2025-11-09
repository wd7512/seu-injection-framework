# API Reference

Welcome to the SEU Injection Framework API documentation. This reference provides detailed information about all public classes, functions, and modules.

## Quick Navigation

### Core Components
- **[SEUInjector](seu_injection.md#seuinjector)** - Main class for SEU injection experiments
- **[Metrics](seu_injection.md#metrics)** - Evaluation functions for robustness analysis
- **[Bitflip Operations](seu_injection.md#bitflip-operations)** - Low-level bit manipulation utilities

### Module Organization

```
seu_injection/
├── core/           # Core injection functionality
│   ├── injector.py    - SEUInjector class
│   └── attack.py      - Legacy attack interface
├── bitops/         # Bit manipulation operations
│   ├── float32.py     - Float32 bitflip operations
│   └── bitflip.py     - Legacy bitflip interface
├── metrics/        # Evaluation metrics
│   └── accuracy.py    - Classification accuracy
└── utils/          # Utility functions
    └── criterion.py   - Criterion wrapper utilities
```

## Getting Started

### Installation

```bash
pip install seu-injection-framework
```

Or for development:
```bash
git clone https://github.com/wd7512/seu-injection-framework.git
cd seu-injection-framework
uv sync --all-extras
```

### Basic Usage

```python
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy
import torch

# Load model and data
model = torch.load('model.pth')
x_test = torch.randn(100, 10)
y_test = torch.randint(0, 2, (100,))

# Initialize injector
injector = SEUInjector(
    model=model,
    x=x_test,
    y=y_test,
    criterion=classification_accuracy,
    device='cpu'
)

# Run SEU injection
results = injector.run_seu(bit_position=31)
print(f"Baseline: {injector.baseline_score:.2%}")
print(f"Mean after SEU: {results['criterion_score'].mean():.2%}")
```

## API Overview

### Main Classes

#### SEUInjector
The primary class for conducting SEU injection experiments.

```python
SEUInjector(model, x=None, y=None, data_loader=None, 
            criterion=None, device='cpu', batch_size=32)
```

**Key Methods:**
- `run_seu(bit_position, layer_indices=None)` - Exhaustive SEU injection
- `run_seu_stochastic(num_injections, bit_positions=None)` - Stochastic sampling
- `baseline_score` - Model accuracy without SEUs

**[Full documentation →](seu_injection.md#seuinjector)**

### Metrics Module

Pre-built evaluation functions for common tasks.

#### classification_accuracy
Compute classification accuracy with SEU-aware error handling.

```python
from seu_injection.metrics import classification_accuracy

accuracy = classification_accuracy(
    y_true=labels,
    y_pred=predictions,
    model=model,
    x=inputs,
    device='cpu'
)
```

**[Full documentation →](seu_injection.md#classification_accuracy)**

### Bitflip Operations

Low-level operations for bit manipulation in neural network parameters.

#### float32_bitflip
Fast vectorized bitflip operations for float32 tensors.

```python
from seu_injection.bitops import float32_bitflip

# Flip bit 31 (sign bit) in all values
flipped = float32_bitflip(tensor, bit_position=31)
```

**[Full documentation →](seu_injection.md#float32_bitflip)**

## Module Details

### [seu_injection](seu_injection.md)
Complete reference for the main package including:
- SEUInjector class
- Metrics functions
- Bitflip operations
- Utility functions

### [Examples](examples.md)
Code examples demonstrating:
- Basic SEU injection workflow
- Advanced configurations
- Custom metrics
- Layer-specific targeting
- Batch processing
- GPU acceleration

## API Design Principles

### 1. Simplicity First
The API is designed for ease of use:
```python
# Minimal configuration for quick experiments
injector = SEUInjector(model, x=data, y=labels, criterion=my_metric)
results = injector.run_seu(bit_position=31)
```

### 2. Flexibility
Support multiple input formats:
```python
# NumPy arrays
injector = SEUInjector(model, x=np_array, y=np_labels, criterion=metric)

# PyTorch tensors
injector = SEUInjector(model, x=torch_tensor, y=torch_labels, criterion=metric)

# DataLoader
injector = SEUInjector(model, data_loader=loader, criterion=metric)
```

### 3. Performance
Optimized for speed with vectorized operations:
- 10-100x faster than naive implementations
- GPU acceleration support
- Efficient memory usage for large models

### 4. Safety
Defensive programming with clear error messages:
```python
# Automatic validation
injector = SEUInjector(model, x=data, data_loader=loader)  
# ValueError: Cannot specify both x and data_loader

# Type checking
injector.run_seu(bit_position=35)
# ValueError: bit_position must be between 0 and 31
```

## Common Patterns

### Pattern 1: Comprehensive Robustness Profile

Test all bit positions to build a complete vulnerability map:

```python
results_all = {}
for bit in range(32):
    results = injector.run_seu(bit_position=bit)
    results_all[bit] = results['criterion_score'].mean()

# Visualize
import matplotlib.pyplot as plt
plt.plot(results_all.keys(), results_all.values())
plt.xlabel('Bit Position')
plt.ylabel('Mean Accuracy')
plt.show()
```

### Pattern 2: Layer-by-Layer Analysis

Identify which layers are most vulnerable:

```python
layer_results = {}
for idx, layer in enumerate(model.modules()):
    if isinstance(layer, nn.Linear):
        results = injector.run_seu(bit_position=31, layer_indices=[idx])
        layer_results[f"Layer {idx}"] = results['criterion_score'].mean()

print(layer_results)
```

### Pattern 3: Custom Evaluation Metric

Use domain-specific metrics:

```python
def my_custom_metric(y_true, y_pred, **kwargs):
    """Custom evaluation function."""
    # Your evaluation logic here
    return score

injector = SEUInjector(model, x=data, y=labels, criterion=my_custom_metric)
results = injector.run_seu(bit_position=31)
```

### Pattern 4: Batch Processing for Large Models

Efficient handling of large datasets:

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=128, shuffle=False)
injector = SEUInjector(model, data_loader=loader, criterion=accuracy)
results = injector.run_seu(bit_position=31)
```

## Type Hints and Static Analysis

The framework uses comprehensive type hints for better IDE support:

```python
from typing import Optional, Union, List, Callable
import torch.nn as nn

def run_seu(
    self,
    bit_position: int,
    layer_indices: Optional[List[int]] = None
) -> pd.DataFrame:
    ...
```

Enable mypy for static type checking:
```bash
mypy src/seu_injection/
```

## Error Handling

All functions provide clear, actionable error messages:

```python
# Example: Invalid bit position
injector.run_seu(bit_position=-1)
# ValueError: bit_position must be between 0 and 31, got -1

# Example: Incompatible device
injector = SEUInjector(model.to('cuda'), x=data, device='cpu')
# Warning: Model on cuda but device='cpu'. Consider moving model to CPU.
```

## Performance Considerations

### Memory Usage
- **Exhaustive testing**: O(n_params) memory
- **Stochastic sampling**: O(1) memory per injection
- **DataLoader mode**: Streaming, minimal memory overhead

### Computational Cost
- **Per-parameter injection**: ~10-100 µs (CPU), ~1-10 µs (GPU)
- **Baseline evaluation**: Depends on model and dataset size
- **Vectorized operations**: 10-100x faster than naive loops

### Optimization Tips

1. **Use GPU acceleration**: `device='cuda'`
2. **Batch processing**: Use DataLoader for large datasets
3. **Stochastic sampling**: For quick robustness estimates
4. **Layer targeting**: Focus on critical layers only

```python
# Fast approximate robustness score
results = injector.run_seu_stochastic(
    num_injections=1000,  # Sample 1000 random parameters
    bit_positions=[31, 30, 29],  # Test only critical bits
    device='cuda'
)
```

## Version Compatibility

| Framework Version | PyTorch | Python | Status |
|------------------|---------|--------|--------|
| 1.0.0 | ≥2.0.0 | ≥3.9 | ✅ Current |
| 0.0.6 (legacy) | ≥1.9.0 | ≥3.8 | ⚠️ Deprecated |

## API Stability

**Stable APIs** (guaranteed backward compatibility):
- `SEUInjector.__init__`
- `SEUInjector.run_seu`
- `classification_accuracy`
- `float32_bitflip`

**Experimental APIs** (may change):
- `run_seu_stochastic` (under development)
- Advanced targeting options

## Further Reading

- **[Quickstart Guide](../quickstart.md)** - Get started in 10 minutes
- **[Tutorials](../tutorials/basic_usage.md)** - In-depth learning resources
- **[Examples](examples.md)** - Code examples and patterns
- **[Research Paper](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars)** - Theoretical foundation

## Contributing

Found an issue or want to improve the API? See our [Development Guide](../development/README.md).

---

**Last Updated:** November 2025  
**Version:** 1.0.0 (Phase 3 Complete)  
**License:** See [LICENSE](../../LICENSE)

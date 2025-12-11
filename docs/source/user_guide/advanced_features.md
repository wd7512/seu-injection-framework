# Advanced Features

This guide covers advanced usage patterns and features of the SEU Injection Framework.

## Stochastic Injection

For large models, exhaustive injection can be computationally expensive. The `StochasticSEUInjector` allows you to sample a subset of parameters:

```python
from seu_injection.core import StochasticSEUInjector

injector = StochasticSEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=x_test,
    y=y_test
)

# Inject with 1% probability per parameter
results = injector.run_injector(bit_i=15, p=0.01)
```

### Guaranteed Minimum Injections

The `run_at_least_one_injection` parameter ensures at least one injection occurs even with very low probability:

```python
results = injector.run_injector(
    bit_i=15, 
    p=0.0001,
    run_at_least_one_injection=True  # Default is True
)
```

## Layer Filtering

You can target specific layers for injection:

```python
# Only inject into convolutional layers
results = injector.run_injector(
    bit_i=0,
    layer_filter=lambda name: 'conv' in name.lower()
)

# Only inject into the first layer
results = injector.run_injector(
    bit_i=0,
    layer_filter=lambda name: name == 'layer1.weight'
)
```

## Custom Metrics

Define custom evaluation criteria:

```python
def custom_metric(model, x, y):
    """Custom metric: return percentage of correct predictions."""
    with torch.no_grad():
        outputs = model(x)
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == y).sum().item()
        return correct / len(y)

injector = ExhaustiveSEUInjector(
    trained_model=model,
    criterion=custom_metric,
    x=x_test,
    y=y_test
)
```

## Multi-Bit Analysis

Analyze robustness across different bit positions:

```python
import pandas as pd

results_by_bit = {}
for bit_pos in range(32):
    results = injector.run_injector(bit_i=bit_pos)
    results_by_bit[bit_pos] = results

# Analyze impact by bit position
bit_analysis = pd.DataFrame({
    bit_pos: {
        'mean_accuracy': sum(results['criterion_score']) / len(results['criterion_score']),
        'min_accuracy': min(results['criterion_score']),
        'num_injections': len(results['criterion_score'])
    }
    for bit_pos, results in results_by_bit.items()
}).T
```

## Batch Processing with DataLoaders

For large datasets, use PyTorch DataLoaders:

```python
from seu_injection.metrics import classification_accuracy_loader
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(x_test, y_test)
loader = DataLoader(dataset, batch_size=32)

injector = ExhaustiveSEUInjector(
    trained_model=model,
    criterion=classification_accuracy_loader,
    x=loader,
    y=None  # Labels are in the loader
)
```

## Performance Optimization

### CUDA Acceleration

The framework automatically uses CUDA when available:

```python
# Check if CUDA is being used
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Move model and data to GPU
model = model.cuda()
x_test = x_test.cuda()
y_test = y_test.cuda()
```

### Memory Management

For very large models, consider:

1. Using stochastic injection with low probability
2. Injecting into specific layers only
3. Processing in batches with gradient disabled

```python
# Memory-efficient injection
injector = StochasticSEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=x_test,
    y=y_test
)

# Low probability, specific layers
results = injector.run_injector(
    bit_i=15,
    p=0.001,
    layer_filter=lambda name: 'fc' in name  # Only fully connected layers
)
```

## Result Analysis

### Impact Distribution

```python
import matplotlib.pyplot as plt

# Calculate fault impact
baseline = injector.baseline_score
impacts = [baseline - score for score in results['criterion_score']]

# Plot distribution
plt.hist(impacts, bins=50)
plt.xlabel('Accuracy Drop')
plt.ylabel('Frequency')
plt.title('Distribution of Fault Impacts')
plt.show()
```

### Layer-wise Analysis

```python
import pandas as pd

# Group results by layer
df = pd.DataFrame(results)
layer_impacts = df.groupby('layer_name').agg({
    'criterion_score': ['mean', 'std', 'min', 'count']
})
print(layer_impacts)
```

## Next Steps

- Explore complete examples in [Examples](examples.md)
- Review the [API Reference](../api/core.rst) for all available options
- Check [Known Issues](../known_issues.md) for current limitations

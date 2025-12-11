# Basic Usage

This guide covers the fundamental concepts and basic usage patterns of the SEU Injection Framework.

## Core Concepts

The framework is built around two main injection strategies:

1. **ExhaustiveSEUInjector**: Systematically injects bit flips into every parameter in the model
2. **StochasticSEUInjector**: Randomly samples a subset of parameters for injection

## Simple Example

Here's a minimal working example:

```python
import torch
from seu_injection.core import ExhaustiveSEUInjector
from seu_injection.metrics import classification_accuracy

# Create a simple model
model = torch.nn.Sequential(
    torch.nn.Linear(10, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2)
)

# Generate test data
x_test = torch.randn(100, 10)
y_test = torch.randint(0, 2, (100,))

# Initialize injector
injector = ExhaustiveSEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=x_test,
    y=y_test
)

# Check baseline performance
print(f"Baseline accuracy: {injector.baseline_score:.2%}")

# Inject bit flips into sign bits (bit position 0)
results = injector.run_injector(bit_i=0)
print(f"Performed {len(results['criterion_score'])} injections")
```

## Understanding Results

The `run_injector` method returns a dictionary with the following keys:

- `criterion_score`: List of model performance scores after each injection
- `layer_name`: Name of the layer where each injection occurred
- `param_idx`: Index of the parameter that was flipped
- `bit_position`: The bit position that was flipped

## IEEE 754 Bit Positions

The framework uses IEEE 754 float32 representation:

- **Bit 0**: Sign bit
- **Bits 1-8**: Exponent bits
- **Bits 9-31**: Mantissa bits

Different bit positions have different impacts on model behavior:
- Sign bit flips (bit 0) change the sign of values
- Exponent bit flips (bits 1-8) cause large magnitude changes
- Mantissa bit flips (bits 9-31) cause smaller precision changes

## Device Management

The framework automatically handles CPU and CUDA devices:

```python
# Models on GPU are automatically supported
model = model.cuda()
x_test = x_test.cuda()
y_test = y_test.cuda()

injector = ExhaustiveSEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=x_test,
    y=y_test
)
# Injections will run on GPU
```

## Next Steps

- See [Advanced Features](advanced_features.md) for stochastic injection and filtering
- Check out [Examples](examples.md) for real-world use cases
- Refer to the [API Reference on GitHub](https://github.com/wd7512/seu-injection-framework/blob/main/docs/source/api/core.rst) for detailed documentation

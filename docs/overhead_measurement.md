# Performance Overhead Measurement

This guide explains how to measure and analyze the performance overhead of SEU injection operations.

## Overview

When using the SEU injection framework, there's a computational cost associated with:
- Backing up parameter values
- Performing bit-flip operations
- Running model evaluation after each injection
- Restoring original parameter values

Understanding this overhead helps you:
- Plan computational resources for large-scale studies
- Choose between systematic vs. stochastic injection strategies
- Estimate time requirements for different network architectures

## Basic Usage

### Measuring Baseline Inference Time

```python
from seu_injection import measure_inference_time
import torch

model = # your PyTorch model
sample_input = torch.randn(1, input_size)

baseline_metrics = measure_inference_time(
    model=model,
    input_data=sample_input,
    num_iterations=100
)

print(f"Average inference time: {baseline_metrics['avg_time_ms']:.2f} ms")
```

### Calculating Complete Overhead

```python
from seu_injection import SEUInjector, calculate_overhead
from seu_injection.metrics import classification_accuracy

# Create injector
injector = SEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=x_test,
    y=y_test
)

# Calculate overhead
overhead_results = calculate_overhead(
    model=model,
    injector=injector,
    input_data=sample_input,
    bit_position=0,  # Sign bit
    num_baseline_iterations=100,
    stochastic=True,
    stochastic_probability=0.01  # 1% sampling for efficiency
)

# Print results
print(f"Baseline: {overhead_results['baseline']['avg_time_ms']:.2f} ms")
print(f"Overhead: {overhead_results['overhead_absolute_ms']:.2f} ms")
print(f"Relative: {overhead_results['overhead_relative']:.1f}%")
```

### Formatted Report

```python
from seu_injection import format_overhead_report

report = format_overhead_report(overhead_results)
print(report)
```

Output:
```
============================================================
SEU INJECTION OVERHEAD ANALYSIS
============================================================

BASELINE INFERENCE (without SEU injection):
  Average time per inference: 0.03 ms
  Total iterations: 100
  Throughput: 32724.5 inferences/sec

SEU INJECTION CAMPAIGN:
  Total injections performed: 641
  Total time: 0.85 seconds
  Average time per injection: 1.34 ms

OVERHEAD ANALYSIS:
  Absolute overhead: 1.31 ms per injection
  Relative overhead: 4292.5%
  Baseline inference: 0.03 ms
  Injection + evaluation: 1.34 ms

INTERPRETATION:
  Each SEU injection takes 4292.5% more time than baseline inference
  Throughput with injection: 745.0 injections/sec
============================================================
```

## Comparing Multiple Networks

```python
from seu_injection import benchmark_multiple_networks

networks = [
    ("Small MLP", small_model, torch.randn(1, 10)),
    ("Large CNN", large_model, torch.randn(1, 3, 32, 32)),
]

results = benchmark_multiple_networks(
    networks=networks,
    criterion=classification_accuracy,
    x_test=x_test,
    y_test=y_test,
    bit_position=0,
    num_baseline_iterations=100
)

for name, overhead in results.items():
    print(f"{name}: {overhead['overhead_relative']:.1f}% overhead")
```

## Understanding the Metrics

### Baseline Metrics
- `avg_time`: Average time per inference (seconds)
- `avg_time_ms`: Average time per inference (milliseconds)
- `total_time`: Total time for all iterations (seconds)
- `throughput`: Inferences per second

### Injection Metrics
- `num_injections`: Total number of bit flips performed
- `avg_time_per_injection`: Average time per injection (seconds)
- `total_time`: Total time for injection campaign (seconds)

### Overhead Metrics
- `overhead_absolute`: Extra time per injection compared to baseline (seconds)
- `overhead_relative`: Percentage increase over baseline
- `throughput_with_injection`: Effective injection throughput (injections/sec)

## Optimization Strategies

### 1. Use Stochastic Sampling

For large models, use stochastic sampling instead of systematic injection:

```python
overhead_results = calculate_overhead(
    model=model,
    injector=injector,
    input_data=sample_input,
    stochastic=True,
    stochastic_probability=0.01  # Only 1% of parameters
)
```

### 2. Target Specific Layers

Focus on vulnerable layers to reduce computation:

```python
overhead_results = calculate_overhead(
    model=model,
    injector=injector,
    input_data=sample_input,
    layer_name="classifier.weight",  # Specific layer
    stochastic=False
)
```

### 3. Use GPU Acceleration

Enable CUDA for faster computation:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
sample_input = sample_input.to(device)

injector = SEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=x_test.to(device),
    y=y_test.to(device),
    device=device
)
```

## Typical Overhead Values

Based on benchmarks with small-to-medium networks:

| Network Type | Parameters | Overhead | Time per Injection |
|--------------|------------|----------|-------------------|
| Small MLP    | 641        | ~4000%   | ~1.3 ms          |
| Medium MLP   | 2,817      | ~3200%   | ~1.4 ms          |
| Large MLP    | 11,265     | ~2800%   | ~1.5 ms          |
| Small CNN    | ~50K       | ~2700%   | ~40 ms           |

Note: Overhead percentage is high because baseline inference is very fast (microseconds).
The absolute overhead time is what matters for planning large studies.

## Best Practices

1. **Always measure first**: Overhead varies significantly with network architecture
2. **Use appropriate sampling**: For >10M parameters, use stochastic with p<0.01
3. **Consider batch processing**: Larger batch sizes reduce per-sample overhead
4. **Profile before large runs**: Test with small subset first to estimate total time
5. **Use GPU when available**: Can provide 10-100× speedup for large models

## Example: Estimating Study Duration

```python
# Measure overhead for your model
overhead_results = calculate_overhead(...)

# Calculate for full systematic scan (all 32 bits, all parameters)
num_parameters = sum(p.numel() for p in model.parameters())
num_bits = 32
total_injections = num_parameters * num_bits

time_per_injection = overhead_results['overhead_per_injection']
estimated_hours = (total_injections * time_per_injection) / 3600

print(f"Estimated time for full scan: {estimated_hours:.1f} hours")
print(f"Recommendation: Use stochastic sampling with p={1000/num_parameters:.4f}")
```

## See Also

- [Complete example](../examples/overhead_calculation_example.py)
- [Performance tests](../tests/benchmarks/test_performance.py)
- [API Documentation](../src/seu_injection/utils/overhead.py)

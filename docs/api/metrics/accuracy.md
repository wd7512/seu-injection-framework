# Classification Accuracy Functions Reference

## Overview

The `metrics.accuracy` module provides comprehensive classification accuracy calculation functions optimized for Single Event Upset (SEU) injection experiments. It supports multiple input formats, automatic batch processing, and both binary and multiclass classification scenarios.

## Quick Start

```python
from seu_injection.metrics.accuracy import classification_accuracy
import torch

# Basic usage
accuracy = classification_accuracy(model, x_test, y_test, device='cuda')
print(f"Model accuracy: {accuracy:.3f}")

# With DataLoader
from torch.utils.data import DataLoader
accuracy = classification_accuracy(model, test_loader, device='cuda')
```

## Functions

### Primary Interface

#### `classification_accuracy(model, x_tensor, y_true=None, device=None, batch_size=64) -> float`

Calculate classification accuracy with intelligent input type detection and optimization.

**Parameters:**
- `model` (torch.nn.Module): PyTorch neural network model to evaluate
- `x_tensor` (Union[torch.Tensor, DataLoader]): Input data or DataLoader
- `y_true` (Optional[torch.Tensor]): Target labels (required for tensors)
- `device` (Optional[Union[str, torch.device]]): Computing device
- `batch_size` (int): Batch size for tensor evaluation (ignored for DataLoader)

**Returns:**
- `float`: Classification accuracy in [0.0, 1.0]

**Example:**
```python
import torch
import torch.nn as nn
from seu_injection.metrics.accuracy import classification_accuracy

# Setup model and data
model = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=1))
x_test = torch.randn(1000, 10)
y_test = torch.randint(0, 5, (1000,))

# Basic tensor-based evaluation
accuracy = classification_accuracy(model, x_test, y_test)
print(f"Accuracy: {accuracy:.3f}")

# GPU-accelerated evaluation
if torch.cuda.is_available():
    accuracy_gpu = classification_accuracy(model, x_test, y_test, device='cuda')
    print(f"GPU accuracy: {accuracy_gpu:.3f}")

# Memory-efficient evaluation with custom batch size
accuracy_batched = classification_accuracy(
    model, x_test, y_test, device='cuda', batch_size=128
)

# DataLoader-based evaluation
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(x_test, y_test)
loader = DataLoader(dataset, batch_size=256, shuffle=False)
accuracy_loader = classification_accuracy(model, loader, device='cuda')
```

### DataLoader Evaluation

#### `classification_accuracy_loader(model, data_loader, device=None) -> float`

Memory-efficient accuracy calculation using PyTorch DataLoader with optimized batch processing.

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to evaluate
- `data_loader` (torch.utils.data.DataLoader): DataLoader yielding (x, y) batches
- `device` (Optional[Union[str, torch.device]]): Computing device

**Returns:**
- `float`: Overall accuracy across entire dataset

**Performance Features:**
- Streaming evaluation for memory efficiency
- Non-blocking GPU transfers when supported
- Automatic batch processing optimization
- Zero additional memory allocation beyond single batch

**Example:**
```python
from torch.utils.data import TensorDataset, DataLoader
from seu_injection.metrics.accuracy import classification_accuracy_loader

# Create large dataset
x_large = torch.randn(50000, 784)  # Large MNIST-like dataset
y_large = torch.randint(0, 10, (50000,))

# Memory-efficient DataLoader setup
dataset = TensorDataset(x_large, y_large)
loader = DataLoader(
    dataset, 
    batch_size=512,      # Larger batches for efficiency
    shuffle=False,       # Preserve order for reproducibility
    num_workers=4,       # Parallel data loading
    pin_memory=True      # Faster GPU transfers
)

# Evaluate with minimal memory footprint
accuracy = classification_accuracy_loader(model, loader, device='cuda')
print(f"Large dataset accuracy: {accuracy:.4f}")

# Integration with SEUInjector
from seu_injection import SEUInjector

def loader_criterion(model, data_loader, device):
    return classification_accuracy_loader(model, data_loader, device)

injector = SEUInjector(
    trained_model=model,
    criterion=loader_criterion,
    data_loader=loader
)
```

### Core Accuracy Computation

#### `multiclass_classification_accuracy(y_true, model_output) -> float`

Core accuracy computation with automatic binary/multiclass detection and robust prediction logic.

**Parameters:**
- `y_true` (np.ndarray): Ground truth labels with shape (N,)
- `model_output` (np.ndarray): Model outputs, shape (N,) or (N, K)

**Returns:**
- `float`: Classification accuracy in [0.0, 1.0]

**Classification Logic:**
- **Binary**: Uses adaptive thresholding between min/max labels
- **Multiclass**: Applies argmax over output dimensions
- **Automatic Detection**: Based on output shape and label distribution

**Example:**
```python
import numpy as np
from seu_injection.metrics.accuracy import multiclass_classification_accuracy

# Binary classification examples
y_binary = np.array([0, 1, 0, 1, 1])

# Single output (sigmoid-style)
output_sigmoid = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
acc_binary = multiclass_classification_accuracy(y_binary, output_sigmoid)
print(f"Binary accuracy: {acc_binary:.2f}")  # 0.80

# Two outputs (softmax-style)
output_softmax = np.array([
    [0.8, 0.2], [0.1, 0.9], [0.7, 0.3], 
    [0.0, 1.0], [0.2, 0.8]
])
acc_softmax = multiclass_classification_accuracy(y_binary, output_softmax)

# Multiclass classification
y_multi = np.array([0, 2, 1, 0, 2])
output_multi = np.array([
    [0.9, 0.05, 0.05],  # Pred: 0, True: 0 ✓
    [0.1, 0.1, 0.8],    # Pred: 2, True: 2 ✓  
    [0.3, 0.6, 0.1],    # Pred: 1, True: 1 ✓
    [0.2, 0.7, 0.1],    # Pred: 1, True: 0 ✗
    [0.0, 0.1, 0.9]     # Pred: 2, True: 2 ✓
])
acc_multi = multiclass_classification_accuracy(y_multi, output_multi)
print(f"Multiclass accuracy: {acc_multi:.2f}")  # 0.80

# Alternative binary label encoding
y_alt = np.array([-1, 1, -1, 1, -1])
output_alt = np.array([-0.5, 0.8, -0.2, 1.2, 0.1])
acc_alt = multiclass_classification_accuracy(y_alt, output_alt)
print(f"Alternative encoding: {acc_alt:.2f}")
```

## Usage Patterns

### SEU Injection Integration

```python
from seu_injection import SEUInjector
from seu_injection.metrics.accuracy import classification_accuracy

def create_accuracy_criterion(batch_size=256):
    """Create accuracy criterion function for SEUInjector."""
    
    def accuracy_criterion(model, x, y, device):
        return classification_accuracy(model, x, y, device, batch_size=batch_size)
    
    return accuracy_criterion

# Setup SEU injection with accuracy evaluation
model = torch.load('trained_model.pth')
x_test = torch.load('test_data.pt')
y_test = torch.load('test_labels.pt')

injector = SEUInjector(
    trained_model=model,
    criterion=create_accuracy_criterion(batch_size=128),
    x=x_test,
    y=y_test,
    device='cuda'
)

# Run fault injection analysis
baseline_accuracy = injector.get_criterion_score()
results = injector.run_stochastic_seu(bit_i=15, p=0.001)

# Analyze accuracy impact
scores = results['criterion_score']
accuracy_drops = [baseline_accuracy - score for score in scores]
critical_faults = sum(1 for drop in accuracy_drops if drop > 0.1)

print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(f"Critical faults (>10% drop): {critical_faults}/{len(scores)}")
```

### Custom Criterion Functions

```python
# Top-k accuracy criterion
def top_k_accuracy_criterion(model, x, y, device, k=5):
    """Calculate top-k accuracy instead of top-1."""
    
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
        probabilities = torch.softmax(outputs, dim=1)
        max_probs = torch.max(probabilities, dim=1)[0]
        return torch.mean(max_probs).item()

# Loss-based criterion (negative for "higher is better")
def cross_entropy_criterion(model, x, y, device):
    """Use cross-entropy loss as performance metric."""
    
    model.eval()
    with torch.no_grad():
        outputs = model(x.to(device))
        loss = nn.CrossEntropyLoss()(outputs, y.to(device))
        return -loss.item()  # Negative loss

# Use custom criteria with SEUInjector
top5_injector = SEUInjector(model, top_k_accuracy_criterion, x=x_test, y=y_test)
confidence_injector = SEUInjector(model, confidence_criterion, x=x_test, y=y_test)
```

### Comparative Analysis

```python
def compare_accuracy_metrics(model, x_test, y_test, device='cuda'):
    """Compare different accuracy-based metrics under fault injection."""
    
    # Define multiple criterion functions
    criteria = {
        'top1_accuracy': lambda m, x, y, d: classification_accuracy(m, x, y, d),
        'top5_accuracy': lambda m, x, y, d: top_k_accuracy_criterion(m, x, y, d, k=5),
        'confidence': confidence_criterion,
        'neg_loss': cross_entropy_criterion
    }
    
    results_comparison = {}
    
    for criterion_name, criterion_func in criteria.items():
        print(f"Analyzing {criterion_name}...")
        
        # Create injector for this criterion
        injector = SEUInjector(
            trained_model=model,
            criterion=criterion_func,
            x=x_test, y=y_test,
            device=device
        )
        
        # Get baseline and run injection
        baseline = injector.get_criterion_score()
        injection_results = injector.run_stochastic_seu(bit_i=15, p=0.001)
        
        # Analyze results
        scores = np.array(injection_results['criterion_score'])
        drops = baseline - scores
        
        results_comparison[criterion_name] = {
            'baseline': baseline,
            'mean_score': np.mean(scores),
            'mean_drop': np.mean(drops),
            'max_drop': np.max(drops),
            'critical_faults': np.sum(drops > 0.1 * baseline)  # 10% relative drop
        }
    
    # Print comparison
    print("\nCriterion Comparison:")
    print("Metric        | Baseline | Mean Drop | Max Drop  | Critical Faults")
    print("--------------|----------|-----------|-----------|----------------")
    
    for name, stats in results_comparison.items():
        print(f"{name:13s} | {stats['baseline']:8.4f} | {stats['mean_drop']:9.4f} | "
              f"{stats['max_drop']:9.4f} | {stats['critical_faults']:15d}")
    
    return results_comparison

# Run comparison analysis
# comparison = compare_accuracy_metrics(model, x_test, y_test)
```

### Batch Size Optimization

```python
def optimize_batch_size(model, x_test, y_test, device='cuda'):
    """Find optimal batch size for memory and performance."""
    
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    results = {}
    
    print("Batch Size Optimization:")
    print("Batch Size | Time (s) | Memory (MB) | Accuracy")
    print("-----------|----------|-------------|----------")
    
    for batch_size in batch_sizes:
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Measure time and memory
            start_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            start_time = time.time()
            
            accuracy = classification_accuracy(
                model, x_test, y_test, device=device, batch_size=batch_size
            )
            
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            elapsed_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            results[batch_size] = {
                'time': elapsed_time,
                'memory': memory_used,
                'accuracy': accuracy
            }
            
            print(f"{batch_size:10d} | {elapsed_time:8.3f} | {memory_used:11.1f} | {accuracy:8.4f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:10d} | OOM Error")
                break
            raise
    
    # Find optimal batch size (fastest that doesn't OOM)
    valid_results = {bs: stats for bs, stats in results.items() if 'time' in stats}
    optimal_batch_size = min(valid_results.keys(), key=lambda bs: valid_results[bs]['time'])
    
    print(f"\nOptimal batch size: {optimal_batch_size}")
    return optimal_batch_size, results

# Find optimal configuration
# optimal_bs, optimization_results = optimize_batch_size(model, x_test, y_test)
```

## Performance Optimization

### Memory Management

```python
def memory_efficient_accuracy(model, x, y, device, max_batch_size=256):
    """Calculate accuracy with automatic memory management."""
    
    # Start with desired batch size and reduce if OOM
    batch_size = max_batch_size
    
    while batch_size >= 1:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            accuracy = classification_accuracy(model, x, y, device, batch_size=batch_size)
            print(f"Successfully used batch size: {batch_size}")
            return accuracy
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size = batch_size // 2
                print(f"OOM error, reducing batch size to {batch_size}")
                continue
            raise
    
    raise RuntimeError("Cannot fit even batch size 1 in memory")

# Example usage
accuracy = memory_efficient_accuracy(model, x_test, y_test, 'cuda', max_batch_size=512)
```

### GPU Acceleration

```python
def gpu_optimized_accuracy(model, x, y, device='cuda'):
    """GPU-optimized accuracy calculation with performance monitoring."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Pre-allocate tensors on GPU
    model = model.to(device)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    
    # Warm up GPU (exclude from timing)
    with torch.no_grad():
        _ = model(x[:32])
    
    torch.cuda.synchronize()  # Ensure warmup completes
    
    # Timed evaluation
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    accuracy = classification_accuracy(model, x, y, device, batch_size=256)
    end_event.record()
    
    torch.cuda.synchronize()
    gpu_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
    
    print(f"GPU evaluation time: {gpu_time:.3f}s")
    print(f"Throughput: {len(x) / gpu_time:.0f} samples/sec")
    
    return accuracy

# Performance-optimized evaluation
# accuracy = gpu_optimized_accuracy(model, x_test, y_test)
```

## Error Handling

### Robust Evaluation

```python
def robust_accuracy_evaluation(model, x, y, device='cuda', fallback_device='cpu'):
    """Robust accuracy evaluation with automatic fallback."""
    
    try:
        # Primary evaluation attempt
        return classification_accuracy(model, x, y, device)
        
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e):
            print(f"GPU error ({e}), falling back to CPU")
            
            # Move everything to CPU
            model_cpu = model.cpu()
            x_cpu = x.cpu()
            y_cpu = y.cpu()
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Retry on CPU with smaller batches
            return classification_accuracy(
                model_cpu, x_cpu, y_cpu, fallback_device, batch_size=64
            )
        raise

# Example with error handling
def create_robust_criterion():
    """Create robust criterion function for SEUInjector."""
    
    def robust_criterion(model, x, y, device):
        try:
            return robust_accuracy_evaluation(model, x, y, device)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 0.0  # Worst-case fallback
    
    return robust_criterion

# Use in SEUInjector
robust_injector = SEUInjector(
    trained_model=model,
    criterion=create_robust_criterion(),
    x=x_test, y=y_test
)
```

### Input Validation

```python
def validate_accuracy_inputs(model, x, y=None):
    """Validate inputs for accuracy calculation."""
    
    issues = []
    
    # Model validation
    if not isinstance(model, torch.nn.Module):
        issues.append(f"Model must be torch.nn.Module, got {type(model)}")
    
    if model.training:
        issues.append("Model should be in eval() mode for consistent results")
    
    # Input validation
    if isinstance(x, torch.Tensor):
        if y is None:
            issues.append("y_true required when x is a tensor")
        elif len(x) != len(y):
            issues.append(f"Input/target length mismatch: {len(x)} vs {len(y)}")
    
    elif hasattr(x, '__iter__') and hasattr(x, 'dataset'):
        # DataLoader case
        if y is not None:
            issues.append("y_true should be None when using DataLoader")
    else:
        issues.append(f"Unsupported input type: {type(x)}")
    
    # Report issues
    if issues:
        print("Input validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True

# Validate before evaluation
if validate_accuracy_inputs(model, x_test, y_test):
    accuracy = classification_accuracy(model, x_test, y_test)
else:
    print("Fix validation issues before proceeding")
```

## See Also

- [`../injector.md`](../injector.md) - SEUInjector class for systematic fault injection
- [`../bitops/float32.md`](../bitops/float32.md) - IEEE 754 bit manipulation functions
- [`../../tutorials/basic_usage.md`](../../tutorials/basic_usage.md) - Complete usage examples
- [scikit-learn accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) - Underlying accuracy implementation

## Version History

- **v1.0**: Basic tensor-based accuracy calculation
- **v1.1**: Added DataLoader support for large datasets
- **v1.2**: Implemented automatic binary/multiclass detection
- **v1.3**: Enhanced memory management and GPU optimization
- **v1.4**: Added comprehensive error handling and input validation
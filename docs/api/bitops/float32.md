# Float32 Bit Manipulation Reference

## Overview

The `bitops.float32` module provides comprehensive IEEE 754 single-precision floating-point bit manipulation functions for Single Event Upset (SEU) simulation. It offers multiple implementation strategies optimized for different performance requirements and use cases.

## Quick Start

```python
from seu_injection.bitops.float32 import bitflip_float32_fast
import numpy as np

# Single value bit flip
original = 1.0
corrupted = bitflip_float32_fast(original, bit_i=0)  # Flip sign bit
print(f"{original} -> {corrupted}")  # 1.0 -> -1.0

# Array bit flip (vectorized)
weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
corrupted_weights = bitflip_float32_fast(weights, bit_i=15)
```

## IEEE 754 Float32 Layout

```
Bit Position: 0    1-8        9-31
Component:   Sign  Exponent   Mantissa  
Hex Range:   ±     00-FF      000000-7FFFFF
Impact:      ±     ×2^±127    Precision Changes
```

## Functions

### Primary Interface

#### `bitflip_float32_fast(x, bit_i=None) -> Union[float, np.ndarray]`

Intelligent bit flipping with automatic performance optimization and fallback handling.

**Parameters:**
- `x` (Union[float, np.ndarray]): Input value(s) to manipulate
- `bit_i` (Optional[int]): Bit position [0-31], None for random selection

**Returns:**
- Same type as input with specified bit flipped

**Example:**
```python
# Automatic optimization selection
scalar_result = bitflip_float32_fast(1.0, 0)        # Optimized path
array_result = bitflip_float32_fast([1.0, 2.0], 0)  # Vectorized path
large_array = bitflip_float32_fast(np.ones(10000), 15)  # High performance

# Random bit selection
np.random.seed(42)
random_corruption = bitflip_float32_fast(1.0)  # Random bit position
```

### Performance-Optimized Functions

#### `bitflip_float32_optimized(values, bit_position, inplace=False) -> Union[float, np.ndarray]`

High-performance bit flipping using direct memory manipulation and vectorization.

**Parameters:**
- `values` (Union[float, np.ndarray]): Input values
- `bit_position` (int): Bit position to flip [0-31]
- `inplace` (bool): Modify array in-place for memory efficiency

**Returns:**
- Values with specified bit flipped

**Performance:**
- Scalars: ~32x faster than string-based approach
- Arrays: ~100x+ faster due to vectorization
- Memory: Zero-copy operations when possible

**Example:**
```python
# High-performance scalar manipulation
fast_scalar = bitflip_float32_optimized(1.0, 15)

# Memory-efficient array manipulation
large_array = np.random.randn(1000000).astype(np.float32)
bitflip_float32_optimized(large_array, 0, inplace=True)  # Modifies original

# Performance comparison
import time
test_data = np.ones(100000, dtype=np.float32)

start = time.time()
result_fast = bitflip_float32_optimized(test_data, 15)
fast_time = time.time() - start

print(f"Optimized approach: {fast_time:.4f} seconds")
```

### Educational Functions

#### `bitflip_float32(x, bit_i=None) -> Union[float, np.ndarray]`

String-based bit manipulation for educational clarity and debugging.

**Parameters:**
- `x` (Union[float, np.ndarray]): Input value(s)
- `bit_i` (Optional[int]): Bit position [0-31], None for random

**Returns:**
- Values with specified bit flipped

**Use Cases:**
- Educational purposes and learning IEEE 754
- Debugging bit manipulation logic  
- Small-scale analysis where clarity matters

**Example:**
```python
# Educational bit manipulation with inspection
value = 1.0
print(f"Original binary: {float32_to_binary(value)}")

flipped = bitflip_float32(value, 15)
print(f"Flipped binary:  {float32_to_binary(flipped)}")
print(f"Difference: {abs(flipped - value)}")

# Array processing with explicit loop
weights = [1.0, 2.0, 3.0]
corrupted = bitflip_float32(weights, 0)  # Flips sign bits
```

### Utility Functions

#### `float32_to_binary(f) -> str`

Convert float32 value to IEEE 754 binary string representation.

**Parameters:**
- `f` (float): Float32 value to convert

**Returns:**
- `str`: 32-character binary string

**Example:**
```python
binary_repr = float32_to_binary(1.0)
print(binary_repr)  # '00111111100000000000000000000000'

# Analyze bit layout
sign = binary_repr[0]
exponent = binary_repr[1:9]
mantissa = binary_repr[9:32]

print(f"Sign: {sign}")
print(f"Exponent: {exponent} (decimal: {int(exponent, 2)})")
print(f"Mantissa: {mantissa}")
```

#### `binary_to_float32(binary_str) -> float`

Convert 32-bit binary string to float32 value.

**Parameters:**
- `binary_str` (str): 32-character binary string

**Returns:**
- `float`: Corresponding float32 value

**Example:**
```python
# Reconstruct value from binary
binary = '00111111100000000000000000000000'
value = binary_to_float32(binary)
print(value)  # 1.0

# Manual bit manipulation
binary_list = list(binary)
binary_list[0] = '1'  # Flip sign bit
negative_binary = ''.join(binary_list)
negative_value = binary_to_float32(negative_binary)
print(negative_value)  # -1.0
```

## Usage Patterns

### Systematic Bit Position Analysis

```python
import numpy as np
from seu_injection.bitops.float32 import bitflip_float32_fast, float32_to_binary

def analyze_bit_impacts(value=1.0):
    """Analyze impact of flipping each bit position."""
    
    print(f"Original value: {value}")
    print(f"Binary representation: {float32_to_binary(value)}")
    print("\nBit Impact Analysis:")
    print("Pos | Component | Original -> Flipped      | Magnitude Change")
    print("----|-----------|------------------------|------------------")
    
    for bit_pos in range(32):
        flipped = bitflip_float32_fast(value, bit_pos)
        magnitude_change = abs(flipped) / abs(value) if value != 0 else float('inf')
        
        if bit_pos == 0:
            component = "Sign"
        elif 1 <= bit_pos <= 8:
            component = "Exponent"
        else:
            component = "Mantissa"
        
        print(f"{bit_pos:2d}  | {component:9s} | {value:8.3f} -> {flipped:8.3f} | {magnitude_change:8.2e}")

# Run analysis
analyze_bit_impacts(1.0)
```

### Performance Benchmarking

```python
import time
import numpy as np

def benchmark_implementations():
    """Benchmark different bitflip implementations."""
    
    # Test data
    sizes = [100, 1000, 10000, 100000]
    bit_position = 15
    
    print("Performance Benchmark - Bit Position 15")
    print("Size     | String-based | Optimized   | Fast (Auto) | Speedup")
    print("---------|--------------|-------------|-------------|--------")
    
    for size in sizes:
        test_data = np.ones(size, dtype=np.float32)
        
        # String-based approach
        start = time.time()
        result1 = bitflip_float32(test_data, bit_position)
        string_time = time.time() - start
        
        # Optimized approach  
        start = time.time()
        result2 = bitflip_float32_optimized(test_data, bit_position)
        opt_time = time.time() - start
        
        # Fast (auto-selecting) approach
        start = time.time()
        result3 = bitflip_float32_fast(test_data, bit_position)
        fast_time = time.time() - start
        
        # Verify results are identical
        assert np.allclose(result1, result2)
        assert np.allclose(result2, result3)
        
        speedup = string_time / opt_time
        
        print(f"{size:8d} | {string_time:8.4f}s | {opt_time:8.4f}s | {fast_time:8.4f}s | {speedup:6.1f}x")

# Run benchmark
benchmark_implementations()
```

### Statistical Fault Analysis

```python
def statistical_fault_analysis(original_weights, n_samples=1000):
    """Statistical analysis of fault impacts across bit positions."""
    
    results = {}
    
    for bit_pos in range(32):
        # Generate multiple random faults at this bit position
        fault_impacts = []
        
        for _ in range(n_samples):
            # Select random parameter
            idx = np.random.randint(0, len(original_weights))
            original_val = original_weights[idx]
            
            # Apply bit flip
            corrupted_val = bitflip_float32_fast(original_val, bit_pos)
            
            # Calculate relative impact
            if original_val != 0:
                relative_change = abs(corrupted_val - original_val) / abs(original_val)
            else:
                relative_change = abs(corrupted_val)
            
            fault_impacts.append(relative_change)
        
        # Statistical summary
        results[bit_pos] = {
            'mean_impact': np.mean(fault_impacts),
            'std_impact': np.std(fault_impacts),
            'max_impact': np.max(fault_impacts),
            'median_impact': np.median(fault_impacts)
        }
    
    return results

# Example usage
weights = np.random.randn(10000).astype(np.float32)
stats = statistical_fault_analysis(weights)

# Print summary
print("Statistical Fault Impact Analysis")
print("Bit | Component | Mean Impact | Max Impact  | Std Dev")
print("----|-----------|-------------|-------------|--------")
for bit_pos in range(0, 32, 4):  # Sample every 4th bit
    s = stats[bit_pos]
    comp = "Sign" if bit_pos == 0 else "Exp" if bit_pos <= 8 else "Man"
    print(f"{bit_pos:2d}  | {comp:9s} | {s['mean_impact']:10.2e} | {s['max_impact']:10.2e} | {s['std_impact']:7.2e}")
```

## Advanced Usage

### Custom Bit Manipulation Patterns

```python
def multi_bit_corruption(value, bit_positions):
    """Apply multiple bit flips to simulate multi-bit upsets."""
    
    corrupted = value
    for bit_pos in bit_positions:
        corrupted = bitflip_float32_fast(corrupted, bit_pos)
    
    return corrupted

# Example: Simulate adjacent bit errors
original = 1.0
adjacent_bits = [14, 15, 16]  # Adjacent mantissa bits
multi_corrupted = multi_bit_corruption(original, adjacent_bits)

print(f"Original: {original}")
print(f"Multi-bit corrupted: {multi_corrupted}")
print(f"Impact: {abs(multi_corrupted - original) / abs(original):.2e}")
```

### Memory-Efficient Large Array Processing

```python
def chunk_process_large_array(large_array, bit_position, chunk_size=10000):
    """Process very large arrays in chunks to manage memory."""
    
    n_elements = len(large_array)
    corrupted_array = np.zeros_like(large_array)
    
    for start_idx in range(0, n_elements, chunk_size):
        end_idx = min(start_idx + chunk_size, n_elements)
        chunk = large_array[start_idx:end_idx]
        
        # Process chunk
        corrupted_chunk = bitflip_float32_optimized(chunk, bit_position)
        corrupted_array[start_idx:end_idx] = corrupted_chunk
        
        # Optional: Clear chunk from memory
        del chunk, corrupted_chunk
    
    return corrupted_array

# Example with very large array
very_large_array = np.random.randn(10000000).astype(np.float32)
result = chunk_process_large_array(very_large_array, 15, chunk_size=50000)
```

## Performance Characteristics

### Computational Complexity

| Function | Single Value | Array (n elements) | Memory |
|----------|--------------|-------------------|--------|
| `bitflip_float32` | O(1) | O(n) | O(n) |  
| `bitflip_float32_optimized` | O(1) | O(n) | O(1) additional |
| `bitflip_float32_fast` | O(1) | O(n) | O(1) additional |

### Memory Usage Patterns

```python
import psutil
import os

def measure_memory_usage():
    """Measure memory usage of different approaches."""
    
    process = psutil.Process(os.getpid())
    
    # Large test array
    test_size = 1000000
    test_data = np.random.randn(test_size).astype(np.float32)
    
    initial_memory = process.memory_info().rss / 1024**2  # MB
    
    # In-place optimization
    bitflip_float32_optimized(test_data, 15, inplace=True)
    inplace_memory = process.memory_info().rss / 1024**2
    
    # Copy-based approach
    result_copy = bitflip_float32_optimized(test_data, 15, inplace=False)
    copy_memory = process.memory_info().rss / 1024**2
    
    print(f"Initial memory: {initial_memory:.1f} MB")
    print(f"In-place memory: {inplace_memory:.1f} MB")
    print(f"Copy-based memory: {copy_memory:.1f} MB")
    print(f"Memory overhead (copy): {copy_memory - initial_memory:.1f} MB")

# measure_memory_usage()
```

## Error Handling

### Input Validation

```python
def safe_bitflip(value, bit_position):
    """Bit flip with comprehensive error handling."""
    
    try:
        # Validate bit position
        if not isinstance(bit_position, int) or not (0 <= bit_position <= 31):
            raise ValueError(f"Bit position must be integer in [0, 31], got {bit_position}")
        
        # Validate input type
        if not isinstance(value, (int, float, np.ndarray, list)):
            raise TypeError(f"Unsupported input type: {type(value)}")
        
        # Perform bit flip
        return bitflip_float32_fast(value, bit_position)
        
    except Exception as e:
        print(f"Bit flip error: {e}")
        return value  # Return original on error

# Example usage
safe_result = safe_bitflip(1.0, 15)  # Normal operation
error_result = safe_bitflip(1.0, 35)  # Invalid bit position
```

### Numerical Precision Issues

```python
def validate_bit_flip_precision():
    """Validate bit flip operations maintain precision."""
    
    test_values = [1.0, -1.0, 0.0, float('inf'), -float('inf')]
    
    print("Precision Validation:")
    print("Value    | Bit | Result     | Restored   | Error")
    print("---------|-----|------------|------------|------")
    
    for value in test_values:
        if not np.isfinite(value) and value != 0.0:
            continue  # Skip special values for this test
            
        for bit_pos in [0, 15, 31]:  # Test key positions
            # Flip and restore
            flipped = bitflip_float32_fast(value, bit_pos)
            restored = bitflip_float32_fast(flipped, bit_pos)
            
            error = abs(restored - value) if np.isfinite(value) else 0
            
            print(f"{value:8.3f} | {bit_pos:2d}  | {flipped:10.6f} | {restored:10.6f} | {error:.2e}")

validate_bit_flip_precision()
```

## Integration Examples

### PyTorch Integration

```python
import torch
import torch.nn as nn

class SEUCorruptedLinear(nn.Linear):
    """Linear layer with SEU corruption capability."""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.corruption_enabled = False
        self.bit_position = 15
        self.corruption_probability = 0.001
    
    def enable_corruption(self, bit_pos=15, prob=0.001):
        self.corruption_enabled = True
        self.bit_position = bit_pos
        self.corruption_probability = prob
    
    def forward(self, input):
        weight = self.weight
        
        if self.corruption_enabled and self.training:
            # Apply stochastic corruption during training
            corruption_mask = torch.rand_like(weight) < self.corruption_probability
            
            if corruption_mask.any():
                weight_np = weight.detach().cpu().numpy()
                mask_np = corruption_mask.cpu().numpy()
                
                # Apply bit flips where mask is True
                corrupted_indices = np.where(mask_np)
                for idx in zip(*corrupted_indices):
                    weight_np[idx] = bitflip_float32_fast(weight_np[idx], self.bit_position)
                
                weight = torch.from_numpy(weight_np).to(weight.device)
        
        return nn.functional.linear(input, weight, self.bias)

# Example usage
layer = SEUCorruptedLinear(784, 10)
layer.enable_corruption(bit_pos=15, prob=0.01)

# Use in model
model = nn.Sequential(
    SEUCorruptedLinear(784, 128),
    nn.ReLU(),
    SEUCorruptedLinear(128, 10)
)
```

## See Also

- [`../injector.md`](../injector.md) - SEUInjector class for systematic fault injection
- [`../metrics/accuracy.md`](../metrics/accuracy.md) - Evaluation metrics
- [`../../tutorials/basic_usage.md`](../../tutorials/basic_usage.md) - Complete usage examples
- [IEEE 754 Standard](https://en.wikipedia.org/wiki/IEEE_754) - Floating-point arithmetic standard

## Version History

- **v1.0**: Basic string-based bit manipulation
- **v1.1**: Added optimized direct memory manipulation  
- **v1.2**: Introduced intelligent auto-selection (`bitflip_float32_fast`)
- **v1.3**: Enhanced vectorization and memory efficiency
- **v1.4**: Added comprehensive error handling and validation
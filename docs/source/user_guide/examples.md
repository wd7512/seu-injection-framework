# Examples

This page provides links to complete, runnable examples demonstrating various use cases of the SEU Injection Framework.

## Basic Examples

### CNN Robustness Analysis

A complete example showing how to evaluate a CNN's robustness to SEU:

**Location**: [`examples/basic_cnn_robustness.py`](https://github.com/wd7512/seu-injection-framework/blob/main/examples/basic_cnn_robustness.py)

This example demonstrates:
- Training a simple CNN on MNIST
- Running exhaustive SEU injection across all bit positions
- Analyzing results and generating robustness metrics
- Creating visualizations of fault impact

### Architecture Comparison

Compare the robustness of different neural network architectures:

**Location**: [`examples/architecture_comparison.py`](https://github.com/wd7512/seu-injection-framework/blob/main/examples/architecture_comparison.py)

This example shows:
- Comparing multiple architectures (CNN vs. MLP)
- Systematic robustness evaluation
- Statistical comparison of fault tolerance
- Best practices for comparative studies

## Advanced Examples

### Interactive Notebook

An interactive Jupyter notebook with step-by-step SEU injection:

**Location**: [`examples/Example_Attack_Notebook.ipynb`](https://github.com/wd7512/seu-injection-framework/blob/main/examples/Example_Attack_Notebook.ipynb)

Features:
- Interactive exploration of injection parameters
- Real-time visualization of results
- Detailed explanations of each step
- Customizable for your own models

### ShipsNet Research Example

A complete research-grade example using the ShipsNet dataset:

**Location**: [`examples/shipsnet/`](https://github.com/wd7512/seu-injection-framework/tree/main/examples/shipsnet)

This comprehensive example includes:
- Dataset download and preprocessing
- CNN architecture design for satellite imagery
- Exhaustive robustness evaluation
- Publication-ready analysis and visualization
- Reproducible research workflow

See the README in the shipsnet folder for detailed instructions.

## Running the Examples

### Prerequisites

Install with examples dependencies:

```bash
pip install "seu-injection-framework[examples,analysis,vision]"
```

### Basic CNN Robustness

```bash
cd examples
python basic_cnn_robustness.py
```

Expected output:
- Baseline model accuracy
- Injection results for all 32 bit positions
- Statistical summary of robustness
- Saved plots in `results/` directory

### Architecture Comparison

```bash
cd examples
python architecture_comparison.py
```

Expected output:
- Training results for each architecture
- Comparative robustness analysis
- Statistical significance testing
- Comparative visualizations

### Interactive Notebook

```bash
pip install "seu-injection-framework[notebooks]"
cd examples
jupyter notebook Example_Attack_Notebook.ipynb
```

Follow the notebook cells to explore SEU injection interactively.

## Example Outputs

### Typical Results Structure

```python
results = {
    'criterion_score': [0.95, 0.94, 0.92, ...],  # Performance after each injection
    'layer_name': ['layer1.weight', ...],          # Affected layer
    'param_idx': [0, 1, 2, ...],                   # Parameter index
    'bit_position': [15, 15, 15, ...]              # Bit that was flipped
}
```

### Analysis Workflow

```python
# 1. Run injection
results = injector.run_injector(bit_i=15)

# 2. Calculate impacts
baseline = injector.baseline_score
impacts = [baseline - score for score in results['criterion_score']]

# 3. Statistical analysis
import numpy as np
mean_impact = np.mean(impacts)
std_impact = np.std(impacts)
max_impact = np.max(impacts)

print(f"Mean accuracy drop: {mean_impact:.2%}")
print(f"Std deviation: {std_impact:.2%}")
print(f"Worst case drop: {max_impact:.2%}")
```

## Customizing Examples

All examples are designed to be easily customized:

1. **Replace the model**: Use your own PyTorch model
2. **Change the dataset**: Use your own data
3. **Modify injection parameters**: Adjust bit positions, layers, probabilities
4. **Customize metrics**: Define domain-specific evaluation criteria
5. **Extend analysis**: Add your own visualization and statistical tests

## Contributing Examples

We welcome contributions of new examples! If you have a use case that would benefit others:

1. Create a clear, documented example
2. Include sample data or data loading instructions
3. Add comments explaining key steps
4. Submit a pull request

See [Contributing Guide](../contributing.md) for details.

## Additional Resources

- [Basic Usage Guide](basic_usage.md) - Core concepts and simple patterns
- [Advanced Features](advanced_features.md) - Advanced techniques and optimization
- [API Reference](../api/core.rst) - Complete API documentation
- [Research Paper](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars) - Academic background

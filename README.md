# SEU Injection Framework

A Python framework for Single Event Upset (SEU) injection in neural networks, designed for studying model robustness in harsh environments such as space, nuclear, and high-radiation applications.

**Research Paper:** [A Framework for Developing Robust Machine Learning Models in Harsh Environments: A Review of CNN Design Choices](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars)

## Quick Start

### Installation

This project uses [UV](https://github.com/astral-sh/uv) for fast, reliable package management:

```bash
# Install UV (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix/macOS
# or
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone and install
git clone https://github.com/wd7512/seu-injection-framework.git
cd seu-injection-framework
uv sync --all-extras  # Install all dependencies
```

### Basic Usage

```python
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy
import torch

# Load your trained model
model = torch.load('your_model.pth')

# Set up SEU injection
injector = SEUInjector(model, x=test_data, y=test_labels, criterion=classification_accuracy)

# Run SEU analysis
results = injector.run_seu(bit_position=0)  # Test sign bit flips
print(f"Baseline accuracy: {injector.baseline_score}")
print(f"Mean post-SEU accuracy: {results['criterion_score'].mean()}")
```

For detailed examples, see the `docs/examples/` directory.

## Features

- **High-performance SEU injection** with 10-100x optimized bitflip operations
- **Multiple injection strategies**: exhaustive and stochastic sampling  
- **GPU acceleration** support via CUDA
- **Layer-specific targeting** for focused robustness analysis
- **Comprehensive test coverage** with 109 tests and 94% code coverage
- **Support for NN, CNN, and RNN architectures**
- **CI/CD pipeline** with automated testing across platforms
- **Enterprise-grade codebase** with zero linting violations and automated quality enforcement

## Research Applications

This framework enables researchers to:
- Study fault propagation in neural networks under radiation
- Evaluate robustness of different CNN architectural choices
- Develop radiation-hardened models for space applications  
- Benchmark fault tolerance across model types
- Simulate harsh environment conditions for ML deployment

## Development Status

**Current Status**: Phase 3 Complete - Production-Ready Framework ✅

### Phase 3 ✅ (November 2025) - Performance Optimization & Quality Complete
- **10-100x performance improvement** in bitflip operations via NumPy vectorization
- **109 comprehensive tests** with **94% code coverage**
- **Modern package structure** with `src/seu_injection/` layout  
- **CI/CD pipeline** with cross-platform testing (Ubuntu/Windows/macOS)
- **Zero linting violations** with automated code quality enforcement
- **Zero breaking changes** - full backward compatibility maintained
- **Enterprise-grade codebase** with modern tooling (UV, pytest, ruff, mypy)

### Previous Milestones
- **Phase 1**: Modern tooling and comprehensive testing
- **Phase 2**: Package structure migration and optimization
- **Legacy versions (v0.0.1-0.0.6)**: Research prototype development

For detailed development history, see `docs/development/MIGRATION_HISTORY.md`. 

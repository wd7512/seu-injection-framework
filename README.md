# SEU Injection Framework

[![PyPI version](https://img.shields.io/pypi/v/seu-injection-framework.svg)](https://pypi.org/project/seu-injection-framework/)
[![Python versions](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://github.com/wd7512/seu-injection-framework)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-109%20passed-green)](https://github.com/wd7512/seu-injection-framework)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)](https://github.com/wd7512/seu-injection-framework)

A Python framework for **Single Event Upset (SEU) injection** in neural networks for robustness analysis in harsh environments.

**📖 [Documentation](docs/)** | **🚀 [Quick Start](docs/quickstart.md)** | **🔬 [Research Paper](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars)**

## Installation

**Option 1: Install from PyPI (Recommended)**

```bash
# Minimal core dependencies (PyTorch, NumPy, SciPy, tqdm)
pip install seu-injection-framework

# With analysis tools (scikit-learn, pandas, matplotlib, seaborn)
pip install "seu-injection-framework[analysis]"

# Everything (development, notebooks, vision models, docs)
pip install "seu-injection-framework[all]"
```

**Option 2: Install from Source (Development)**

```bash
git clone https://github.com/wd7512/seu-injection-framework.git
cd seu-injection-framework

# Using uv (recommended for development)
uv sync --extra dev --extra analysis --extra vision --extra notebooks

# Or using pip
pip install -e ".[all]"
```

**GPU Support (Optional):**

If you need CUDA-enabled PyTorch, install it first:

```bash
# Example for CUDA 12.x (adjust for your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install seu-injection-framework
```

**Verify Installation:**

```bash
python -c "from seu_injection import SEUInjector; print('✅ Ready')"
```

> **Having issues?** See [`docs/installation.md`](docs/installation.md) for troubleshooting.

## Quick Example

```python
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy

# Initialize with your model and test data
injector = SEUInjector(
    trained_model=model,
    criterion=classification_accuracy,
    x=x_test,
    y=y_test
)

# Run systematic bit flip injection
results = injector.run_seu(bit_i=0)  # Test sign bit
print(f"Baseline: {injector.baseline_score:.2%}")
```

**More examples:** [`examples/`](examples/) | **📖 Tutorial:** [`docs/quickstart.md`](docs/quickstart.md)

## ✨ Features

- **High-Performance**: Optimized bit manipulation (10-100x speedup)
- **Flexible**: Systematic or stochastic injection modes
- **GPU Accelerated**: Full CUDA support
- **Production Ready**: 94% test coverage, multi-platform
- **Research Focused**: Space, nuclear, and reliability applications

## 🤝 Contributing & Support

- **Documentation:** [`docs/`](docs/)
- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Issues:** [GitHub Issues](https://github.com/wd7512/seu-injection-framework/issues)
- **Contact:** wwdennis.home@gmail.com

## 📝 Citation

```bibtex
@software{seu_injection_framework,
  author = {William Dennis},
  title = {SEU Injection Framework},
  year = {2025},
  url = {https://github.com/wd7512/seu-injection-framework},
  version = {1.1.7}
}
```

## 🚀 Future Enhancements

Planned improvements for upcoming releases:

### **Documentation**
- 📚 **ReadTheDocs Site**: Comprehensive online documentation with API reference, tutorials, and research guides
- 📝 **Advanced Examples**: More complex use cases including custom metrics and multi-architecture comparisons
- 🎓 **Video Tutorials**: Step-by-step video guides for common workflows

### **Features**
- 🔄 **Additional Fault Models**: Support for multi-bit upsets, stuck-at faults, and transient errors
- 📊 **Enhanced Visualization**: Built-in plotting utilities for fault injection results
- 🎯 **Layer Importance Analysis**: Automatic identification of critical layers
- 🚀 **Performance Optimizations**: Further speedups for large-scale campaigns

### **Integration**
- 🐳 **Docker Images**: Pre-configured containers for reproducible experiments
- ☁️ **Cloud Support**: Integration with cloud platforms for distributed fault injection
- 🔌 **Framework Extensions**: Support for TensorFlow, JAX, and other ML frameworks

**Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md) or reach out at wwdennis.home@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for the research community studying neural network robustness in harsh environments.*

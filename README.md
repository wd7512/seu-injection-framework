# SEU Injection Framework

[![PyPI version](https://img.shields.io/pypi/v/seu-injection-framework.svg)](https://pypi.org/project/seu-injection-framework/)
[![Python versions](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://github.com/wd7512/seu-injection-framework)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-109%20passed-green)](https://github.com/wd7512/seu-injection-framework)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)](https://github.com/wd7512/seu-injection-framework)

A Python framework for **Single Event Upset (SEU) injection** in neural networks, designed for systematic robustness analysis in harsh environments including space missions, nuclear facilities, and radiation-prone applications.

**🔬 Research Paper:** [*A Framework for Developing Robust Machine Learning Models in Harsh Environments*](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars)

Please reach out to me if you find this interesting!

## 🚀 Quick Start

### Installation (PyPI)

Install the minimal core (fast, few dependencies):
```bash
pip install seu-injection-framework
```

Install with extended analysis stack (metrics, plots, data science helpers):
```bash
pip install "seu-injection-framework[analysis]"
```

Install everything (development, notebooks, vision models, docs toolchain):
```bash
pip install "seu-injection-framework[all]"
```

If you need GPU-specific PyTorch wheels, install PyTorch first following
the official instructions (e.g. CUDA):
```bash
# Example for CUDA 12.x (adjust per your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install seu-injection-framework
```

### Development Setup (from source with uv)
```bash
git clone https://github.com/wd7512/seu-injection-framework.git
cd seu-injection-framework
uv sync --extra dev --extra analysis --extra vision --extra notebooks
```

### Verify Installation
```bash
python -c "from seu_injection import SEUInjector; print('✅ SEU Injection Framework ready')"
```

### 🚨 Common Setup Issues & Solutions

<details>
<summary><b>❌ "No module named pytest" or test failures</b></summary>

**Problem**: You ran `uv sync` without the `--all-extras` flag, so development dependencies aren't installed.

**Solution**:
```bash
# Install all dependencies including testing tools
uv sync --all-extras

# Or specifically install dev dependencies
uv sync --extra dev
```
</details>

<details>
<summary><b>❌ "No module named 'testing'" import errors</b></summary>

**Problem**: Older version of the repository missing the testing package structure.

**Solution**:
```bash
# Make sure you're on the latest branch
git checkout ai_refactor
git pull origin ai_refactor

# Reinstall dependencies
uv sync --all-extras
```
</details>

<details>
<summary><b>❌ Individual test files failing with coverage errors</b></summary>

**Problem**: Running single test files with pytest may fail coverage thresholds.

**Solution**:
```bash
# Run individual tests without coverage requirements
uv run pytest tests/test_injector.py --no-cov

# Or run the full test suite which meets coverage requirements
uv run pytest tests/
```

**Note**: The framework uses an embedded TODO system throughout the codebase to track improvements and optimizations. These are normal and indicate active development priorities rather than bugs.
</details>

<details>
<summary><b>❌ PyTorch installation issues</b></summary>

**Problem**: PyTorch might not install correctly on some systems.

**Solution**:
```bash
# Force reinstall PyTorch
uv sync --all-extras --reinstall

# Or install PyTorch manually first
pip install torch torchvision
uv sync --all-extras
```
</details>

> **💡 Tip**: Always use `uv run` before commands to ensure you're using the correct virtual environment.

> **Note**: This README reflects PyPI distribution. Source installs remain fully supported.

### Basic Usage

```python
import torch
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy

# Create a simple model and test data
model = torch.nn.Sequential(
    torch.nn.Linear(10, 64),
    torch.nn.ReLU(), 
    torch.nn.Linear(64, 2)
)
x_test = torch.randn(100, 10)
y_test = torch.randint(0, 2, (100,))

# Initialize SEU injector
injector = SEUInjector(
    trained_model=model,
    criterion=classification_accuracy, 
    x=x_test,
    y=y_test
)

# Check baseline performance
print(f"Baseline accuracy: {injector.baseline_score:.2%}")

# Inject bit flips into sign bits (bit position 0)
results = injector.run_seu(bit_i=0)
print(f"Performed {len(results['criterion_score'])} injections")

# Sample some results
fault_impacts = [injector.baseline_score - score for score in results['criterion_score']]
print(f"Average accuracy drop: {sum(fault_impacts)/len(fault_impacts):.1%}")
```

> **💡 Need a full tutorial?** See [`docs/quickstart.md`](docs/quickstart.md) for a complete 10-minute walkthrough.

### 📚 Complete Examples

- **Basic CNN Robustness**: [`examples/basic_cnn_robustness.py`](examples/basic_cnn_robustness.py)
- **Architecture Comparison**: [`examples/architecture_comparison.py`](examples/architecture_comparison.py)  
- **Interactive Tutorial**: [`examples/Example_Attack_Notebook.ipynb`](examples/Example_Attack_Notebook.ipynb)

For comprehensive documentation and guides, visit the [`docs/`](docs/) directory.

## ✨ Key Features

- **🚀 High-Performance Bit Manipulation**: Optimized SEU injection with 10-100x speedup
- **🎯 Flexible Injection Modes**: Systematic exhaustive or stochastic sampling
- **⚡ GPU Acceleration**: Full CUDA support for large models
- **🔍 Layer Targeting**: Precise control over which components to test
- **�️ Production Ready**: 94% test coverage, multi-platform support
- **🔥 PyTorch Native**: Seamless integration with existing workflows

## 🔬 Research Applications

**Space & Aerospace**: Radiation tolerance for spacecraft AI, satellite systems, aviation safety

**Nuclear & Energy**: Robust monitoring systems, power grid AI, industrial automation

**Research**: Architecture benchmarking, fault propagation studies, reliability assessment

## 📈 Performance & Quality

- **⚡ Fast**: <1ms per bitflip operation, memory efficient
- **✅ Tested**: 94% coverage with 109 tests across platforms  
- **🔍 Clean**: Zero critical linting violations, automated quality checks
- **📚 Documented**: Complete API documentation with examples

## 🤝 Community & Support

**Contributing**: See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup and guidelines

**Getting Help**: 
- 📖 Start with [`docs/`](docs/) directory
- 🐛 Use [issue templates](https://github.com/wd7512/seu-injection-framework/issues/new/choose) for bugs
- 💡 Share feature requests through GitHub issues

### **Citation**

If you use this framework in your research, please cite:

```bibtex
@software{seu_injection_framework,
  author = {William Dennis},
  title = {SEU Injection Framework: Fault Tolerance Analysis for Neural Networks},
  year = {2025},
  url = {https://github.com/wd7512/seu-injection-framework},
  version = {1.1.0},
  note = {Production-ready framework for Single Event Upset injection in neural networks}
}
```

**Research Paper:**
```bibtex
@article{dennis2025framework,
  title = {A Framework for Developing Robust Machine Learning Models in Harsh Environments},
  author = {William Dennis},
  year = {2025},
  url = {https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

*Built with ❤️ for the research community studying neural network robustness in harsh environments.* 

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

💡 **Need a full tutorial?** See [`docs/quickstart.md`](docs/quickstart.md) for a complete 10-minute walkthrough.

### 📚 Complete Examples

- **Basic CNN Robustness:** [`examples/basic_cnn_robustness.py`](examples/basic_cnn_robustness.py)
- **Architecture Comparison:** [`examples/architecture_comparison.py`](examples/architecture_comparison.py)
- **Interactive Tutorial:** [`examples/Example_Attack_Notebook.ipynb`](examples/Example_Attack_Notebook.ipynb)

## Features

- **Works with Any PyTorch Model**: Drop-in compatibility with standard PyTorch models - no modifications required
- **Built for Research**: Designed for reliability analysis in space systems, nuclear environments, and harsh conditions
- **High-Performance**: Optimized bit manipulation operations (10-100x faster than naive implementations)
- **Multiple Injection Methods**: Systematic bit-by-bit analysis for small models, stochastic sampling for large-scale campaigns
- **GPU Accelerated**: Full CUDA support for efficient fault injection on neural networks of any size

## 🤝 Contributing & Support

- **Documentation:** [`docs/`](docs/)
- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Issues:** [GitHub Issues](https://github.com/wd7512/seu-injection-framework/issues)
- **Contact:** wwdennis.home@gmail.com

## 📝 Citation

If you use this framework in your research, please cite both the software and the research paper:

```bibtex
@software{seu_injection_framework,
  author = {William Dennis},
  title = {SEU Injection Framework},
  year = {2025},
  url = {https://github.com/wd7512/seu-injection-framework},
  version = {1.1.8}
}

@conference{icaart25,
  author = {William Dennis and James Pope},
  title = {A Framework for Developing Robust Machine Learning Models in Harsh Environments: A Review of CNN Design Choices},
  booktitle = {Proceedings of the 17th International Conference on Agents and Artificial Intelligence - Volume 2: ICAART},
  year = {2025},
  pages = {322-333},
  publisher = {SciTePress},
  organization = {INSTICC},
  doi = {10.5220/0013155000003890},
  isbn = {978-989-758-737-5},
  issn = {2184-433X}
}
```

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for the research community studying neural network robustness in harsh environments.*

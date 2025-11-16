<div align="center">

<h1>SEU Injection Framework</h1>

<a href="https://pypi.org/project/seu-injection-framework/"><img alt="PyPI"
src="https://img.shields.io/pypi/v/seu-injection-framework.svg"/></a>
<a href="https://github.com/wd7512/seu-injection-framework"><img alt="Python Versions"
src="https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue"/></a>
<a href="https://opensource.org/licenses/MIT"><img alt="License: MIT"
src="https://img.shields.io/badge/License-MIT-yellow.svg"/></a>
<br>
<a href="https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars">
<b><br>Research Paper<br></b>
</a>

</div>

<hr>

<div align="center">
  <p>
    A Python framework for <b>Single Event Upset (SEU) injection</b> in neural networks for robustness analysis in harsh environments.
  </p>
  <p>
    <b><a href="docs/">Documentation</a></b> |
    <b><a href="docs/quickstart.md">Quick Start</a></b>
  </p>
</div>

## Installation

**Option 1 (recommended): PyPI**

```bash
# Minimal core dependencies (PyTorch, NumPy, SciPy, tqdm)
pip install seu-injection-framework

# With analysis tools (scikit-learn, pandas, matplotlib, seaborn)
pip install "seu-injection-framework[analysis]"

# Everything (development, notebooks, vision models, docs)
pip install "seu-injection-framework[all]"
```

**Option 2: Source (development)**

```bash
git clone https://github.com/wd7512/seu-injection-framework.git
cd seu-injection-framework

# Using uv (recommended for development)
uv sync --extra dev --extra analysis --extra vision --extra notebooks

# Or using pip
pip install -e ".[all]"
```

**GPU (optional):** install a CUDA-enabled PyTorch first if required:

```bash
# Example for CUDA 12.x (adjust for your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install seu-injection-framework
```

**Verify installation:**

```bash
python -c "from seu_injection import SEUInjector; print('Ready')"
```

Having issues? See [`docs/installation.md`](docs/installation.md).

## Quick example

```python
import torch
from seu_injection.core import ExhaustiveSEUInjector
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

# Sample some results
fault_impacts = [injector.baseline_score - score for score in results['criterion_score']]
print(f"Average accuracy drop: {sum(fault_impacts)/len(fault_impacts):.1%}")
```

Full tutorial: [`docs/quickstart.md`](docs/quickstart.md).

### Examples

- Basic CNN robustness: [`examples/basic_cnn_robustness.py`](examples/basic_cnn_robustness.py)
- Architecture comparison: [`examples/architecture_comparison.py`](examples/architecture_comparison.py)
- Interactive notebook: [`examples/Example_Attack_Notebook.ipynb`](examples/Example_Attack_Notebook.ipynb)

## Features

- Works with standard PyTorch models (no code changes required)
- Suitable for reliability studies in harsh environments (space, nuclear, radiation)
- Optimized bit operations (10–100× faster than naive Python)
- Multiple injection modes: systematic per-bit or stochastic sampling
- Optional CUDA acceleration

## Contributing & support

- Documentation: [`docs/`](docs/)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Issues: [GitHub Issues](https://github.com/wd7512/seu-injection-framework/issues)
- Contact: wwdennis.home@gmail.com

## Citation

If you use this framework in your research, please cite both the software and the research paper:

```bibtex
@software{seu_injection_framework,
  author = {William Dennis},
  title = {SEU Injection Framework},
  year = {2025},
  url = {https://github.com/wd7512/seu-injection-framework},
  version = {1.1.10}
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

______________________________________________________________________

*Built with ❤️ for the research community studying neural network robustness in harsh environments.*

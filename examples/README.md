# Research Examples

Ready-to-run examples demonstrating fault injection methodologies for neural network robustness analysis in harsh environments.

## Quick Start

You can run the examples after installing the framework:

### From PyPI

```bash
pip install seu-injection-framework
python basic_cnn_robustness.py
```

### From Source

```bash
git clone https://github.com/wd7512/seu-injection-framework.git
cd seu-injection-framework
uv sync --all-extras  # or pip install -e ".[dev,notebooks,extras]"
uv run python examples/basic_cnn_robustness.py
```

## Available Examples

- [Example_Attack_Notebook.ipynb](Example_Attack_Notebook.ipynb): Interactive research notebook with comprehensive fault injection analysis and visualization tools.
- [basic_cnn_robustness.py](basic_cnn_robustness.py): Single-architecture vulnerability analysis using systematic bit-flip injection across network layers. Useful for space mission deployment assessment.
- [architecture_comparison.py](architecture_comparison.py): Comparative robustness evaluation using standardized fault injection protocol across multiple architectures. Useful for architecture selection in critical systems.
- [shipsnet/](shipsnet/): Experiments re-creating the methodology from the 2025 paper, [A Framework for Developing Robust Machine Learning Models in Harsh Environments: A Review of CNN Design Choices](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars/). These focus on fault injection and robustness analysis for the ShipsNet dataset. See the README in this folder for details and updates.

## Experimental Protocol

All studies implement IEEE 754-compliant fault injection methodology:

- **Fault Model**: Single Event Upset (SEU) via targeted bit manipulation
- **Coverage**: Exhaustive layer-wise vulnerability mapping
- **Statistics**: Multiple-trial averaging with confidence intervals
- **Validation**: Baseline accuracy verification and repeatability testing

## Customization

You can easily adapt these examples for your own models and data:

```python
from seu_injection.core import ExhaustiveSEUInjector
from seu_injection.metrics import classification_accuracy

# Use your own model and data
injector = ExhaustiveSEUInjector(
  trained_model=your_model,
  criterion=classification_accuracy,
  x=your_test_data,
  y=your_labels
)
results = injector.run_injector(bit_i=0)
```

## Results

Each script generates detailed reports showing:

- Which layers are most vulnerable to bit flips
- How different bit positions affect accuracy
- Robustness comparison across architectures

## Citation

When using these examples in research, please cite:

```bibtex
@software{seu_injection_framework,
  author = {William Dennis},
  title = {SEU Injection Framework: Fault Tolerance Analysis for Neural Networks},
  year = {2025},
  url = {https://github.com/wd7512/seu-injection-framework},
  version = {1.1.10}
}
```

For methodology details and validation results, see the main [repository documentation](../docs/).

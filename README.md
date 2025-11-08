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
from framework.attack import Injector
from framework.criterion import classification_accuracy
import torch

# Load your trained model
model = torch.load('your_model.pth')

# Set up SEU injection
injector = Injector(model, X=test_data, y=test_labels, criterion=classification_accuracy)

# Run SEU analysis
results = injector.run_seu(bit_i=0)  # Test sign bit flips
print(f"Baseline accuracy: {injector.baseline_score}")
print(f"Mean post-SEU accuracy: {results['criterion_score'].mean()}")
```

See `UV_SETUP.md` for detailed installation instructions.

## Features

- **Systematic SEU injection** across neural network parameters
- **Multiple injection strategies**: exhaustive and stochastic sampling  
- **GPU acceleration** support via CUDA
- **Layer-specific targeting** for focused robustness analysis
- **Comprehensive metrics** for fault tolerance evaluation
- **Support for NN, CNN, and RNN architectures**

## Research Applications

This framework enables researchers to:
- Study fault propagation in neural networks under radiation
- Evaluate robustness of different CNN architectural choices
- Develop radiation-hardened models for space applications  
- Benchmark fault tolerance across model types
- Simulate harsh environment conditions for ML deployment

# Code log

### v0.0.6
*date: 13/06/2025*
- added dockerfile to test if we get a performance boost using wsl/linux
  - added a benchamarking.py file in tests
- removed alternative pytorch requirements
- clean up the code with `black`

### v0.0.5
*date: 12.06.2025* 
- merge in changes from Research/ViT branch to enable batches during inference in `framework/criterion.py`
- added a few more print statements in `framework/attack.py`
- added the ability to use dataloaders in the criterion which speed up inference

### v0.0.4
*date: 11.06.2025*
- refactored criterion.py to take inputs as (model, X, y) as this is more intuitive

### v0.0.3
*date: 08.06.2025*
- allows `layer_name__` to be specified in the `.run_seu()` function of the injector
- `.run_stochastic_seu()` function added to injector, aimed out larger models where one only wants to tests bitflips on values with probability `p`
- added module `framework/bitflip.py` so there is no reliance on legacy code

### v0.0.2
*date: 07.06.2025*

Things added 
- `attack()` is removed from framework.attack, we now have `injector`, a class to handle seu injections
- tests added for 3 types of nn, NN, CNN, RNN

### v0.0.1

*date: 05.06.2025*

I have a simple version of the code working in the Example_Attack_Notebook.ipynb. This pulls from some of my legacy code which we will want to remove and the new attack.py module. 

NOTE: This is the MVP and will only work for a NN with binary classification, similar to the example one. Other forms of NN should work, i.e. CNN so long it is binary classification. 

"""
SEU Injection Framework
======================

A comprehensive framework for Single Event Upset (SEU) injection in neural networks
for studying fault tolerance in harsh environments.

This package provides tools for:
- Systematic SEU injection in PyTorch models
- Performance evaluation under radiation-induced bit flips
- Analysis of neural network robustness in space and nuclear environments

Basic Usage:
    >>> from seu_injection import SEUInjector
    >>> injector = SEUInjector(model)
    >>> results = injector.run_stochastic_seu(X, y, p=0.01)

For detailed examples, see the documentation at:
https://seu-injection-framework.readthedocs.io
"""

__version__ = "1.0.0"
__author__ = "William Dennis"
__email__ = "william.dennis@bristol.ac.uk"

# Core public API
from .bitops.float32 import bitflip_float32
from .core.injector import SEUInjector

# Convenience imports for common use cases
from .core.injector import SEUInjector as Injector  # Short alias
from .metrics.accuracy import classification_accuracy, classification_accuracy_loader

__all__ = [
    # Core classes
    "SEUInjector",
    "Injector",  # Short alias

    # Metrics functions
    "classification_accuracy",
    "classification_accuracy_loader",

    # Bitflip operations
    "bitflip_float32",

    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]

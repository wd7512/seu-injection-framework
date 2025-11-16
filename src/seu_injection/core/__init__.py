"""
Core SEU injection functionality.

This module provides the main SEUInjector class for systematic fault injection
in PyTorch neural networks to study robustness in harsh environments.
"""

from .exhaustive_seu_injector import ExhaustiveSEUInjector
from .stochastic_seu_injector import StochasticSEUInjector

__all__ = ["ExhaustiveSEUInjector", "StochasticSEUInjector"]

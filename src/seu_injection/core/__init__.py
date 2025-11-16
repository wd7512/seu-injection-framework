"""
Core SEU injection functionality.

This module provides the main Injector class for systematic fault injection
in PyTorch neural networks to study robustness in harsh environments.
"""

from .injector import Injector

__all__ = ["Injector"]

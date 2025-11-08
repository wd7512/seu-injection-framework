"""
Core SEU injection functionality.

This module provides the main SEUInjector class for systematic fault injection
in PyTorch neural networks to study robustness in harsh environments.
"""

from .injector import SEUInjector

__all__ = ["SEUInjector"]

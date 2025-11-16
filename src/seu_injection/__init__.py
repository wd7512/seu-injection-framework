"""
SEU Injection Framework
======================

A comprehensive framework for Single Event Upset (SEU) injection in neural networks
for studying fault tolerance in harsh environments.

# TODO PRODUCTION READINESS: Public API improvements per PRODUCTION_READINESS_PLAN.md
# ISSUE: Current API requires deep understanding (IEEE 754, layer names, device management)
# NEEDED: High-level convenience functions for common scenarios:
#   - quick_robustness_check(model, test_data) -> simple robustness score
#   - compare_architectures(models_dict, test_data) -> comparative analysis
#   - space_mission_simulation(model, radiation_level) -> space-specific testing
# TYPE SAFETY: Add comprehensive type stubs for better IDE support
# ERROR MESSAGES: Custom exception types with helpful guidance for beginners
# EXAMPLES: In-docstring examples need to be more comprehensive and tested
# COMPATIBILITY: Ensure consistent device handling across all functions
# PRIORITY: MEDIUM - Improve user onboarding experience

This package provides tools for:
- Systematic SEU injection in PyTorch models
- Performance evaluation under radiation-induced bit flips
- Analysis of neural network robustness in space and nuclear environments

Basic Usage:
    >>> from seu_injection.core import ExhaustiveSEUInjector, StochasticSEUInjector
    >>> from seu_injection.metrics import classification_accuracy
    >>> injector = ExhaustiveSEUInjector(trained_model=model, criterion=classification_accuracy, x=X, y=y)
    >>> results = injector.run_injector(bit_i=15)
    >>> injector = StochasticSEUInjector(trained_model=model, criterion=classification_accuracy, x=X, y=y)
    >>> results = injector.run_injector(bit_i=15, p=0.01)

For detailed examples, see the documentation at:
https://github.com/wd7512/seu-injection-framework/blob/main/README.md
"""

try:  # Prefer dynamic version from installed metadata
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("seu-injection-framework")
except Exception:  # Fallback for editable/source checkouts prior to build
    __version__ = "1.1.10"  # Latest stable PyPI release with working build pipeline
__author__ = "William Dennis"
__email__ = "wwdennis.home@gmail.com"

# Core public API
from .bitops.float32 import bitflip_float32
from .core import ExhaustiveSEUInjector, StochasticSEUInjector
from .metrics.accuracy import classification_accuracy, classification_accuracy_loader

# Alias SEUInjector to ExhaustiveSEUInjector for convenience
# DEPRECATION NOTE: SEUInjector will be removed in future releases
SEUInjector = ExhaustiveSEUInjector

__all__ = [
    # Core classes
    "ExhaustiveSEUInjector",
    "StochasticSEUInjector",
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

# TODO DEPRECATION: Remove SEUInjector alias in future release
__all__.extend(
    [
        "SEUInjector",
    ]
)


print("Initialised version:", __version__)

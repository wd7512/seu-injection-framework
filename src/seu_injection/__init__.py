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
- Performance overhead measurement and benchmarking

Basic Usage:
    >>> from seu_injection import SEUInjector
    >>> from seu_injection.metrics import classification_accuracy
    >>> injector = SEUInjector(trained_model=model, criterion=classification_accuracy, x=X, y=y)
    >>> results = injector.run_stochastic_seu(bit_i=15, p=0.01)

For detailed examples, see the documentation at:
https://github.com/wd7512/seu-injection-framework/blob/main/README.md
"""

try:  # Prefer dynamic version from installed metadata
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("seu-injection-framework")
except Exception:  # Fallback for editable/source checkouts prior to build
    __version__ = "1.1.8"  # Latest stable PyPI release with working build pipeline
__author__ = "William Dennis"
__email__ = "wwdennis.home@gmail.com"

# Core public API
from .bitops.float32 import bitflip_float32
from .core.injector import SEUInjector

# Convenience imports for common use cases
from .core.injector import SEUInjector as Injector  # Short alias
from .metrics.accuracy import classification_accuracy, classification_accuracy_loader

# Overhead calculation utilities
from .utils.overhead import (
    benchmark_multiple_networks,
    calculate_overhead,
    format_overhead_report,
    measure_inference_time,
    measure_seu_injection_time,
)

__all__ = [
    # Core classes
    "SEUInjector",
    "Injector",  # Short alias
    # Metrics functions
    "classification_accuracy",
    "classification_accuracy_loader",
    # Bitflip operations
    "bitflip_float32",
    # Overhead calculation utilities
    "measure_inference_time",
    "measure_seu_injection_time",
    "calculate_overhead",
    "benchmark_multiple_networks",
    "format_overhead_report",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]

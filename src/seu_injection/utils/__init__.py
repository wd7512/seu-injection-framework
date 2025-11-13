"""
Utility functions and helpers for the SEU injection framework.

This module provides common utilities for device management, tensor operations,
logging, and other supporting functionality.
"""

# Import device utilities for common use
from .device import detect_device, ensure_tensor

# Import overhead calculation utilities
from .overhead import (
    benchmark_multiple_networks,
    calculate_overhead,
    format_overhead_report,
    measure_inference_time,
    measure_seu_injection_time,
)

__all__ = [
    "detect_device",
    "ensure_tensor",
    "measure_inference_time",
    "measure_seu_injection_time",
    "calculate_overhead",
    "benchmark_multiple_networks",
    "format_overhead_report",
]

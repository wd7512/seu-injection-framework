"""
Utility functions and helpers for the SEU injection framework.

This module provides common utilities for device management, tensor operations,
logging, and other supporting functionality.
"""

# Import device utilities for common use
from .device import detect_device, ensure_tensor

__all__ = [
    "detect_device",
    "ensure_tensor",
]

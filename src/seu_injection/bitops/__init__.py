"""
Bit manipulation operations for SEU injection.

This module provides efficient bit-level operations for different precision formats
used in neural network fault tolerance research.
"""

from .float32 import (
    binary_to_float32,
    bitflip_float32,
    bitflip_float32_fast,
    bitflip_float32_optimized,
    float32_to_binary,
)

__all__ = [
    "bitflip_float32",
    "bitflip_float32_fast",
    "bitflip_float32_optimized",
    "float32_to_binary",
    "binary_to_float32",
]

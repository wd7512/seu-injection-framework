"""
Utility functions for IEEE 754 Float32 bit manipulation.

This module contains shared utility functions used across both legacy and optimized
implementations of float32 bit manipulation.
"""

import struct


def float32_to_binary(f: float) -> str:
    """
    Convert a float32 value to its IEEE 754 binary representation.

    Args:
        f: Float32 value to convert.

    Returns:
        32-character binary string representation.
    """
    [bits] = struct.unpack("!I", struct.pack("!f", f))
    return f"{bits:032b}"


def binary_to_float32(binary_str: str) -> float:
    """
    Convert a 32-bit binary string to a float32 value.

    Args:
        binary_str: 32-character binary string.

    Returns:
        Corresponding float32 value.

    Raises:
        ValueError: If binary_str is not exactly 32 characters.
    """
    if len(binary_str) != 32:
        raise ValueError(
            f"Binary string must be exactly 32 characters, got {len(binary_str)}"
        )
    bits = int(binary_str, 2)
    return float(struct.unpack("!f", struct.pack("!I", bits))[0])

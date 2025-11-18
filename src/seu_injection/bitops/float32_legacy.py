"""Legacy implementations of IEEE 754 Float32 bit manipulation operations.

This module contains educational and fallback implementations of bit-level manipulation functions for float32 values.
These functions prioritize clarity and compatibility over performance and are not intended for production use.
"""

from typing import Union

import numpy as np

from .utils import binary_to_float32, float32_to_binary


def bitflip_float32(x: Union[float, np.ndarray], bit_i: Union[int, None] = None) -> Union[float, np.ndarray]:
    """Flip a specific bit in IEEE 754 float32 values using string-based manipulation.

    Args:
        x (Union[float, np.ndarray]): Input float32 value or numpy array of values.
        bit_i (Optional[int]): Bit position to flip in IEEE 754 representation.

    Returns:
        Union[float, np.ndarray]: Value(s) with the specified bit flipped.

    Raises:
        ValueError: If bit_i is not in valid range [0, 31] when specified.
        TypeError: If input contains non-numeric values.

    """
    if bit_i is None:
        bit_i = np.random.randint(0, 32)
    elif not (0 <= bit_i <= 31):
        raise ValueError(f"Bit position must be between 0 and 31, got {bit_i}")

    if hasattr(x, "__iter__"):
        x_ = np.zeros_like(x, dtype=np.float32)
        for i, item in enumerate(x):
            string = list(float32_to_binary(item))
            string[bit_i] = "0" if string[bit_i] == "1" else "1"
            x_[i] = binary_to_float32("".join(string))
        return x_
    else:
        string = list(float32_to_binary(x))
        string[bit_i] = "0" if string[bit_i] == "1" else "1"
        return binary_to_float32("".join(string))


def _bitflip_original_scalar(x: float, bit_i: int) -> float:
    """Original string-based scalar bitflip for compatibility fallback."""
    string = list(float32_to_binary(x))
    string[bit_i] = "0" if string[bit_i] == "1" else "1"
    return float(binary_to_float32("".join(string)))


def _bitflip_original_array(x, bit_i: int) -> np.ndarray:
    """Original string-based array bitflip for compatibility fallback."""
    x_ = np.zeros_like(x, dtype=np.float32)
    for i, item in enumerate(x):
        string = list(float32_to_binary(item))
        string[bit_i] = "0" if string[bit_i] == "1" else "1"
        x_[i] = binary_to_float32("".join(string))
    return x_

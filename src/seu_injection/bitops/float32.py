"""Optimized IEEE 754 Float32 bit manipulation operations.

This module contains high-performance implementations of bit-level manipulation functions for float32 values. These
functions are designed for production use and prioritize speed and memory efficiency.
"""

from typing import Union

import numpy as np


def bitflip_float32_optimized(
    values: Union[float, np.ndarray], bit_position: int, inplace: bool = False
) -> Union[float, np.ndarray]:
    """High-performance bit flipping using direct memory manipulation and vectorization.

    Args:
        values (Union[float, np.ndarray]): Input float32 value or numpy array.
        bit_position (int): Bit position to flip in IEEE 754 representation.
        inplace (bool): Whether to modify input array directly for memory efficiency.

    Returns:
        Union[float, np.ndarray]: Value(s) with specified bit flipped.

    """
    if not (0 <= bit_position <= 31):
        raise ValueError(f"Bit position must be in range [0, 31], got {bit_position}")

    if np.isscalar(values):
        import struct

        bytes_val = struct.pack("f", values)
        uint32_val = struct.unpack("I", bytes_val)[0]
        actual_bit_pos = 31 - bit_position
        flipped_uint32 = uint32_val ^ (1 << actual_bit_pos)
        bytes_flipped = struct.pack("I", flipped_uint32)
        return float(struct.unpack("f", bytes_flipped)[0])

    values_array = np.asarray(values, dtype=np.float32)
    return _bitflip_array_optimized(values_array, bit_position, inplace)


def _bitflip_array_optimized(values: np.ndarray, bit_position: int, inplace: bool) -> np.ndarray:
    """Internal optimized array bitflip using direct bit manipulation.

    Args:
        values: Input float32 array
        bit_position: Bit position to flip (0-31, where 0 is MSB)
        inplace: Whether to modify input array directly

    Returns:
        Array with specified bits flipped

    """

    work_array = values if inplace else values.copy()
    uint_view = work_array.view(np.uint32)
    mask = np.uint32(1 << (31 - bit_position))
    uint_view ^= mask
    return work_array


def bitflip_float32_fast(
    x: Union[float, np.ndarray], bit_i: Union[int, None] = None, inplace: bool = False
) -> Union[float, np.ndarray]:
    """Intelligent bit flipping with automatic performance optimization and fallback handling.

    Args:
        x (Union[float, np.ndarray]): Input float32 value or numpy array to manipulate.
        bit_i (Optional[int]): Bit position to flip in IEEE 754 representation.
        inplace (bool): If True, modifies the input array in place (only for numpy arrays).

    Returns:
        Union[float, np.ndarray]: Value(s) with specified bit flipped.

    """
    if bit_i is None:
        bit_i = np.random.randint(0, 32)

    if not (0 <= bit_i <= 31):
        raise ValueError(f"Bit position must be in range [0, 31], got {bit_i}")

    if hasattr(x, "__iter__") and not isinstance(x, str):
        try:
            return bitflip_float32_optimized(x, bit_i, inplace=inplace)
        except (ValueError, TypeError):
            from .float32_legacy import _bitflip_original_array

            return _bitflip_original_array(x, bit_i)
    else:
        try:
            return bitflip_float32_optimized(x, bit_i, inplace=inplace)
        except (ValueError, TypeError):
            from .float32_legacy import _bitflip_original_scalar

            # Ensure x is a float for legacy fallback
            if isinstance(x, (int, float)):
                return _bitflip_original_scalar(float(x), bit_i)
            else:
                raise TypeError(f"Expected a scalar float or int for _bitflip_original_scalar, got {type(x)}")

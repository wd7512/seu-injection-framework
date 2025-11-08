"""
Float32 bitflip operations for SEU injection.

This module provides efficient bit manipulation functions for IEEE 754 float32
values, supporting both single values and arrays.
"""

import struct
from typing import Union

import numpy as np


def bitflip_float32(
    x: Union[float, np.ndarray],
    bit_i: int = None
) -> Union[float, np.ndarray]:
    """
    Flip a specific bit in a float32 value or array of values.
    
    This function performs Single Event Upset (SEU) simulation by flipping
    a specific bit in the IEEE 754 float32 representation.
    
    Args:
        x: Input float32 value or numpy array
        bit_i: Bit position to flip (0-31, where 0 is MSB).
               If None, randomly selects a bit position.
               
    Returns:
        Value(s) with the specified bit flipped
        
    Note:
        Bit positions follow IEEE 754 convention:
        - Bit 0: Sign bit (MSB)
        - Bits 1-8: Exponent 
        - Bits 9-31: Mantissa (LSB)
        
    Example:
        >>> bitflip_float32(1.0, 0)  # Flip sign bit
        -1.0
        >>> bitflip_float32([1.0, 2.0], 0)  # Flip sign bit in array
        array([-1., -2.])
    """
    if bit_i is None:
        bit_i = np.random.randint(0, 32)

    if hasattr(x, "__iter__"):
        # Handle arrays/iterables
        x_ = np.zeros_like(x, dtype=np.float32)
        for i, item in enumerate(x):
            string = list(float32_to_binary(item))
            string[bit_i] = "0" if string[bit_i] == "1" else "1"
            x_[i] = binary_to_float32("".join(string))
        return x_
    else:
        # Handle single values
        string = list(float32_to_binary(x))
        string[bit_i] = "0" if string[bit_i] == "1" else "1"
        return binary_to_float32("".join(string))


def float32_to_binary(f: float) -> str:
    """
    Convert a float32 value to its IEEE 754 binary representation.
    
    Args:
        f: Float32 value to convert
        
    Returns:
        32-character binary string representation
        
    Example:
        >>> float32_to_binary(1.0)
        '00111111100000000000000000000000'
    """
    # Pack float into 4 bytes, then unpack as a 32-bit integer
    [bits] = struct.unpack("!I", struct.pack("!f", f))
    # Format the integer as a 32-bit binary string
    return f"{bits:032b}"


def binary_to_float32(binary_str: str) -> float:
    """
    Convert a 32-bit binary string to a float32 value.
    
    Args:
        binary_str: 32-character binary string
        
    Returns:
        Corresponding float32 value
        
    Raises:
        ValueError: If binary_str is not exactly 32 characters
        
    Example:
        >>> binary_to_float32('00111111100000000000000000000000')
        1.0
    """
    if len(binary_str) != 32:
        raise ValueError(f"Binary string must be exactly 32 characters, got {len(binary_str)}")

    # Convert binary string to a 32-bit integer
    bits = int(binary_str, 2)
    # Pack the integer into bytes, then unpack as a float
    return struct.unpack("!f", struct.pack("!I", bits))[0]


# Optimized version for future implementation (Phase 2 performance target)
def bitflip_float32_optimized(
    values: np.ndarray,
    bit_position: int,
    inplace: bool = False
) -> np.ndarray:
    """
    Efficiently flip bits using direct memory manipulation (FUTURE).
    
    This is a placeholder for the optimized O(1) bitflip implementation
    that will replace the current O(32) string-based approach.
    
    Args:
        values: Input float32 array
        bit_position: Bit position to flip (0-31, where 0 is MSB)
        inplace: Whether to modify input array directly
        
    Returns:
        Array with specified bits flipped
        
    Note:
        This function is not yet implemented. Use bitflip_float32() instead.
        Target performance: 32x speedup over string-based operations.
    """
    raise NotImplementedError(
        "Optimized bitflip operations are planned for Phase 2. "
        "Use bitflip_float32() for now."
    )

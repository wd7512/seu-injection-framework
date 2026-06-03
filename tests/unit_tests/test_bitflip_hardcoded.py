"""Hardcoded unit tests for bitflip operations.

This module contains hardcoded test cases with specific input values and expected
outputs to verify that bitflips are being performed in the correct location and
in the correct way. These tests provide strong guarantees about bitflip correctness
by using predetermined values rather than probabilistic or reversibility tests.

Bit Indexing Convention:
This module uses MSB-first indexing where:
- Bit 0 = Sign bit (leftmost bit)
- Bits 1-8 = Exponent bits
- Bits 9-31 = Mantissa bits (fraction)
This matches the string representation used by the bitflip functions.
"""

import numpy as np
import pytest

from seu_injection.bitops import (
    bitflip_float32,
    bitflip_float32_fast,
    bitflip_float32_optimized,
)


def _all_impl(value, bit_pos):
    """Convenience: run all three implementations and return results."""
    return (
        bitflip_float32(value, bit_pos),
        bitflip_float32_optimized(value, bit_pos),
        bitflip_float32_fast(value, bit_pos),
    )


def _assert_all_eq(value, bit_pos, expected, atol=0.0):
    """Assert all three implementations match expected within tolerance."""
    r1, r2, r3 = _all_impl(value, bit_pos)
    assert abs(r1 - expected) <= atol, f"legacy: got {r1}, expected {expected}"
    assert abs(r2 - expected) <= atol, f"optimized: got {r2}, expected {expected}"
    assert abs(r3 - expected) <= atol, f"fast: got {r3}, expected {expected}"


def _assert_all_exact(value, bit_pos, expected):
    """Assert all three implementations exactly equal expected."""
    return _assert_all_eq(value, bit_pos, expected, atol=0.0)


class TestHardcodedBitflips:
    """Test suite with hardcoded inputs and expected outputs for bitflip operations."""

    def test_sign_bit_flip_positive_to_negative(self):
        """Test that flipping bit 0 (sign bit) changes positive to negative."""
        # Using MSB-first indexing: bit 0 is the sign bit (leftmost)
        # Flipping it should change the sign
        for v in [1.0, 2.0, 3.0, 42.0]:
            _assert_all_exact(v, 0, -v)

    def test_sign_bit_flip_negative_to_positive(self):
        """Test that flipping bit 0 (sign bit) changes negative to positive."""
        for v in [1.0, 5.0, 100.0]:
            _assert_all_exact(-v, 0, v)

    def test_matrix_specific_position_flip(self):
        """Test flipping a specific bit at a specific position in a 2D array.

        This is the example from the issue:
        Input: [[0,1],[2,3]]
        Operation: flip bit_i=0 at position [1,1] (value 3)
        Expected: [[0,1],[2,-3]]
        """
        matrix = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        flipped_value = bitflip_float32(matrix[1, 1], 0)
        assert flipped_value == -3.0

        result_matrix = matrix.copy()
        result_matrix[1, 1] = flipped_value
        expected = np.array([[0.0, 1.0], [2.0, -3.0]], dtype=np.float32)
        np.testing.assert_array_equal(result_matrix, expected)

    def test_array_sign_bit_flips_all_elements(self):
        """Test that flipping bit 0 on an entire array changes all signs."""
        input_array = np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float32)
        expected = np.array([-1.0, -2.0, 1.0, 2.0], dtype=np.float32)

        np.testing.assert_array_equal(bitflip_float32(input_array, 0), expected)
        np.testing.assert_array_equal(bitflip_float32_optimized(input_array, 0), expected)
        np.testing.assert_array_equal(bitflip_float32_fast(input_array, 0), expected)

    def test_exponent_bit_flip_known_value(self):
        """Test flipping specific exponent bits with known outcomes."""
        # For value 1.0 in MSB-first indexing:
        # IEEE 754: 0 01111111 00000000000000000000000
        # Flipping bit 1 (MSB of exponent) changes exponent from
        # 01111111 to 11111111, which represents infinity
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            result = impl(1.0, 1)
            assert np.isinf(result)
            assert result > 0  # positive infinity

    def test_mantissa_lsb_flip_known_value(self):
        """Test flipping the least significant bit of mantissa."""
        # For value 1.0, bit 31 is the LSB of the mantissa
        # Flipping it gives exactly 1.0 + 2^-23 (the next representable float)
        expected = np.float32(1.0 + 2**-23)
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            result = impl(1.0, 31)
            assert result == expected, f"{impl.__name__}: got {result}, expected {expected}"

    def test_zero_sign_bit_flip(self):
        """Test flipping the sign bit of zero."""
        # Positive zero (0.0) — flipping bit 0 should give negative zero
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            result = impl(0.0, 0)
            assert result == 0.0
            assert np.signbit(result)  # should be negative zero

        # Flipping back should give positive zero
        result2 = bitflip_float32(bitflip_float32(0.0, 0), 0)
        assert result2 == 0.0
        assert not np.signbit(result2)

    def test_specific_small_value_flip(self):
        """Test flipping bits on a small specific value."""
        # 0.5 — flipping sign bit gives -0.5 exactly
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            assert impl(0.5, 0) == -0.5

            # Flipping bit 31 (LSB) gives 0.5 + 2^-24 exactly
            # 0.5 = 1.0 * 2^-1, mantissa LSB = 2^-1 * 2^-23 = 2^-24
            expected = np.float32(0.5 + 2**-24)
            assert impl(0.5, 31) == expected

    def test_large_value_sign_flip(self):
        """Test sign bit flip on large values."""
        _assert_all_exact(1000000.0, 0, -1000000.0)

        # 123456.78 is not exactly representable in float32
        # Use the proper float32 representation
        v = np.float32(123456.78)
        r1, r2, r3 = _all_impl(v, 0)
        assert r1 == -v
        assert r2 == -v
        assert r3 == -v

    def test_multiple_known_values_same_bit(self):
        """Test the same bit flip on multiple known values."""
        # Use float32 values for exact comparison
        test_cases = [
            (np.float32(1.0), np.float32(-1.0)),
            (np.float32(2.0), np.float32(-2.0)),
            (np.float32(-5.0), np.float32(5.0)),
            (np.float32(-10.5), np.float32(10.5)),
            (np.float32(0.25), np.float32(-0.25)),
        ]

        for input_val, expected in test_cases:
            r1, r2, r3 = _all_impl(input_val, 0)
            assert r1 == expected, f"legacy: {r1} != {expected}"
            assert r2 == expected, f"optimized: {r2} != {expected}"
            assert r3 == expected, f"fast: {r3} != {expected}"

    def test_2d_array_partial_flip(self):
        """Test flipping specific elements in a 2D array."""
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)

        result_matrix = matrix.copy()
        result_matrix[1, 1] = bitflip_float32(matrix[1, 1], 0)

        expected = np.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
        np.testing.assert_array_equal(result_matrix, expected)

    def test_sequence_of_specific_flips(self):
        """Test a sequence of specific bit flips on the same value."""
        value = 10.0

        # First flip bit 0 (sign bit)
        step1 = bitflip_float32(value, 0)
        assert step1 == -10.0

        # Then flip bit 0 again (should restore)
        step2 = bitflip_float32(step1, 0)
        assert step2 == 10.0

        # Now flip bit 31 (LSB of mantissa) — 10.0 + 2^-20
        # 10.0 = 1.25 * 2^3, so LSB = 2^3 * 2^-23 = 2^-20
        expected = np.float32(10.0 + 2**-20)
        step3 = bitflip_float32(step2, 31)
        assert step3 == expected, f"got {step3}, expected {expected}"

    def test_all_implementations_consistent_hardcoded(self):
        """Verify all three implementations give identical results for hardcoded cases."""
        test_cases = [
            (1.0, 0, -1.0),
            (2.0, 0, -2.0),
            (-3.0, 0, 3.0),
            (0.5, 0, -0.5),
            (100.0, 0, -100.0),
        ]

        for input_val, bit_pos, expected in test_cases:
            r1 = bitflip_float32(input_val, bit_pos)
            r2 = bitflip_float32_optimized(input_val, bit_pos)
            r3 = bitflip_float32_fast(input_val, bit_pos)

            # All should match the expected value exactly
            assert r1 == expected, f"legacy: {r1}"
            assert r2 == expected, f"optimized: {r2}"
            assert r3 == expected, f"fast: {r3}"

            # All should match each other
            assert r1 == r2
            assert r1 == r3

    def test_vector_hardcoded_values(self):
        """Test bitflip on a vector with hardcoded expected results."""
        input_vec = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        expected = np.array([-10.0, -20.0, -30.0, -40.0, -50.0], dtype=np.float32)

        np.testing.assert_array_equal(bitflip_float32(input_vec, 0), expected)

    def test_mixed_signs_array_hardcoded(self):
        """Test bitflip on array with mixed positive and negative values."""
        input_array = np.array([5.0, -10.0, 15.0, -20.0], dtype=np.float32)
        expected = np.array([-5.0, 10.0, -15.0, 20.0], dtype=np.float32)

        np.testing.assert_array_equal(bitflip_float32(input_array, 0), expected)

    def test_single_element_array_hardcoded(self):
        """Test bitflip on a single-element array."""
        input_array = np.array([7.0], dtype=np.float32)
        expected = np.array([-7.0], dtype=np.float32)

        np.testing.assert_array_equal(bitflip_float32(input_array, 0), expected)

    def test_power_of_two_values(self):
        """Test bitflip on powers of two (simple binary representations)."""
        for power in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
            _assert_all_exact(power, 0, -power)

    def test_fractional_powers_of_two(self):
        """Test bitflip on fractional powers of two."""
        for frac in [0.5, 0.25, 0.125, 0.0625]:
            _assert_all_exact(frac, 0, -frac)

    def test_specific_bit_positions_on_one(self):
        """Test various bit positions on value 1.0 with known expected behaviors."""
        # For 1.0 in MSB-first indexing:
        # Bit layout: 0 01111111 00000000000000000000000

        # Bit 0 (sign bit): 1.0 -> -1.0
        assert bitflip_float32(1.0, 0) == -1.0

        # Bit 1 (exponent MSB): flipping creates all 1's -> infinity
        result = bitflip_float32(1.0, 1)
        assert np.isinf(result)
        assert result > 0

        # Bit 9 (first bit of mantissa): sets implicit 1.0 + 0.5 in mantissa
        # Result should be exactly 1.5
        assert bitflip_float32(1.0, 9) == np.float32(1.5)

        # Bit 31 (mantissa LSB): gives 1.0 + 2^-23 exactly
        assert bitflip_float32(1.0, 31) == np.float32(1.0 + 2**-23)

    # --- Edge case tests added from swarm review ---

    def test_nan_input_sign_bit(self):
        """Test flipping sign bit on NaN preserves NaN."""
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            result = impl(np.float32(np.nan), 0)
            assert np.isnan(result), f"{impl.__name__} did not preserve NaN"
            # Sign bit may change, but value stays NaN

    def test_nan_input_exponent_bit(self):
        """Test flipping exponent bits on NaN produces a normal value.

        NaN has exponent = all 1s (0xFF). Flipping any exponent bit
        changes the exponent to a valid value, producing a normal float.
        """
        # For NaN, the exponent bits are all 1.
        # Flipping bit 1 (MSB of exponent) gives exponent 0x7F = 127
        # This produces a normal float with the original NaN mantissa
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            result = impl(np.float32(np.nan), 1)
            # NaN -> normal float (finite, not NaN)
            assert not np.isnan(result), f"{impl.__name__} should produce normal"
            assert np.isfinite(result)

    def test_infinity_sign_flip(self):
        """Test flipping sign bit on infinity."""
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            # +inf -> -inf
            pos_result = impl(np.float32(np.inf), 0)
            assert np.isinf(pos_result)
            assert pos_result < 0

            # -inf -> +inf
            neg_result = impl(np.float32(-np.inf), 0)
            assert np.isinf(neg_result)
            assert neg_result > 0

    def test_infinity_exponent_bit_flip(self):
        """Test flipping exponent bit on infinity produces a normal value.

        Infinity has exponent = all 1s (0xFF), mantissa = 0.
        Flipping bit 1 (MSB of exponent) changes exponent to 0x7F = 127.
        Result: 1.0 * 2^(127-127) = 1.0
        """
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            result = impl(np.float32(np.inf), 1)
            assert result == np.float32(1.0), f"{impl.__name__}: got {result}"
            assert np.isfinite(result)
            assert not np.isnan(result)

    def test_denormal_input(self):
        """Test bitflip on a denormalized (subnormal) float32 value.

        The smallest positive normal float32 is ~1.175e-38.
        Values below this are denormalized (exponent = 0, implicit bit = 0).
        """
        # Smallest subnormal positive float32 = 2^-149
        denorm = np.float32(2**-149)
        assert denorm > 0
        assert denorm < np.finfo(np.float32).tiny  # smaller than smallest normal

        # Flipping sign bit should give negative of same denorm
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            result = impl(denorm, 0)
            assert np.signbit(result)
            assert abs(result) == denorm

    def test_denormal_mantissa_flip(self):
        """Test flipping the LSB on the smallest subnormal produces zero.

        The smallest positive subnormal float32 is 2^-149 with
        bit pattern: sign=0, exponent=0, mantissa=0x000001.
        Flipping bit 31 (the only set mantissa bit) gives mantissa=0,
        producing exactly +0.0.
        """
        denorm = np.float32(2**-149)

        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            result = impl(denorm, 31)
            assert result == np.float32(0.0), f"{impl.__name__}: got {result}"
            assert not np.signbit(result)  # positive zero

    def test_bit_index_boundary_zero(self):
        """Test bit position 0 (valid — sign bit)."""
        _assert_all_exact(1.0, 0, -1.0)

    def test_bit_index_boundary_31(self):
        """Test bit position 31 (valid — LSB of mantissa)."""
        expected = np.float32(1.0 + 2**-23)
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            result = impl(1.0, 31)
            assert result == expected

    def test_negative_zero_non_sign_bit_flip(self):
        """Test flipping a non-sign bit on negative zero.

        Negative zero has bit pattern 0x80000000 (sign=1, everything else=0).
        Flipping a mantissa or exponent bit on it should produce a non-zero value
        with sign preserved (since sign bit 0 is untouched).
        """
        neg_zero = np.float32(-0.0)

        # Flip bit 31 (mantissa LSB) on -0.0
        # This sets the LSB, producing a small negative denormal (sign preserved)
        for impl in [bitflip_float32, bitflip_float32_optimized, bitflip_float32_fast]:
            result = impl(neg_zero, 31)
            assert result != 0.0
            # Sign bit is unchanged (still 1) — result is negative
            assert np.signbit(result)

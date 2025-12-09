"""Hardcoded unit tests for bitflip operations.

This module contains hardcoded test cases with specific input values and expected
outputs to verify that bitflips are being performed in the correct location and
in the correct way. These tests provide strong guarantees about bitflip correctness
by using predetermined values rather than probabilistic or reversibility tests.
"""

import numpy as np
import pytest

from seu_injection.bitops import bitflip_float32, bitflip_float32_fast, bitflip_float32_optimized


class TestHardcodedBitflips:
    """Test suite with hardcoded inputs and expected outputs for bitflip operations."""

    def test_sign_bit_flip_positive_to_negative(self):
        """Test that flipping bit 0 (sign bit) changes positive to negative."""
        # Bit 0 is the sign bit in IEEE 754 float32
        # Flipping it should change the sign
        
        # Test with value 1.0 -> -1.0
        assert bitflip_float32(1.0, 0) == -1.0
        assert bitflip_float32_optimized(1.0, 0) == -1.0
        assert bitflip_float32_fast(1.0, 0) == -1.0
        
        # Test with value 2.0 -> -2.0
        assert bitflip_float32(2.0, 0) == -2.0
        assert bitflip_float32_optimized(2.0, 0) == -2.0
        assert bitflip_float32_fast(2.0, 0) == -2.0
        
        # Test with value 3.0 -> -3.0
        assert bitflip_float32(3.0, 0) == -3.0
        assert bitflip_float32_optimized(3.0, 0) == -3.0
        assert bitflip_float32_fast(3.0, 0) == -3.0
        
        # Test with value 42.0 -> -42.0
        assert bitflip_float32(42.0, 0) == -42.0
        assert bitflip_float32_optimized(42.0, 0) == -42.0
        assert bitflip_float32_fast(42.0, 0) == -42.0

    def test_sign_bit_flip_negative_to_positive(self):
        """Test that flipping bit 0 (sign bit) changes negative to positive."""
        # Test with value -1.0 -> 1.0
        assert bitflip_float32(-1.0, 0) == 1.0
        assert bitflip_float32_optimized(-1.0, 0) == 1.0
        assert bitflip_float32_fast(-1.0, 0) == 1.0
        
        # Test with value -5.0 -> 5.0
        assert bitflip_float32(-5.0, 0) == 5.0
        assert bitflip_float32_optimized(-5.0, 0) == 5.0
        assert bitflip_float32_fast(-5.0, 0) == 5.0
        
        # Test with value -100.0 -> 100.0
        assert bitflip_float32(-100.0, 0) == 100.0
        assert bitflip_float32_optimized(-100.0, 0) == 100.0
        assert bitflip_float32_fast(-100.0, 0) == 100.0

    def test_matrix_specific_position_flip(self):
        """Test flipping a specific bit at a specific position in a 2D array.
        
        This is the example from the issue:
        Input: [[0,1],[2,3]]
        Operation: flip bit_i=0 at position [1,1] (value 3)
        Expected: [[0,1],[2,-3]]
        """
        # Create the input matrix
        matrix = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        
        # Extract the value at [1,1]
        value_at_1_1 = matrix[1, 1]  # This is 3.0
        
        # Flip bit 0 (sign bit) on this value
        flipped_value = bitflip_float32(value_at_1_1, 0)
        
        # Verify it's -3.0
        assert flipped_value == -3.0
        
        # Create expected output matrix
        expected = np.array([[0.0, 1.0], [2.0, -3.0]], dtype=np.float32)
        
        # Apply the flip to the matrix
        result_matrix = matrix.copy()
        result_matrix[1, 1] = flipped_value
        
        # Verify the result matches expected
        np.testing.assert_array_equal(result_matrix, expected)

    def test_array_sign_bit_flips_all_elements(self):
        """Test that flipping bit 0 on an entire array changes all signs."""
        # Input array
        input_array = np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float32)
        
        # Expected output (all signs flipped)
        expected = np.array([-1.0, -2.0, 1.0, 2.0], dtype=np.float32)
        
        # Test with each implementation
        result_legacy = bitflip_float32(input_array, 0)
        result_optimized = bitflip_float32_optimized(input_array, 0)
        result_fast = bitflip_float32_fast(input_array, 0)
        
        np.testing.assert_array_equal(result_legacy, expected)
        np.testing.assert_array_equal(result_optimized, expected)
        np.testing.assert_array_equal(result_fast, expected)

    def test_exponent_bit_flip_known_value(self):
        """Test flipping specific exponent bits with known outcomes."""
        # For value 1.0:
        # IEEE 754: 0 01111111 00000000000000000000000
        #           ^ sign
        #             ^^^^^^^^ exponent (127 = bias)
        #                     ^^^^^^^^^^^^^^^^^^^^^^^ mantissa
        
        # Flipping bit 1 (MSB of exponent) changes exponent to 11111111
        # which represents infinity in IEEE 754
        value = 1.0
        result = bitflip_float32(value, 1)
        
        # The result should be positive infinity
        assert np.isinf(result)
        assert result > 0  # positive infinity
        assert result != value

    def test_mantissa_lsb_flip_known_value(self):
        """Test flipping the least significant bit of mantissa."""
        # For value 1.0:
        # IEEE 754: 0 01111111 00000000000000000000000
        # Bit 31 is the LSB of the mantissa
        
        value = 1.0
        result = bitflip_float32(value, 31)
        
        # Flipping the LSB should give us the next representable float
        # For 1.0, this should be approximately 1.0 + 2^-23
        expected_diff = 2**-23  # smallest change for value near 1.0
        
        # The result should be very close to but not equal to 1.0
        assert result != value
        assert abs(result - value) < 2 * expected_diff

    def test_zero_sign_bit_flip(self):
        """Test flipping the sign bit of zero."""
        # Positive zero (0.0) has all bits 0
        # Flipping bit 0 should give negative zero (-0.0)
        result = bitflip_float32(0.0, 0)
        
        # In Python, -0.0 == 0.0, but we can distinguish them
        assert result == 0.0  # Equal value
        assert np.signbit(result) == True  # But different sign bit
        
        # Flipping back should give positive zero
        result2 = bitflip_float32(result, 0)
        assert result2 == 0.0
        assert np.signbit(result2) == False

    def test_specific_small_value_flip(self):
        """Test flipping bits on a small specific value."""
        # Test with 0.5
        # IEEE 754: 0 01111110 00000000000000000000000
        
        # Flipping bit 0 should give -0.5
        result = bitflip_float32(0.5, 0)
        assert result == -0.5
        
        # Flipping bit 31 (LSB) should give a slightly different value
        result = bitflip_float32(0.5, 31)
        assert result != 0.5
        assert abs(result - 0.5) < 0.0001

    def test_large_value_sign_flip(self):
        """Test sign bit flip on large values."""
        large_val = 1000000.0
        result = bitflip_float32(large_val, 0)
        assert result == -1000000.0
        
        large_val = 123456.78
        result = bitflip_float32(large_val, 0)
        assert abs(result + 123456.78) < 0.01

    def test_multiple_known_values_same_bit(self):
        """Test the same bit flip on multiple known values."""
        # Test bit 0 (sign bit) on multiple values
        test_cases = [
            (1.0, -1.0),
            (2.0, -2.0),
            (3.14159, -3.14159),
            (-5.0, 5.0),
            (-10.5, 10.5),
            (0.25, -0.25),
        ]
        
        for input_val, expected in test_cases:
            result = bitflip_float32(input_val, 0)
            if expected >= 0:
                assert abs(result - expected) < 1e-5, f"Failed for {input_val}: got {result}, expected {expected}"
            else:
                assert abs(result - expected) < 1e-5, f"Failed for {input_val}: got {result}, expected {expected}"

    def test_2d_array_partial_flip(self):
        """Test flipping specific elements in a 2D array."""
        # Create a 3x3 matrix
        matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float32)
        
        # Flip bit 0 on element at [1, 1] (value 5.0)
        result_matrix = matrix.copy()
        result_matrix[1, 1] = bitflip_float32(matrix[1, 1], 0)
        
        # Expected: 5.0 -> -5.0
        expected = np.array([
            [1.0, 2.0, 3.0],
            [4.0, -5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float32)
        
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
        
        # Now flip bit 31 (LSB of mantissa)
        step3 = bitflip_float32(step2, 31)
        assert step3 != 10.0  # Should be slightly different
        assert abs(step3 - 10.0) < 0.01

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
            result_legacy = bitflip_float32(input_val, bit_pos)
            result_optimized = bitflip_float32_optimized(input_val, bit_pos)
            result_fast = bitflip_float32_fast(input_val, bit_pos)
            
            # All should match the expected value
            assert abs(result_legacy - expected) < 1e-6
            assert abs(result_optimized - expected) < 1e-6
            assert abs(result_fast - expected) < 1e-6
            
            # All should match each other
            assert abs(result_legacy - result_optimized) < 1e-6
            assert abs(result_legacy - result_fast) < 1e-6

    def test_vector_hardcoded_values(self):
        """Test bitflip on a vector with hardcoded expected results."""
        # Input vector
        input_vec = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        
        # Flip bit 0 (sign bit)
        result = bitflip_float32(input_vec, 0)
        
        # Expected: all signs flipped
        expected = np.array([-10.0, -20.0, -30.0, -40.0, -50.0], dtype=np.float32)
        
        np.testing.assert_array_equal(result, expected)

    def test_mixed_signs_array_hardcoded(self):
        """Test bitflip on array with mixed positive and negative values."""
        # Input with mixed signs
        input_array = np.array([5.0, -10.0, 15.0, -20.0], dtype=np.float32)
        
        # Flip bit 0 (sign bit)
        result = bitflip_float32(input_array, 0)
        
        # Expected: all signs flipped
        expected = np.array([-5.0, 10.0, -15.0, 20.0], dtype=np.float32)
        
        np.testing.assert_array_equal(result, expected)

    def test_single_element_array_hardcoded(self):
        """Test bitflip on a single-element array."""
        # Single element array
        input_array = np.array([7.0], dtype=np.float32)
        
        # Flip bit 0
        result = bitflip_float32(input_array, 0)
        
        # Expected
        expected = np.array([-7.0], dtype=np.float32)
        
        np.testing.assert_array_equal(result, expected)

    def test_power_of_two_values(self):
        """Test bitflip on powers of two (simple binary representations)."""
        # Powers of 2 have simple IEEE 754 representations
        powers = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        
        for power in powers:
            # Flip sign bit
            result = bitflip_float32(power, 0)
            assert result == -power
            
            # Verify with optimized version
            result_opt = bitflip_float32_optimized(power, 0)
            assert result_opt == -power

    def test_fractional_powers_of_two(self):
        """Test bitflip on fractional powers of two."""
        fractions = [0.5, 0.25, 0.125, 0.0625]
        
        for frac in fractions:
            # Flip sign bit
            result = bitflip_float32(frac, 0)
            assert result == -frac
            
            # Verify with fast version
            result_fast = bitflip_float32_fast(frac, 0)
            assert result_fast == -frac

    def test_specific_bit_positions_on_one(self):
        """Test various bit positions on value 1.0 with known expected behaviors."""
        # For 1.0: 0 01111111 00000000000000000000000
        
        # Bit 0 (sign): 1.0 -> -1.0
        result = bitflip_float32(1.0, 0)
        assert result == -1.0
        
        # Bit 1 (exponent MSB): flipping creates all 1's in exponent = infinity
        result = bitflip_float32(1.0, 1)
        assert np.isinf(result)
        assert result > 0
        
        # Bit 9 (mantissa MSB): should give a value > 1.0
        result = bitflip_float32(1.0, 9)
        assert result > 1.0
        assert result < 2.0
        
        # Bit 31 (mantissa LSB): should give value very close to 1.0
        result = bitflip_float32(1.0, 31)
        assert result != 1.0
        assert abs(result - 1.0) < 0.0001

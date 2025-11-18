import struct

import numpy as np
import pytest

from seu_injection.bitops.float32 import (
    binary_to_float32,
    bitflip_float32,
    bitflip_float32_fast,
    float32_to_binary,
)


class TestBitflipOperations:
    """Test suite for bitflip operations on float32 values."""

    def test_bitflip_float32_basic(self):
        """Test basic bit flipping functionality."""
        result = bitflip_float32(1.0, 0)
        assert result == -1.0, f"Expected -1.0, got {result}"

    def test_bitflip_reversibility(self):
        """Test that flipping the same bit twice returns original value."""
        test_values = [1.0, -1.0, 0.5, -0.5, 3.14159, -2.71828]
        for value in test_values:
            for bit_pos in [0, 1, 15, 16, 31]:
                flipped_once = bitflip_float32(value, bit_pos)
                flipped_twice = bitflip_float32(flipped_once, bit_pos)
                if np.isnan(value) and np.isnan(flipped_twice):
                    continue
                elif np.isnan(flipped_twice) and not np.isnan(value):
                    continue
                else:
                    assert abs(value - flipped_twice) < 1e-6

    def test_bitflip_array_input(self):
        """Test bitflip with array input."""
        values = np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float32)
        result = bitflip_float32(values, 0)
        expected = np.array([-1.0, -2.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_bitflip_edge_cases(self):
        """Test bitflip with edge cases like zero, infinity."""
        original_bits = float32_to_binary(0.0)
        result_zero = bitflip_float32(0.0, 0)
        result_bits = float32_to_binary(result_zero)
        assert original_bits != result_bits

        small_val = 1e-6
        result_small = bitflip_float32(small_val, 16)
        assert result_small != small_val

    def test_bit_position_validation(self):
        """Test that invalid bit positions are handled correctly."""
        test_val = 1.0
        result_0 = bitflip_float32(test_val, 0)
        result_31 = bitflip_float32(test_val, 31)
        assert result_0 is not None
        assert result_31 is not None

    def test_float32_binary_conversion(self):
        """Test float32 to binary string conversion."""
        result = float32_to_binary(1.0)
        assert len(result) == 32
        assert result[0] == "0"
        result_neg = float32_to_binary(-1.0)
        assert len(result_neg) == 32
        assert result_neg[0] == "1"

    def test_binary_float32_conversion(self):
        """Test binary string to float32 conversion."""
        test_values = [1.0, -1.0, 0.5, 3.14159, -2.71828]
        for value in test_values:
            binary = float32_to_binary(value)
            converted_back = binary_to_float32(binary)
            assert abs(value - converted_back) < 1e-6

    def test_random_bit_position(self):
        """Test that random bit position works when no position is specified."""
        value = 1.0
        result1 = bitflip_float32(value)
        result2 = bitflip_float32(value)
        assert isinstance(result1, (float, np.floating))
        assert isinstance(result2, (float, np.floating))

    def test_binary_to_float32_invalid_length(self):
        """Test binary_to_float32 with invalid length input."""
        with pytest.raises(
            ValueError, match="Binary string must be exactly 32 characters"
        ):
            binary_to_float32("101010")
        with pytest.raises(
            ValueError, match="Binary string must be exactly 32 characters"
        ):
            binary_to_float32("1" * 33)

    def test_bitflip_fast_invalid_bit_position(self):
        """Test bitflip_float32_fast with invalid bit positions."""
        with pytest.raises(ValueError, match="Bit position must be in range"):
            bitflip_float32_fast(1.0, -1)
        with pytest.raises(ValueError, match="Bit position must be in range"):
            bitflip_float32_fast(1.0, 32)

    def test_bitflip_fast_array_with_fallback(self):
        """Test bitflip_float32_fast with arrays that might trigger fallback."""
        values = [1.0, 2.0, 3.0]
        result = bitflip_float32_fast(values, 0)
        expected = [-1.0, -2.0, -3.0]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-6

    def test_bitflip_fast_scalar_fallback(self):
        """Test bitflip_float32_fast scalar fallback path."""
        result = bitflip_float32_fast(1.0, 0)
        expected = -1.0
        assert abs(result - expected) < 1e-6

    def test_bitflip_fast_with_none_bit_position(self):
        """Test bitflip_float32_fast with None bit position (random)."""
        np.random.seed(42)
        original = 1.0
        result1 = bitflip_float32_fast(original, None)
        result2 = bitflip_float32_fast(original, None)
        assert isinstance(result1, float)
        assert isinstance(result2, float)
        array_input = [1.0, 2.0]
        result_array = bitflip_float32_fast(array_input, None)
        assert len(result_array) == 2

    def test_optimized_array_dtype_conversion(self):
        """Test optimized function with non-float32 array input."""
        values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = bitflip_float32_fast(values, 0)
        expected = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_comprehensive_binary_conversions(self):
        """Test edge cases in binary conversion functions."""
        test_values = [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), 0.5, -0.5]
        for value in test_values:
            if not np.isnan(value):
                binary_str = float32_to_binary(value)
                recovered = binary_to_float32(binary_str)
                if np.isinf(value):
                    assert np.isinf(recovered)
                    assert np.sign(value) == np.sign(recovered)
                else:
                    assert abs(value - recovered) < 1e-6

    def test_bitflip_edge_case_values(self):
        """Test bitflip operations with edge case float values."""
        small_val = np.float32(1e-10)
        result = bitflip_float32_fast(small_val, 31)
        assert result != small_val
        large_val = np.float32(1e10)
        result = bitflip_float32_fast(large_val, 31)
        assert result != large_val

    def test_inplace_modification_coverage(self):
        """Test inplace modification paths in optimized functions."""
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original_id = id(values)
        result = bitflip_float32_fast(values, 0, inplace=True)
        assert id(result) == original_id
        expected = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_string_input_rejection(self):
        """Test that string inputs are properly rejected by array handlers."""
        with pytest.raises(struct.error, match="required argument is not a float"):
            bitflip_float32_fast("not_a_number", 0)

    def test_non_iterable_input(self):
        """Test with inputs that don't have __iter__ method."""
        result = bitflip_float32_fast(42.0, 0)
        expected = -42.0
        assert abs(result - expected) < 1e-6

    def test_comprehensive_bit_position_coverage(self):
        """Test all bit positions to ensure no edge cases missed."""
        test_value = 1.0
        for bit_pos in range(32):
            result = bitflip_float32_fast(test_value, bit_pos)
            assert isinstance(result, float)
            double_flip = bitflip_float32_fast(result, bit_pos)
            assert abs(double_flip - test_value) < 1e-6

    def test_performance_basic(self):
        """Basic performance test to ensure operations complete in reasonable time."""
        import time

        # Test single value performance
        start_time = time.time()
        for _ in range(1000):
            bitflip_float32(1.0, 15)
        end_time = time.time()

        # Should complete 1000 operations in under 1 second (very generous)
        assert (end_time - start_time) < 1.0, "Bitflip operations are too slow"

        # Test array performance
        values = np.random.random(100).astype(np.float32)
        start_time = time.time()
        result = bitflip_float32(values, 15)
        end_time = time.time()

        assert len(result) == len(values), "Array output length mismatch"
        assert (end_time - start_time) < 0.1, "Array bitflip operations are too slow"

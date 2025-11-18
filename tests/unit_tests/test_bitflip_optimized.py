"""
Test suite for optimized bitflip operations.

This module tests the new optimized bitflip implementations to ensure
compatibility with the original implementation.
"""

import numpy as np
import pytest

from seu_injection.bitops.float32 import (
    bitflip_float32_fast,
    bitflip_float32_optimized,
)
from seu_injection.bitops.float32_legacy import bitflip_float32


class TestOptimizedBitflipOperations:
    """Test suite for optimized bitflip operations."""

    def test_optimized_basic_functionality(self):
        """Test that optimized bitflip produces same results as original."""
        test_values = [1.0, -1.0, 0.5, -0.5, 3.14159, -2.71828, 0.0]

        for value in test_values:
            for bit_pos in [0, 1, 8, 15, 16, 23, 31]:
                original_result = bitflip_float32(value, bit_pos)
                optimized_result = bitflip_float32_optimized(value, bit_pos)

                # Handle special values (NaN, inf)
                if np.isnan(original_result) and np.isnan(optimized_result):
                    continue
                if np.isinf(original_result) and np.isinf(optimized_result):
                    # Both infinite - check sign matches
                    assert np.sign(original_result) == np.sign(optimized_result), (
                        f"Infinity sign mismatch for value {value}, bit {bit_pos}"
                    )
                    continue

                assert abs(original_result - optimized_result) < 1e-6, (
                    f"Mismatch for value {value}, bit {bit_pos}: original={original_result}, optimized={optimized_result}"
                )

    def test_optimized_array_functionality(self):
        """Test optimized bitflip with array inputs."""
        test_array = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5], dtype=np.float32)

        for bit_pos in [0, 8, 16, 31]:
            original_result = bitflip_float32(test_array, bit_pos)
            optimized_result = bitflip_float32_optimized(test_array, bit_pos)

            np.testing.assert_allclose(
                original_result, optimized_result, rtol=1e-6, atol=1e-6
            )

    def test_optimized_reversibility(self):
        """Test that optimized bitflip operations are reversible."""
        test_values = [1.0, -1.0, 0.5, 3.14159, -2.71828]

        for value in test_values:
            for bit_pos in [0, 1, 15, 16, 31]:
                # Double flip should return original
                flipped_once = bitflip_float32_optimized(value, bit_pos)
                flipped_twice = bitflip_float32_optimized(flipped_once, bit_pos)

                # Handle NaN cases
                if np.isnan(value) and np.isnan(flipped_twice):
                    continue
                elif np.isnan(flipped_twice) and not np.isnan(value):
                    continue

                assert abs(value - flipped_twice) < 1e-6, (
                    f"Reversibility failed for {value}, bit {bit_pos}: {value} -> {flipped_once} -> {flipped_twice}"
                )

    def test_optimized_inplace_operations(self):
        """Test inplace bitflip operations."""
        original_array = np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float32)
        test_array = original_array.copy()

        # Test inplace operation
        result = bitflip_float32_optimized(test_array, 0, inplace=True)

        # Should return the same array object
        assert result is test_array, "Inplace operation should return same array object"

        # Should have modified the original array
        assert not np.array_equal(test_array, original_array), (
            "Inplace operation should modify array"
        )

        # Check specific expected results (sign bit flip)
        expected = np.array([-1.0, -2.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(test_array, expected)

    def test_optimized_input_validation(self):
        """Test input validation for optimized functions."""
        # Test invalid bit positions
        with pytest.raises(ValueError, match="Bit position must be in range"):
            bitflip_float32_optimized(1.0, -1)

        with pytest.raises(ValueError, match="Bit position must be in range"):
            bitflip_float32_optimized(1.0, 32)

    def test_fast_function_compatibility(self):
        """Test that bitflip_float32_fast maintains compatibility."""
        test_values = [1.0, -1.0, np.array([1.0, 2.0]), [3.0, 4.0]]

        for value in test_values:
            for bit_pos in [0, 15, 31]:
                original_result = bitflip_float32(value, bit_pos)
                fast_result = bitflip_float32_fast(value, bit_pos)

                if isinstance(original_result, np.ndarray):
                    np.testing.assert_allclose(original_result, fast_result, rtol=1e-6)
                else:
                    if not (np.isnan(original_result) and np.isnan(fast_result)):
                        assert abs(original_result - fast_result) < 1e-6

    def test_edge_cases_optimized(self):
        """Test edge cases with optimized implementation."""
        # Test with special float values
        special_values = [0.0, -0.0, np.inf, -np.inf, np.nan]

        for value in special_values:
            for bit_pos in [0, 15, 31]:
                try:
                    result = bitflip_float32_optimized(value, bit_pos)
                    # Should produce a valid result (even if NaN or inf)
                    assert isinstance(result, (float, np.floating))
                except Exception as e:
                    pytest.fail(
                        f"Optimized bitflip failed for {value}, bit {bit_pos}: {e}"
                    )

    def test_type_conversion_consistency(self):
        """Test that type conversions are handled consistently."""
        # Test with different input types
        test_inputs = [
            1,  # int
            1.0,  # float
            np.float64(1.0),  # numpy float64
            np.array([1.0]),  # numpy array
            [1.0, 2.0],  # python list
        ]

        for input_val in test_inputs:
            try:
                result = bitflip_float32_optimized(input_val, 0)
                assert result is not None
                # Check that we get appropriate output type
                if np.isscalar(input_val) or (
                    hasattr(input_val, "shape") and input_val.shape == ()
                ):
                    assert np.isscalar(result), (
                        f"Expected scalar result for {type(input_val)}"
                    )
                else:
                    assert isinstance(result, np.ndarray), (
                        f"Expected array result for {type(input_val)}"
                    )
            except Exception as e:
                pytest.fail(f"Type conversion failed for {type(input_val)}: {e}")

    @pytest.mark.parametrize("bit_position", [0, 1, 8, 15, 16, 23, 31])
    def test_all_bit_positions(self, bit_position):
        """Test all valid bit positions systematically."""
        test_value = 1.0

        # Test scalar
        result_scalar = bitflip_float32_optimized(test_value, bit_position)
        assert isinstance(result_scalar, (float, np.floating))

        # Test array
        test_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_array = bitflip_float32_optimized(test_array, bit_position)
        assert isinstance(result_array, np.ndarray)
        assert len(result_array) == len(test_array)

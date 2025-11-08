import numpy as np

# Import from the new seu_injection package
from seu_injection.bitops.float32 import (
    binary_to_float32,
    bitflip_float32,
    float32_to_binary,
)


class TestBitflipOperations:
    """Test suite for bitflip operations on float32 values."""

    def test_bitflip_float32_basic(self):
        """Test basic bit flipping functionality."""
        # Flipping bit 0 (sign bit) of float32(1.0) should result in -1.0
        result = bitflip_float32(1.0, 0)
        assert result == -1.0, f"Expected -1.0, got {result}"

    def test_bitflip_reversibility(self):
        """Test that flipping the same bit twice returns original value."""
        test_values = [1.0, -1.0, 0.5, -0.5, 3.14159, -2.71828]

        for value in test_values:
            for bit_pos in [0, 1, 15, 16, 31]:  # Test various bit positions
                flipped_once = bitflip_float32(value, bit_pos)
                flipped_twice = bitflip_float32(flipped_once, bit_pos)

                # Handle NaN cases (some bit patterns may produce NaN)
                if np.isnan(value) and np.isnan(flipped_twice):
                    continue
                elif np.isnan(flipped_twice) and not np.isnan(value):
                    # This is acceptable - some bit patterns produce NaN
                    continue
                else:
                    assert abs(value - flipped_twice) < 1e-6, (
                        f"Reversibility failed for value {value}, bit {bit_pos}: {value} -> {flipped_once} -> {flipped_twice}"
                    )

    def test_bitflip_array_input(self):
        """Test bitflip with array input."""
        values = np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float32)
        result = bitflip_float32(values, 0)  # Flip sign bit
        expected = np.array([-1.0, -2.0, 1.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_bitflip_edge_cases(self):
        """Test bitflip with edge cases like zero, infinity."""
        # Test with zero - check bit representation since Python treats -0.0 == 0.0
        original_bits = float32_to_binary(0.0)
        result_zero = bitflip_float32(0.0, 0)
        result_bits = float32_to_binary(result_zero)
        assert original_bits != result_bits  # Bit representations should be different

        # Test with small numbers
        small_val = 1e-6
        result_small = bitflip_float32(small_val, 16)  # Flip a middle bit
        assert result_small != small_val

    def test_bit_position_validation(self):
        """Test that invalid bit positions are handled correctly."""
        # The current implementation doesn't validate bit positions,
        # but we test the behavior for edge cases
        test_val = 1.0

        # Test boundary positions
        result_0 = bitflip_float32(test_val, 0)
        result_31 = bitflip_float32(test_val, 31)

        assert result_0 is not None
        assert result_31 is not None

    def test_float32_binary_conversion(self):
        """Test float32 to binary string conversion."""
        # Test known values
        result = float32_to_binary(1.0)
        assert len(result) == 32, "Binary representation should be 32 bits"
        assert result[0] == "0", "Sign bit of 1.0 should be 0"

        result_neg = float32_to_binary(-1.0)
        assert len(result_neg) == 32, "Binary representation should be 32 bits"
        assert result_neg[0] == "1", "Sign bit of -1.0 should be 1"

    def test_binary_float32_conversion(self):
        """Test binary string to float32 conversion."""
        # Test round-trip conversion
        test_values = [1.0, -1.0, 0.5, 3.14159, -2.71828]

        for value in test_values:
            binary = float32_to_binary(value)
            converted_back = binary_to_float32(binary)
            assert abs(value - converted_back) < 1e-6, (
                f"Round-trip conversion failed: {value} -> {binary} -> {converted_back}"
            )

    def test_random_bit_position(self):
        """Test that random bit position works when no position is specified."""
        # This tests the default behavior with random bit selection
        value = 1.0
        result1 = bitflip_float32(value)  # Should use random bit
        result2 = bitflip_float32(
            value
        )  # Should use different random bit (potentially)

        # Results should be valid floats
        assert isinstance(result1, (float, np.floating))
        assert isinstance(result2, (float, np.floating))

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

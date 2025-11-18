import time

import numpy as np

from seu_injection.bitops.float32 import bitflip_float32, bitflip_float32_optimized


class TestBitflipPerformance:
    def test_performance_improvement_scalar(self):
        """Test performance improvement for scalar operations with realistic benchmarks."""
        test_value = 3.14159
        iterations = 10000

        # Warm-up runs
        for _ in range(100):
            bitflip_float32(test_value, 15)
            bitflip_float32_optimized(test_value, 15)

        # Measure original implementation
        times_original = []
        for _ in range(5):
            start_time = time.perf_counter()
            for _ in range(iterations):
                bitflip_float32(test_value, 15)
            times_original.append(time.perf_counter() - start_time)
        original_time = min(times_original)

        # Measure optimized implementation
        times_optimized = []
        for _ in range(5):
            start_time = time.perf_counter()
            for _ in range(iterations):
                bitflip_float32_optimized(test_value, 15)
            times_optimized.append(time.perf_counter() - start_time)
        optimized_time = min(times_optimized)

        speedup = original_time / optimized_time
        assert speedup >= 1.0, f"Expected at least 1.0x speedup, got {speedup:.1f}x."

    def test_performance_improvement_array(self):
        """Test performance improvement for array operations with realistic neural network sizes."""
        array_size = 5000
        iterations = 100
        test_array = np.random.randn(array_size).astype(np.float32)

        # Warm-up runs
        bitflip_float32(test_array[:100], 15)
        bitflip_float32_optimized(test_array[:100], 15)

        # Measure original implementation
        times_original = []
        for _ in range(3):
            start_time = time.perf_counter()
            for _ in range(iterations):
                bitflip_float32(test_array, 15)
            times_original.append(time.perf_counter() - start_time)
        original_time = min(times_original)

        # Measure optimized implementation
        times_optimized = []
        for _ in range(3):
            start_time = time.perf_counter()
            for _ in range(iterations):
                bitflip_float32_optimized(test_array, 15)
            times_optimized.append(time.perf_counter() - start_time)
        optimized_time = min(times_optimized)

        speedup = original_time / optimized_time
        assert speedup >= 5.0, f"Expected at least 5.0x speedup, got {speedup:.1f}x."

    def test_memory_efficiency(self):
        """Test memory efficiency of optimized operations."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        large_array = np.random.randn(100000).astype(np.float32)

        # Measure memory before operation
        memory_before = process.memory_info().rss
        bitflip_float32_optimized(large_array, 15, inplace=False)
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        array_size = large_array.nbytes
        assert memory_increase < 3 * array_size, f"Memory increase too large: {memory_increase} bytes"

        # Test inplace operation
        memory_before_inplace = process.memory_info().rss
        bitflip_float32_optimized(large_array, 16, inplace=True)
        memory_after_inplace = process.memory_info().rss
        inplace_increase = memory_after_inplace - memory_before_inplace
        assert inplace_increase < array_size, f"Inplace memory increase too large: {inplace_increase} bytes"

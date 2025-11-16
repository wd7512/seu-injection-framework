"""
Performance benchmarks for SEU injection framework.

These tests validate that core operations meet performance requirements
and don't regress over time. Based on testing/benchmark.py.
"""

import platform
import time

import pytest
import torch
import torch.nn as nn

# Type annotations use built-in dict instead of typing.Dict


class SmallConvNet(nn.Module):
    """Small CNN for performance benchmarking."""

    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for the SEU injection framework."""

    @pytest.fixture(scope="class")
    def device(self):
        """Get the best available device for testing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture(scope="class")
    def model(self, device):
        """Create a test model for benchmarking."""
        model = SmallConvNet().to(device)
        model.eval()
        return model

    def test_baseline_model_performance(self, model, device):
        """Benchmark baseline model inference time."""
        input_size = (8, 3, 64, 64)  # Batch, channels, height, width
        x = torch.randn(input_size, device=device)

        def run_inference():
            with torch.no_grad():
                return model(x)

        # Manual timing without pytest-benchmark dependency
        start = time.time()
        for _ in range(10):
            run_inference()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / 10

        # Performance assertion: should complete in reasonable time
        assert avg_time < 1.0, (
            f"Model inference too slow: {avg_time:.4f}s per forward pass"
        )
        result = run_inference()

        assert result is not None
        assert result.shape == (8, 10)  # Batch size 8, 10 classes

    def test_bitflip_operation_performance(self):
        """Benchmark bitflip operations for performance regression."""
        import numpy as np

        from seu_injection.bitops.float32 import bitflip_float32

        # Test data
        test_values = np.random.randn(1000).astype(np.float32)

        def run_bitflips():
            results = []
            for value in test_values[:100]:  # Subset for timing
                result = bitflip_float32(value, 0)
                results.append(result)
            return results

        # Manual timing
        start = time.time()
        results = run_bitflips()
        end = time.time()

        # Performance assertion: bitflip should be reasonably fast
        time_per_flip = (end - start) / 100
        assert time_per_flip < 0.001, f"Bitflip too slow: {time_per_flip:.6f}s per flip"

        assert len(results) == 100
        # Verify bitflips actually changed values (sign bit flip)
        assert all(r != orig for r, orig in zip(results, test_values[:100]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_performance_vs_cpu(self, model):
        """Compare CUDA vs CPU performance (if CUDA available)."""
        input_size = (4, 3, 64, 64)

        # CPU timing
        model_cpu = SmallConvNet()
        model_cpu.eval()
        x_cpu = torch.randn(input_size)

        start = time.time()
        with torch.no_grad():
            for _ in range(5):
                model_cpu(x_cpu)
        cpu_time = time.time() - start

        # GPU timing
        model_gpu = model.cuda()
        x_gpu = torch.randn(input_size, device="cuda")

        # Warmup
        with torch.no_grad():
            model_gpu(x_gpu)
        torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            for _ in range(5):
                model_gpu(x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        # GPU should be faster (or at least not significantly slower)
        # Allow GPU to be up to 2x slower for small models due to overhead
        assert gpu_time < cpu_time * 2, (
            f"GPU ({gpu_time:.4f}s) much slower than CPU ({cpu_time:.4f}s)"
        )

    def test_memory_usage_reasonable(self, model, device):
        """Test that memory usage stays within reasonable bounds."""
        if device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Run several forward passes
            for batch_size in [1, 4, 8, 16]:
                x = torch.randn(batch_size, 3, 64, 64, device=device)
                with torch.no_grad():
                    _ = model(x)

            current_memory = torch.cuda.memory_allocated()
            memory_increase = current_memory - initial_memory

            # Memory increase should be reasonable (< 100MB for this small model)
            assert memory_increase < 100 * 1024 * 1024, (
                f"Memory usage too high: {memory_increase / 1024 / 1024:.1f}MB"
            )

    def test_framework_import_performance(self):
        """Test that framework imports are reasonably fast."""

        def import_framework():
            # Clear module cache to force re-import
            import sys

            modules_to_clear = [
                k for k in sys.modules.keys() if k.startswith("seu_injection")
            ]
            for module in modules_to_clear:
                if module in sys.modules:
                    del sys.modules[module]

            # Import framework
            from seu_injection.bitops.float32 import bitflip_float32
            from seu_injection.core import ExhaustiveSEUInjector
            from seu_injection.metrics import classification_accuracy

            return ExhaustiveSEUInjector, classification_accuracy, bitflip_float32

        start = time.time()
        classes = import_framework()
        import_time = time.time() - start

        # Import should be fast (< 1 second)
        assert import_time < 1.0, f"Framework import too slow: {import_time:.3f}s"

        assert all(c is not None for c in classes)

    def generate_performance_report(self, device) -> dict:
        """Generate a comprehensive performance report."""
        report = {
            "device": str(device),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
        }

        if device.type == "cuda":
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory

        return report

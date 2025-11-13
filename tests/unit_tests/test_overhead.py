"""
Unit tests for overhead calculation utilities.

Tests the functionality for measuring and analyzing SEU injection overhead.
"""

import pytest
import torch
import torch.nn as nn

from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy
from seu_injection.utils.overhead import (
    calculate_overhead,
    format_overhead_report,
    measure_inference_time,
    measure_seu_injection_time,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = SimpleModel()
    model.eval()
    return model


@pytest.fixture
def test_data():
    """Create test data."""
    x = torch.randn(50, 10)
    y = torch.randint(0, 2, (50,)).float().unsqueeze(1)
    return x, y


@pytest.fixture
def injector(simple_model, test_data):
    """Create an SEUInjector instance."""
    x, y = test_data
    return SEUInjector(
        trained_model=simple_model, criterion=classification_accuracy, x=x, y=y
    )


class TestMeasureInferenceTime:
    """Tests for measure_inference_time function."""

    def test_basic_measurement(self, simple_model):
        """Test basic inference time measurement."""
        input_data = torch.randn(1, 10)
        results = measure_inference_time(
            model=simple_model, input_data=input_data, num_iterations=10
        )

        assert "total_time" in results
        assert "avg_time" in results
        assert "avg_time_ms" in results
        assert "iterations" in results

        assert results["iterations"] == 10
        assert results["total_time"] > 0
        assert results["avg_time"] > 0
        assert results["avg_time_ms"] > 0
        assert abs(results["avg_time_ms"] - results["avg_time"] * 1000) < 0.001

    def test_different_batch_sizes(self, simple_model):
        """Test measurement with different batch sizes."""
        for batch_size in [1, 4, 8]:
            input_data = torch.randn(batch_size, 10)
            results = measure_inference_time(
                model=simple_model, input_data=input_data, num_iterations=5
            )

            assert results["total_time"] > 0
            assert results["avg_time"] > 0

    def test_with_warmup(self, simple_model):
        """Test that warmup iterations don't affect timing."""
        input_data = torch.randn(1, 10)
        results = measure_inference_time(
            model=simple_model,
            input_data=input_data,
            num_iterations=10,
            warmup_iterations=5,
        )

        assert results["iterations"] == 10  # Warmup not counted
        assert results["total_time"] > 0


class TestMeasureSEUInjectionTime:
    """Tests for measure_seu_injection_time function."""

    def test_systematic_injection_timing(self, injector):
        """Test timing for systematic injection."""
        results = measure_seu_injection_time(
            injector=injector, bit_position=0, stochastic=False
        )

        assert "total_time" in results
        assert "num_injections" in results
        assert "avg_time_per_injection" in results
        assert "avg_time_per_injection_ms" in results
        assert "results" in results

        assert results["total_time"] > 0
        assert results["num_injections"] > 0
        assert results["avg_time_per_injection"] > 0

    def test_stochastic_injection_timing(self, injector):
        """Test timing for stochastic injection."""
        results = measure_seu_injection_time(
            injector=injector,
            bit_position=0,
            stochastic=True,
            stochastic_probability=0.1,
        )

        assert results["total_time"] > 0
        assert results["num_injections"] >= 0  # May be 0 with low probability
        assert "results" in results

    def test_layer_specific_timing(self, injector):
        """Test timing for layer-specific injection."""
        # Get first layer name
        layer_name = list(injector.model.named_parameters())[0][0]

        results = measure_seu_injection_time(
            injector=injector, bit_position=0, layer_name=layer_name, stochastic=False
        )

        assert results["num_injections"] > 0
        assert results["total_time"] > 0


class TestCalculateOverhead:
    """Tests for calculate_overhead function."""

    def test_basic_overhead_calculation(self, simple_model, injector):
        """Test basic overhead calculation."""
        input_data = torch.randn(1, 10)

        results = calculate_overhead(
            model=simple_model,
            injector=injector,
            input_data=input_data,
            bit_position=0,
            num_baseline_iterations=10,
            stochastic=True,
            stochastic_probability=0.1,
        )

        assert "baseline" in results
        assert "injection" in results
        assert "overhead_absolute" in results
        assert "overhead_absolute_ms" in results
        assert "overhead_relative" in results
        assert "overhead_per_injection" in results
        assert "throughput_baseline" in results
        assert "throughput_with_injection" in results

        # Verify calculations
        assert results["overhead_absolute"] >= 0
        assert results["overhead_relative"] >= 0
        assert results["throughput_baseline"] > 0

    def test_overhead_is_positive(self, simple_model, injector):
        """Test that overhead is positive (injection slower than baseline)."""
        input_data = torch.randn(1, 10)

        results = calculate_overhead(
            model=simple_model,
            injector=injector,
            input_data=input_data,
            bit_position=0,
            num_baseline_iterations=10,
            stochastic=True,
            stochastic_probability=0.05,
        )

        # Injection should take longer than baseline
        if results["injection"]["num_injections"] > 0:
            assert results["overhead_absolute"] > 0, (
                "Overhead should be positive (injection is slower)"
            )

    def test_different_bit_positions(self, simple_model, injector):
        """Test overhead calculation with different bit positions."""
        input_data = torch.randn(1, 10)

        for bit_pos in [0, 15, 31]:
            results = calculate_overhead(
                model=simple_model,
                injector=injector,
                input_data=input_data,
                bit_position=bit_pos,
                num_baseline_iterations=5,
                stochastic=True,
                stochastic_probability=0.05,
            )

            assert results["baseline"]["total_time"] > 0
            # Overhead calculation should work for all bit positions


class TestFormatOverheadReport:
    """Tests for format_overhead_report function."""

    def test_report_formatting(self, simple_model, injector):
        """Test that report formatting produces valid output."""
        input_data = torch.randn(1, 10)

        overhead_results = calculate_overhead(
            model=simple_model,
            injector=injector,
            input_data=input_data,
            bit_position=0,
            num_baseline_iterations=10,
            stochastic=True,
            stochastic_probability=0.1,
        )

        report = format_overhead_report(overhead_results)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "OVERHEAD ANALYSIS" in report
        assert "BASELINE INFERENCE" in report
        assert "SEU INJECTION CAMPAIGN" in report

    def test_report_contains_metrics(self, simple_model, injector):
        """Test that report contains expected metrics."""
        input_data = torch.randn(1, 10)

        overhead_results = calculate_overhead(
            model=simple_model,
            injector=injector,
            input_data=input_data,
            bit_position=0,
            num_baseline_iterations=10,
            stochastic=True,
            stochastic_probability=0.1,
        )

        report = format_overhead_report(overhead_results)

        # Check for key metrics in report
        assert "ms" in report  # Time unit
        assert "%" in report or "injections" in report  # Percentage or count


class TestIntegration:
    """Integration tests for overhead calculation workflow."""

    def test_full_workflow(self, simple_model, test_data):
        """Test complete overhead calculation workflow."""
        x, y = test_data
        input_data = torch.randn(1, 10)

        # Create injector
        injector = SEUInjector(
            trained_model=simple_model, criterion=classification_accuracy, x=x, y=y
        )

        # Measure baseline
        baseline = measure_inference_time(
            model=simple_model, input_data=input_data, num_iterations=10
        )

        # Measure injection
        injection = measure_seu_injection_time(
            injector=injector,
            bit_position=0,
            stochastic=True,
            stochastic_probability=0.05,
        )

        # Calculate overhead
        overhead = calculate_overhead(
            model=simple_model,
            injector=injector,
            input_data=input_data,
            bit_position=0,
            num_baseline_iterations=10,
            stochastic=True,
            stochastic_probability=0.05,
        )

        # Format report
        report = format_overhead_report(overhead)

        # Verify all steps completed
        assert baseline["total_time"] > 0
        assert injection["total_time"] > 0
        assert overhead["overhead_absolute"] >= 0
        assert len(report) > 0

    def test_multiple_models_comparison(self, test_data):
        """Test comparing overhead across multiple models."""
        x, y = test_data

        # Create models of different sizes
        small_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        large_model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2))

        overheads = []

        for model in [small_model, large_model]:
            model.eval()
            injector = SEUInjector(
                trained_model=model, criterion=classification_accuracy, x=x, y=y
            )

            overhead = calculate_overhead(
                model=model,
                injector=injector,
                input_data=torch.randn(1, 10),
                bit_position=0,
                num_baseline_iterations=5,
                stochastic=True,
                stochastic_probability=0.05,
            )

            overheads.append(overhead)

        # Both models should have valid overhead measurements
        assert all(o["baseline"]["total_time"] > 0 for o in overheads)
        assert all(o["overhead_absolute"] >= 0 for o in overheads)

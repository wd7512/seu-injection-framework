"""
Overhead calculation utilities for SEU injection framework.

This module provides tools to measure and analyze the performance overhead
introduced by SEU injection operations compared to baseline inference.
"""

import time
from typing import Any, Callable, Optional

import torch
import torch.nn as nn


def measure_inference_time(
    model: nn.Module,
    input_data: torch.Tensor,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """
    Measure baseline inference time for a model without SEU injection.

    Args:
        model: PyTorch model to benchmark
        input_data: Input tensor for inference
        num_iterations: Number of inference iterations to average
        warmup_iterations: Number of warmup iterations before timing
        device: Device to run on (defaults to model's current device)

    Returns:
        Dictionary containing timing metrics:
            - total_time: Total time for all iterations (seconds)
            - avg_time: Average time per iteration (seconds)
            - avg_time_ms: Average time per iteration (milliseconds)
            - iterations: Number of iterations performed
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_data)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Actual timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_data)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations

    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "avg_time_ms": avg_time * 1000,
        "iterations": num_iterations,
    }


def measure_seu_injection_time(
    injector: Any,
    bit_position: int,
    layer_name: Optional[str] = None,
    stochastic: bool = False,
    stochastic_probability: float = 0.01,
) -> dict[str, Any]:
    """
    Measure time taken for SEU injection operations.

    Args:
        injector: SEUInjector instance
        bit_position: Bit position to inject faults
        layer_name: Specific layer to target (None for all layers)
        stochastic: Use stochastic injection instead of systematic
        stochastic_probability: Probability for stochastic injection

    Returns:
        Dictionary containing timing and injection metrics:
            - total_time: Total time for injection campaign (seconds)
            - num_injections: Number of injections performed
            - avg_time_per_injection: Average time per injection (seconds)
            - avg_time_per_injection_ms: Average time per injection (milliseconds)
            - results: Full injection results dictionary
    """
    start_time = time.time()

    if stochastic:
        results = injector.run_stochastic_seu(
            bit_i=bit_position, p=stochastic_probability
        )
    else:
        results = injector.run_seu(bit_i=bit_position, layer_name=layer_name)

    end_time = time.time()
    total_time = end_time - start_time

    num_injections = len(results["criterion_score"])
    avg_time_per_injection = total_time / num_injections if num_injections > 0 else 0

    return {
        "total_time": total_time,
        "num_injections": num_injections,
        "avg_time_per_injection": avg_time_per_injection,
        "avg_time_per_injection_ms": avg_time_per_injection * 1000,
        "results": results,
    }


def calculate_overhead(
    model: nn.Module,
    injector: Any,
    input_data: torch.Tensor,
    bit_position: int = 0,
    num_baseline_iterations: int = 100,
    layer_name: Optional[str] = None,
    stochastic: bool = False,
    stochastic_probability: float = 0.01,
) -> dict[str, Any]:
    """
    Calculate the overhead of SEU injection compared to baseline inference.

    This function measures both baseline inference time and SEU injection time,
    then calculates the overhead introduced by the fault injection process.

    Args:
        model: PyTorch model to analyze
        injector: SEUInjector instance configured for the model
        input_data: Input tensor for baseline inference timing
        bit_position: Bit position for SEU injection
        num_baseline_iterations: Number of iterations for baseline timing
        layer_name: Specific layer to target (None for all layers)
        stochastic: Use stochastic injection instead of systematic
        stochastic_probability: Probability for stochastic injection

    Returns:
        Dictionary containing comprehensive overhead analysis:
            - baseline: Baseline inference timing metrics
            - injection: SEU injection timing metrics
            - overhead_absolute: Absolute time overhead (seconds)
            - overhead_relative: Relative overhead as percentage
            - overhead_per_injection: Time overhead per single injection (seconds)
            - throughput_baseline: Baseline inferences per second
            - throughput_with_injection: Effective throughput with injection
    """
    # Measure baseline inference time
    baseline_metrics = measure_inference_time(
        model=model,
        input_data=input_data,
        num_iterations=num_baseline_iterations,
    )

    # Measure SEU injection time
    injection_metrics = measure_seu_injection_time(
        injector=injector,
        bit_position=bit_position,
        layer_name=layer_name,
        stochastic=stochastic,
        stochastic_probability=stochastic_probability,
    )

    # Calculate overhead
    baseline_time = baseline_metrics["avg_time"]
    injection_time_per_eval = (
        injection_metrics["avg_time_per_injection"]
        if injection_metrics["num_injections"] > 0
        else 0
    )

    overhead_absolute = injection_time_per_eval - baseline_time
    overhead_relative = (
        (overhead_absolute / baseline_time * 100) if baseline_time > 0 else 0
    )

    return {
        "baseline": baseline_metrics,
        "injection": injection_metrics,
        "overhead_absolute": overhead_absolute,
        "overhead_absolute_ms": overhead_absolute * 1000,
        "overhead_relative": overhead_relative,
        "overhead_per_injection": injection_time_per_eval,
        "throughput_baseline": 1.0 / baseline_time if baseline_time > 0 else 0,
        "throughput_with_injection": (
            1.0 / injection_time_per_eval if injection_time_per_eval > 0 else 0
        ),
    }


def benchmark_multiple_networks(
    networks: list[tuple[str, nn.Module, torch.Tensor]],
    criterion: Callable,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    bit_position: int = 0,
    num_baseline_iterations: int = 100,
    device: Optional[torch.device] = None,
) -> dict[str, dict[str, Any]]:
    """
    Benchmark overhead across multiple network architectures.

    Args:
        networks: List of (name, model, sample_input) tuples
        criterion: Criterion function for SEUInjector
        x_test: Test data for criterion evaluation
        y_test: Test labels for criterion evaluation
        bit_position: Bit position for SEU injection
        num_baseline_iterations: Number of iterations for baseline timing
        device: Device to use for computation

    Returns:
        Dictionary mapping network names to their overhead analysis results
    """
    from ..core.injector import SEUInjector

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for name, model, sample_input in networks:
        print(f"\nBenchmarking {name}...")

        # Move model and data to device
        model = model.to(device)
        sample_input = sample_input.to(device)
        x_test_dev = x_test.to(device) if isinstance(x_test, torch.Tensor) else x_test
        y_test_dev = y_test.to(device) if isinstance(y_test, torch.Tensor) else y_test

        # Create injector
        injector = SEUInjector(
            trained_model=model,
            criterion=criterion,
            x=x_test_dev,
            y=y_test_dev,
            device=device,
        )

        # Calculate overhead using stochastic sampling for efficiency
        overhead_results = calculate_overhead(
            model=model,
            injector=injector,
            input_data=sample_input,
            bit_position=bit_position,
            num_baseline_iterations=num_baseline_iterations,
            stochastic=True,
            stochastic_probability=0.01,  # 1% sampling
        )

        results[name] = overhead_results

    return results


def format_overhead_report(overhead_results: dict[str, Any]) -> str:
    """
    Format overhead analysis results as a readable report.

    Args:
        overhead_results: Results from calculate_overhead function

    Returns:
        Formatted string report
    """
    baseline = overhead_results["baseline"]
    injection = overhead_results["injection"]

    report = [
        "=" * 60,
        "SEU INJECTION OVERHEAD ANALYSIS",
        "=" * 60,
        "",
        "BASELINE INFERENCE (without SEU injection):",
        f"  Average time per inference: {baseline['avg_time_ms']:.2f} ms",
        f"  Total iterations: {baseline['iterations']}",
        f"  Throughput: {overhead_results['throughput_baseline']:.1f} inferences/sec",
        "",
        "SEU INJECTION CAMPAIGN:",
        f"  Total injections performed: {injection['num_injections']}",
        f"  Total time: {injection['total_time']:.2f} seconds",
        f"  Average time per injection: {injection['avg_time_per_injection_ms']:.2f} ms",
        "",
        "OVERHEAD ANALYSIS:",
        f"  Absolute overhead: {overhead_results['overhead_absolute_ms']:.2f} ms per injection",
        f"  Relative overhead: {overhead_results['overhead_relative']:.1f}%",
        f"  Baseline inference: {baseline['avg_time_ms']:.2f} ms",
        f"  Injection + evaluation: {overhead_results['overhead_per_injection'] * 1000:.2f} ms",
        "",
        "INTERPRETATION:",
        f"  Each SEU injection takes {overhead_results['overhead_relative']:.1f}% more time than baseline inference",
        f"  Throughput with injection: {overhead_results['throughput_with_injection']:.1f} injections/sec",
        "=" * 60,
    ]

    return "\n".join(report)

#!/usr/bin/env python3
"""Minimal overhead measurement for SEU injection operations."""

import csv
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from seu_injection.core import StochasticSEUInjector
from seu_injection.metrics import classification_accuracy


def create_test_model():
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


def prepare_data():
    """Prepare test data."""
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X, y


def measure_baseline(model, input_data, iterations=50):
    """Measure baseline inference time."""
    model.eval()
    # Warmup (extra)
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_data)
    total_time = time.perf_counter() - start
    return total_time / iterations


def measure_injection(injector, bit_position=0, probability=0.05):
    """Measure SEU injection time."""
    # Warmup injector
    for _ in range(2):
        _ = injector.run_injector(bit_i=bit_position, p=probability)

    start = time.perf_counter()
    results = injector.run_injector(bit_i=bit_position, p=probability)
    total_time = time.perf_counter() - start
    num_injections = len(results["criterion_score"])
    return total_time, num_injections


def main():
    """Run overhead measurement."""
    print("SEU Injection Overhead Measurement")
    print("=" * 50)

    # Setup
    model = create_test_model()
    x_test, y_test = prepare_data()
    sample_input = torch.randn(1, 2)

    # Measure baseline
    print("\nMeasuring baseline inference...")
    baseline_time = measure_baseline(model, sample_input, iterations=100)
    print(f"Baseline: {baseline_time * 1000:.3f} ms per inference")

    # Measure injection
    print("\nMeasuring SEU injection overhead...")
    injector = StochasticSEUInjector(
        trained_model=model, criterion=classification_accuracy, x=x_test, y=y_test
    )
    injection_time, num_injections = measure_injection(injector)
    avg_injection_time = injection_time / num_injections if num_injections > 0 else 0

    print(f"Injections: {num_injections}")
    print(f"Average injection time: {avg_injection_time * 1000:.3f} ms")

    # Calculate overhead
    overhead_abs = avg_injection_time - baseline_time
    overhead_rel = (overhead_abs / baseline_time * 100) if baseline_time > 0 else 0

    print(f"\nOverhead: {overhead_abs * 1000:.3f} ms ({overhead_rel:.1f}%)")

    # Gather system info
    import os
    import platform

    import psutil

    sys_info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    # Battery info (laptop detection)
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if battery:
        sys_info["is_laptop"] = True
        sys_info["battery_percent"] = battery.percent
        sys_info["plugged_in"] = battery.power_plugged
    else:
        sys_info["is_laptop"] = False
        sys_info["battery_percent"] = None
        sys_info["plugged_in"] = None

    # Results
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline_ms": baseline_time * 1000,
        "injection_ms": avg_injection_time * 1000,
        "overhead_ms": overhead_abs * 1000,
        "overhead_percent": overhead_rel,
        "num_injections": num_injections,
        "system_info": sys_info,
    }

    # Save JSON in results folder with timestamp
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"overhead_results_{timestamp_str}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

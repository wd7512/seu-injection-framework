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
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_data)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_data)
    total_time = time.perf_counter() - start
    return total_time / iterations


def measure_injection(injector, bit_position=0, probability=0.05):
    """Measure SEU injection time."""
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
    baseline_time = measure_baseline(model, sample_input)
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

    # Save JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline_ms": baseline_time * 1000,
        "injection_ms": avg_injection_time * 1000,
        "overhead_ms": overhead_abs * 1000,
        "overhead_percent": overhead_rel,
        "num_injections": num_injections,
    }

    with open("overhead_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    with open("overhead_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print("\nResults saved to overhead_results.json and overhead_results.csv")
    print("=" * 50)


if __name__ == "__main__":
    main()

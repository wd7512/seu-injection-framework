#!/usr/bin/env python3
"""
SEU Injection Overhead Calculation Example

This script demonstrates how to measure the performance overhead of SEU injection
operations compared to baseline inference. It tests several small network architectures
and generates a comprehensive overhead report.

Usage:
    python overhead_calculation_example.py

Output:
    - Console output with overhead metrics for each network
    - Summary report comparing overhead across architectures
"""

import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# SEU Injection Framework imports
from seu_injection import (
    SEUInjector,
    calculate_overhead,
    format_overhead_report,
    measure_inference_time,
)
from seu_injection.metrics import classification_accuracy


def create_small_mlp():
    """Create a small Multi-Layer Perceptron."""
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


def create_medium_mlp():
    """Create a medium Multi-Layer Perceptron."""
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


def create_large_mlp():
    """Create a larger Multi-Layer Perceptron."""
    return nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


def prepare_data():
    """Prepare training and test data."""
    print("Preparing dataset...")

    # Generate moon-shaped data for binary classification
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Convert to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    print(f"Dataset ready: {len(X_test)} test samples")
    return X_test, y_test


def train_model(model, x_train, y_train, epochs=50):
    """Quickly train a model for demonstration."""
    print("Training model...")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    print("Training complete")
    return model


def measure_network_overhead(
    name: str, model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor
):
    """Measure overhead for a single network architecture."""
    print(f"\n{'=' * 60}")
    print(f"ANALYZING: {name}")
    print(f"{'=' * 60}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Create a sample input for baseline timing
    sample_input = torch.randn(1, 2)

    # Measure baseline inference time
    print("\nMeasuring baseline inference time...")
    baseline_metrics = measure_inference_time(
        model=model, input_data=sample_input, num_iterations=100
    )
    print(
        f"Baseline inference: {baseline_metrics['avg_time_ms']:.3f} ms per inference"
    )

    # Create SEU injector
    print("\nInitializing SEU injector...")
    injector = SEUInjector(
        trained_model=model, criterion=classification_accuracy, x=x_test, y=y_test
    )
    print(f"Baseline accuracy: {injector.baseline_score:.2%}")

    # Calculate overhead with stochastic sampling for efficiency
    print("\nCalculating SEU injection overhead (using 1% stochastic sampling)...")
    overhead_results = calculate_overhead(
        model=model,
        injector=injector,
        input_data=sample_input,
        bit_position=0,  # Sign bit
        num_baseline_iterations=100,
        stochastic=True,
        stochastic_probability=0.01,  # 1% sampling
    )

    # Print detailed report
    print("\n" + format_overhead_report(overhead_results))

    return {
        "name": name,
        "num_params": num_params,
        "baseline_ms": baseline_metrics["avg_time_ms"],
        "overhead_ms": overhead_results["overhead_absolute_ms"],
        "overhead_percent": overhead_results["overhead_relative"],
        "num_injections": overhead_results["injection"]["num_injections"],
    }


def main():
    """Main overhead analysis pipeline."""
    print("=" * 60)
    print("SEU INJECTION OVERHEAD CALCULATION")
    print("=" * 60)
    print("\nThis example demonstrates how to measure the performance overhead")
    print("of SEU injection operations on different network architectures.\n")

    # Prepare data
    x_test, y_test = prepare_data()

    # For quick training, we'll use a subset for training
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Test different network architectures
    networks = [
        ("Small MLP (2→32→16→1)", create_small_mlp()),
        ("Medium MLP (2→64→32→16→1)", create_medium_mlp()),
        ("Large MLP (2→128→64→32→16→1)", create_large_mlp()),
    ]

    results = []

    for name, model in networks:
        # Train the model
        print(f"\n\nPreparing {name}...")
        model = train_model(model, X_train, y_train, epochs=30)

        # Measure overhead
        result = measure_network_overhead(name, model, x_test, y_test)
        results.append(result)

    # Print summary comparison
    print("\n\n" + "=" * 60)
    print("SUMMARY: OVERHEAD COMPARISON ACROSS ARCHITECTURES")
    print("=" * 60)
    print(
        f"\n{'Architecture':<35} {'Params':>10} {'Base (ms)':>12} {'Overhead':>12} {'% Overhead':>12}"
    )
    print("-" * 85)

    for r in results:
        print(
            f"{r['name']:<35} {r['num_params']:>10,} "
            f"{r['baseline_ms']:>12.3f} {r['overhead_ms']:>12.3f} "
            f"{r['overhead_percent']:>11.1f}%"
        )

    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("-" * 60)

    # Calculate some statistics
    avg_overhead = sum(r["overhead_percent"] for r in results) / len(results)
    min_overhead = min(results, key=lambda x: x["overhead_percent"])
    max_overhead = max(results, key=lambda x: x["overhead_percent"])

    print(f"Average overhead across networks: {avg_overhead:.1f}%")
    print(
        f"Lowest overhead: {min_overhead['name']} ({min_overhead['overhead_percent']:.1f}%)"
    )
    print(
        f"Highest overhead: {max_overhead['name']} ({max_overhead['overhead_percent']:.1f}%)"
    )

    total_injections = sum(r["num_injections"] for r in results)
    print(f"\nTotal SEU injections performed: {total_injections:,}")

    print("\nINTERPRETATION:")
    print("-" * 60)
    print(
        "The overhead represents the additional time required for SEU injection"
    )
    print(
        "compared to normal inference. This includes parameter backup, bit flipping,"
    )
    print("model evaluation, and parameter restoration.")
    print(
        f"\nFor the tested networks, SEU injection adds approximately {avg_overhead:.0f}% overhead."
    )
    print("This is the price of comprehensive fault tolerance analysis.")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

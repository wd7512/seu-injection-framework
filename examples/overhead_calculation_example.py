#!/usr/bin/env python3
"""
SEU Injection Overhead Calculation Example

This script demonstrates how to measure the performance overhead of SEU injection
operations compared to baseline inference. It tests several small network architectures
and generates overhead reports in JSON/CSV format.

Usage:
    python overhead_calculation_example.py

Output:
    - Console output with overhead metrics
    - overhead_results.json (structured results)
    - overhead_results.csv (tabular results)
"""

import csv
import json
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# SEU Injection Framework imports
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy
from seu_injection.utils.overhead import (
    calculate_overhead,
    format_overhead_report,
    measure_inference_time,
)


def create_mlp(layer_sizes):
    """
    Create a Multi-Layer Perceptron with specified layer sizes.

    Args:
        layer_sizes: List of layer sizes, e.g., [2, 32, 16, 1]

    Returns:
        Sequential model with ReLU activations and final Sigmoid
    """
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:  # ReLU for hidden layers
            layers.append(nn.ReLU())
        else:  # Sigmoid for output
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def prepare_data():
    """Prepare training and test data."""
    print("Preparing dataset...")
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    print(f"Dataset ready: {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test


def train_model(model, x_train, y_train, epochs=30):
    """Quickly train a model for demonstration."""
    print("Training model...")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    print("Training complete")
    return model


def measure_network_overhead(name, model, x_test, y_test):
    """Measure overhead for a single network architecture."""
    print(f"\n{'=' * 60}")
    print(f"ANALYZING: {name}")
    print(f"{'=' * 60}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    sample_input = torch.randn(1, 2)

    # Measure baseline inference time
    print("\nMeasuring baseline inference time...")
    baseline_metrics = measure_inference_time(
        model=model, input_data=sample_input, num_iterations=100
    )
    print(f"Baseline inference: {baseline_metrics['avg_time_ms']:.3f} ms per inference")

    # Create SEU injector
    print("\nInitializing SEU injector...")
    injector = SEUInjector(
        trained_model=model, criterion=classification_accuracy, x=x_test, y=y_test
    )
    print(f"Baseline accuracy: {injector.baseline_score:.2%}")

    # Calculate overhead with stochastic sampling
    print("\nCalculating SEU injection overhead (using 1% stochastic sampling)...")
    overhead_results = calculate_overhead(
        model=model,
        injector=injector,
        input_data=sample_input,
        bit_position=0,
        num_baseline_iterations=100,
        stochastic=True,
        stochastic_probability=0.01,
    )

    report = format_overhead_report(overhead_results)
    print("\n" + report)

    return {
        "name": name,
        "num_params": num_params,
        "baseline_ms": baseline_metrics["avg_time_ms"],
        "overhead_ms": overhead_results["overhead_absolute_ms"],
        "overhead_percent": overhead_results["overhead_relative"],
        "num_injections": overhead_results["injection"]["num_injections"],
        "baseline_accuracy": injector.baseline_score,
        "report": report,
    }


def save_results_json(results, filename="overhead_results.json"):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "framework_version": "1.1.9",
        "networks": results,
        "summary": {
            "avg_overhead_percent": sum(r["overhead_percent"] for r in results)
            / len(results),
            "total_injections": sum(r["num_injections"] for r in results),
        },
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {filename}")


def save_results_csv(results, filename="overhead_results.csv"):
    """Save results to CSV file."""
    if not results:
        return

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {filename}")


def save_report_text(report_text, filename="overhead_report.txt"):
    """Save formatted report to text file."""
    with open(filename, "w") as f:
        f.write(report_text)
    print(f"Report saved to {filename}")


def main():
    """Main overhead analysis pipeline."""
    print("=" * 60)
    print("SEU INJECTION OVERHEAD CALCULATION")
    print("=" * 60)
    print("\nMeasuring performance overhead of SEU injection operations.\n")

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # Define network architectures using layer sizes
    network_configs = [
        ("Small MLP (2→32→16→1)", [2, 32, 16, 1]),
        ("Medium MLP (2→64→32→16→1)", [2, 64, 32, 16, 1]),
        ("Large MLP (2→128→64→32→16→1)", [2, 128, 64, 32, 16, 1]),
    ]

    results = []

    for name, layer_sizes in network_configs:
        print(f"\n\nPreparing {name}...")
        model = create_mlp(layer_sizes)
        model = train_model(model, X_train, y_train)
        result = measure_network_overhead(name, model, X_test, y_test)
        results.append(result)

    # Print summary
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
    print(
        f"\nTotal SEU injections performed: {sum(r['num_injections'] for r in results):,}"
    )

    print("\nINTERPRETATION:")
    print("-" * 60)
    print("The overhead represents the additional time required for SEU injection")
    print("compared to normal inference. This includes parameter backup, bit flipping,")
    print("model evaluation, and parameter restoration.")
    print(
        f"\nFor the tested networks, SEU injection adds approximately {avg_overhead:.0f}% overhead."
    )
    print("This is the price of comprehensive fault tolerance analysis.")

    # Save results
    save_results_json(results)
    save_results_csv(results)

    # Save individual reports for each network
    for r in results:
        if "report" in r:
            safe_name = (
                r["name"]
                .replace("→", "_")
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            save_report_text(r["report"], f"overhead_report_{safe_name}.txt")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

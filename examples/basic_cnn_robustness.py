#!/usr/bin/env python3
"""
Basic CNN Robustness Analysis Example

This example demonstrates basic SEU injection for analyzing CNN robustness
to Single Event Upsets using the SEU Injection Framework.

Research Application:
- CNN robustness analysis for harsh environments
- Bit-level fault tolerance assessment
- Layer vulnerability comparison

Usage:
    python basic_cnn_robustness.py

Output:
    - Baseline model accuracy
    - Sign bit injection results
    - Layer-wise vulnerability comparison
    - Bit position sensitivity analysis
"""

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # Use non-interactive backend for headless execution
import matplotlib.pyplot as plt

# SEU Injection Framework - CORRECT imports
from seu_injection.core import ExhaustiveSEUInjector
from seu_injection.metrics import classification_accuracy


def create_simple_cnn():
    """Create a simple CNN model for demonstration."""
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


def prepare_data():
    """Prepare training and test data."""
    print("Preparing dataset...")

    # Generate moon-shaped data for binary classification
    X, y = make_moons(n_samples=2000, noise=0.3, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    print(f"Dataset ready: {len(X_train)} train, {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test


def train_model(model, x_train, y_train, epochs=100):
    """Train the CNN model."""
    print("Training model...")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    print("Training complete")
    return model


def run_baseline_analysis(model, x_test, y_test):
    """Run baseline analysis without SEU injection."""
    print("\nBASELINE ANALYSIS")
    print("=" * 50)

    # Initialize SEU injector - CORRECT API
    injector = ExhaustiveSEUInjector(
        trained_model=model, criterion=classification_accuracy, x=x_test, y=y_test
    )

    baseline_acc = injector.baseline_score
    print(f"Baseline Accuracy: {baseline_acc:.2%}")

    return injector, baseline_acc


def analyze_sign_bit_vulnerability(injector):
    """Analyze vulnerability to sign bit flips."""
    print("\nSIGN BIT ANALYSIS")
    print("=" * 50)

    # Test sign bit flips (bit 0) - CORRECT API
    print("Running sign bit injection across all parameters...")
    results = injector.run_injector(bit_i=0)

    # Analyze results
    fault_scores = results["criterion_score"]
    baseline = injector.baseline_score

    accuracy_drops = [baseline - score for score in fault_scores]
    critical_faults = sum(1 for drop in accuracy_drops if drop > 0.1)

    print(f"Total injections performed: {len(fault_scores)}")
    print(f"Mean accuracy after injection: {np.mean(fault_scores):.2%}")
    print(f"Worst case accuracy: {min(fault_scores):.2%}")
    print(f"Critical faults (>10% drop): {critical_faults}")
    print(f"Average accuracy drop: {np.mean(accuracy_drops):.2%}")

    return results


def analyze_layer_vulnerability(injector):
    """Compare vulnerability across different layers."""
    print("\nLAYER VULNERABILITY ANALYSIS")
    print("=" * 50)

    # Get layer names from model
    layer_results = {}

    for layer_name, _param in injector.model.named_parameters():
        if "weight" in layer_name:
            print(f"Testing layer: {layer_name}")

            # Test sign bit for this specific layer - CORRECT API
            results = injector.run_injector(bit_i=0, layer_name=layer_name)

            fault_scores = results["criterion_score"]
            avg_accuracy = np.mean(fault_scores)
            accuracy_drop = injector.baseline_score - avg_accuracy

            layer_results[layer_name] = {
                "avg_accuracy": avg_accuracy,
                "accuracy_drop": accuracy_drop,
                "num_injections": len(fault_scores),
            }

            print(f"  Average accuracy: {avg_accuracy:.2%}")
            print(f"  Accuracy drop: {accuracy_drop:.2%}")
            print(f"  Injections: {len(fault_scores)}")

    # Find most vulnerable layer
    most_vulnerable = max(
        layer_results.keys(), key=lambda x: layer_results[x]["accuracy_drop"]
    )

    print(f"\nMost vulnerable layer: {most_vulnerable}")
    print(f"   Accuracy drop: {layer_results[most_vulnerable]['accuracy_drop']:.2%}")

    return layer_results


def analyze_bit_position_sensitivity(injector):
    """Analyze sensitivity to different IEEE 754 bit positions."""
    print("\nBIT POSITION SENSITIVITY")
    print("=" * 50)

    # Test representative bit positions
    bit_positions = [0, 1, 2, 8, 15, 23, 31]  # Sign, exponent, mantissa
    bit_names = [
        "Sign",
        "Exp MSB",
        "Exp",
        "Exp LSB",
        "Mantissa",
        "Mantissa",
        "Mantissa LSB",
    ]

    bit_results = {}

    print("Testing bit positions (this may take a while for larger models)...")

    # Use a StochasticSEUInjector for stochastic sampling
    from seu_injection.core import StochasticSEUInjector

    stochastic_injector = StochasticSEUInjector(
        trained_model=injector.model,
        criterion=injector.criterion,
        x=injector.X,
        y=injector.y,
        device=injector.device,
    )

    for i, bit_pos in enumerate(bit_positions):
        print(f"  Testing bit {bit_pos} ({bit_names[i]})...")

        results = stochastic_injector.run_injector(bit_i=bit_pos, p=0.1)  # 10% sampling

        if len(results["criterion_score"]) > 0:
            fault_scores = results["criterion_score"]
            avg_accuracy = np.mean(fault_scores)
            accuracy_drop = injector.baseline_score - avg_accuracy

            bit_results[bit_pos] = {
                "name": bit_names[i],
                "avg_accuracy": avg_accuracy,
                "accuracy_drop": accuracy_drop,
                "num_injections": len(fault_scores),
            }

            print(
                f"    Average accuracy: {avg_accuracy:.2%} (drop: {accuracy_drop:.2%})"
            )
        else:
            print(f"    No injections sampled for bit {bit_pos}")

    return bit_results


def create_visualizations(baseline_acc, layer_results, bit_results):
    """Create visualization plots."""
    print("\nCREATING VISUALIZATIONS")
    print("=" * 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Layer vulnerability plot
    if layer_results:
        layers = list(layer_results.keys())
        drops = [layer_results[layer]["accuracy_drop"] * 100 for layer in layers]

        ax1.bar(range(len(layers)), drops, color="red", alpha=0.7)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Accuracy Drop (%)")
        ax1.set_title("Layer Vulnerability to Sign Bit Flips")
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels([f"Layer {i}" for i in range(len(layers))], rotation=45)
        ax1.grid(True, alpha=0.3)

    # Bit position sensitivity plot
    if bit_results:
        positions = list(bit_results.keys())
        drops = [bit_results[pos]["accuracy_drop"] * 100 for pos in positions]
        names = [bit_results[pos]["name"] for pos in positions]

        colors = [
            "red" if pos == 0 else "orange" if pos <= 8 else "blue" for pos in positions
        ]
        ax2.bar(range(len(positions)), drops, color=colors, alpha=0.7)
        ax2.set_xlabel("Bit Position")
        ax2.set_ylabel("Average Accuracy Drop (%)")
        ax2.set_title("IEEE 754 Bit Position Vulnerability")
        ax2.set_xticks(range(len(positions)))
        ax2.set_xticklabels(
            [f"{pos}\n({names[i]})" for i, pos in enumerate(positions)], rotation=45
        )
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cnn_robustness_analysis.png", dpi=300, bbox_inches="tight")
    print("Visualizations saved as 'cnn_robustness_analysis.png'")
    plt.show()


def main():
    """Main analysis pipeline."""
    print("CNN ROBUSTNESS ANALYSIS - SEU INJECTION FRAMEWORK")
    print("=" * 60)

    try:
        # 1. Prepare data and model
        x_train, x_test, y_train, y_test = prepare_data()

        # 2. Create and train model
        model = create_simple_cnn()
        model = train_model(model, x_train, y_train)

        # 3. Initialize SEU injection analysis
        injector, baseline_acc = run_baseline_analysis(model, x_test, y_test)

        # 4. Run SEU injection analyses
        analyze_sign_bit_vulnerability(injector)
        layer_results = analyze_layer_vulnerability(injector)
        bit_results = analyze_bit_position_sensitivity(injector)

        # 5. Create visualizations
        create_visualizations(baseline_acc, layer_results, bit_results)

        # 6. Summary
        print("\nANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Baseline accuracy: {baseline_acc:.2%}")
        print(f"Layers analyzed: {len(layer_results)}")
        print(f"Bit positions tested: {len(bit_results)}")
        print("Check 'cnn_robustness_analysis.png' for visualizations!")

    except Exception as e:
        print(f"ERROR during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()

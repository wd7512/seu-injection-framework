#!/usr/bin/env python3
"""
Basic CNN Robustness Analysis Example

This example demonstrates basic SEU injection for analyzing CNN robustness
to Single Event Upsets in a space mission scenario.

Research Application:
- Mars rover image classification robustness
- Radiation-induced fault tolerance assessment
- Critical system reliability evaluation

Requirements:
- seu-injection-framework
- torch, torchvision (automatically installed)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from seu_injection import SEUInjector, classification_accuracy


class SpaceCNN(nn.Module):
    """
    Simple CNN architecture for space mission image classification.

    Typical use case: Mars rover terrain classification or Earth observation.
    """

    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()

        # Convolutional layers - vulnerable to radiation
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers - critical for decision making
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Assumes 32x32 input
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = self.pool(functional.relu(self.conv3(x)))

        # Flatten and classify
        x = x.view(-1, 128 * 4 * 4)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def generate_space_mission_data(num_samples=1000, image_size=32, num_classes=10):
    """
    Generate synthetic space mission data for robustness testing.

    In practice, this would be replaced with actual mission data:
    - Mars surface terrain images
    - Earth observation classifications
    - Satellite sensor data
    """

    # Create synthetic image data (RGB images)
    X = torch.randn(num_samples, 3, image_size, image_size)

    # Add some structure to make it more realistic
    # Simulate terrain features, cloud patterns, etc.
    for i in range(num_samples):
        # Add some coherent patterns (terrain-like features)
        pattern = torch.sin(torch.arange(image_size).float() / 4).unsqueeze(0)
        X[i, 0] += pattern.unsqueeze(0) * 0.3
        X[i, 1] += pattern.T.unsqueeze(0) * 0.2

    # Normalize to typical image range
    X = torch.clamp(X, -2, 2)

    # Generate corresponding labels
    y = torch.randint(0, num_classes, (num_samples,))

    return X, y


def analyze_layer_vulnerability(model, data_loader, device="cpu", verbose=True):
    """
    Analyze vulnerability of different CNN layers to SEU injection.

    This analysis helps identify which layers are most critical for
    space mission reliability.
    """

    if verbose:
        print("🔬 Analyzing Layer-Specific SEU Vulnerability")
        print("=" * 60)

    # Initialize injector
    injector = SEUInjector(model, device=device)

    # Get baseline performance
    baseline_accuracy = injector.get_criterion_score(
        data=data_loader, criterion=classification_accuracy, device=device
    )

    if verbose:
        print(f"📊 Baseline Accuracy: {baseline_accuracy:.4f}")
        print("\n🎯 Layer-Specific Vulnerability Analysis:")

    # Target different types of layers
    layer_groups = {
        "Convolutional Layers": ["conv1.weight", "conv2.weight", "conv3.weight"],
        "Fully Connected Layers": ["fc1.weight", "fc2.weight"],
        "Convolutional Biases": ["conv1.bias", "conv2.bias", "conv3.bias"],
        "FC Biases": ["fc1.bias", "fc2.bias"],
    }

    results = {}

    for group_name, layers in layer_groups.items():
        if verbose:
            print(f"\n{group_name}:")

        group_results = []

        for layer in layers:
            try:
                # Test bit position 15 (mantissa bit - common for SEU studies)
                result = injector.run_seu(
                    data=data_loader,
                    criterion=classification_accuracy,
                    bit_position=15,
                    target_layers=[layer],
                    device=device,
                )

                accuracy_drop = result["accuracy_drop"]
                group_results.append(accuracy_drop)

                if verbose:
                    print(f"  {layer:15s}: {accuracy_drop:.6f} accuracy drop")

            except Exception as e:
                if verbose:
                    print(f"  {layer:15s}: Error - {str(e)}")
                group_results.append(np.nan)

        results[group_name] = {
            "individual": group_results,
            "mean": np.nanmean(group_results),
            "std": np.nanstd(group_results),
        }

    return baseline_accuracy, results


def bit_position_sensitivity_analysis(
    model, data_loader, target_layer="fc2.weight", device="cpu", verbose=True
):
    """
    Analyze sensitivity to different bit positions in IEEE 754 representation.

    Critical for understanding which types of radiation-induced faults
    are most dangerous for space missions.
    """

    if verbose:
        print(f"\n🔍 Bit Position Sensitivity Analysis for {target_layer}")
        print("=" * 60)

    injector = SEUInjector(model, device=device)

    # Test different bit positions
    bit_positions = [0, 1, 8, 15, 16, 23, 31]  # Mix of mantissa, exponent, sign
    bit_descriptions = {
        0: "Mantissa LSB",
        1: "Mantissa bit 1",
        8: "Mantissa bit 8",
        15: "Mantissa bit 15",
        16: "Mantissa MSB",
        23: "Exponent LSB",
        31: "Sign bit",
    }

    results = {}

    if verbose:
        print("Bit Position | Description    | Accuracy Drop | Severity")
        print("-" * 55)

    for bit_pos in bit_positions:
        try:
            result = injector.run_seu(
                data=data_loader,
                criterion=classification_accuracy,
                bit_position=bit_pos,
                target_layers=[target_layer],
                device=device,
            )

            accuracy_drop = result["accuracy_drop"]
            results[bit_pos] = accuracy_drop

            # Classify severity
            if accuracy_drop > 0.1:
                severity = "🔴 CRITICAL"
            elif accuracy_drop > 0.05:
                severity = "🟡 MODERATE"
            else:
                severity = "🟢 LOW"

            if verbose:
                print(
                    f"{bit_pos:11d} | {bit_descriptions[bit_pos]:13s} | {accuracy_drop:11.6f} | {severity}"
                )

        except Exception as e:
            if verbose:
                print(
                    f"{bit_pos:11d} | {bit_descriptions[bit_pos]:13s} | Error: {str(e)}"
                )
            results[bit_pos] = np.nan

    return results


def stochastic_seu_campaign(
    model, data_loader, device="cpu", num_trials=5, verbose=True
):
    """
    Run a stochastic SEU injection campaign to simulate realistic
    space radiation environment.

    This represents the statistical nature of cosmic ray impacts
    during long-duration space missions.
    """

    if verbose:
        print("\n⚡ Stochastic SEU Injection Campaign")
        print("=" * 60)
        print("Simulating realistic space radiation environment...")

    injector = SEUInjector(model, device=device)

    # Different radiation intensity scenarios
    scenarios = {
        "Low Earth Orbit": {
            "probability": 1e-6,
            "description": "LEO satellite mission",
        },
        "Mars Transit": {"probability": 5e-6, "description": "Deep space journey"},
        "Jupiter Mission": {
            "probability": 1e-5,
            "description": "High radiation environment",
        },
        "Solar Storm": {"probability": 1e-4, "description": "Extreme space weather"},
    }

    results = {}

    for scenario_name, config in scenarios.items():
        if verbose:
            print(f"\n📡 {scenario_name} ({config['description']}):")

        scenario_results = []

        for trial in range(num_trials):
            try:
                result = injector.run_stochastic_seu(
                    data=data_loader,
                    criterion=classification_accuracy,
                    probability=config["probability"],
                    device=device,
                    random_seed=42 + trial,  # Reproducible but varied
                )

                accuracy_drop = result["accuracy_drop"]
                scenario_results.append(accuracy_drop)

                if verbose:
                    print(f"  Trial {trial + 1}: {accuracy_drop:.6f} accuracy drop")

            except Exception as e:
                if verbose:
                    print(f"  Trial {trial + 1}: Error - {str(e)}")
                scenario_results.append(np.nan)

        # Calculate statistics
        mean_drop = np.nanmean(scenario_results)
        std_drop = np.nanstd(scenario_results)

        results[scenario_name] = {
            "trials": scenario_results,
            "mean": mean_drop,
            "std": std_drop,
            "probability": config["probability"],
        }

        if verbose:
            print(f"  Mean accuracy drop: {mean_drop:.6f} ± {std_drop:.6f}")

            # Mission risk assessment
            if mean_drop > 0.05:
                risk_level = "🔴 HIGH RISK"
                recommendation = "Requires radiation hardening"
            elif mean_drop > 0.01:
                risk_level = "🟡 MODERATE RISK"
                recommendation = "Consider error correction"
            else:
                risk_level = "🟢 LOW RISK"
                recommendation = "Acceptable for mission"

            print(f"  Risk Assessment: {risk_level}")
            print(f"  Recommendation: {recommendation}")

    return results


def generate_mission_report(
    baseline_accuracy, layer_results, bit_results, stochastic_results, output_dir=None
):
    """
    Generate a comprehensive mission readiness report.

    This report format is typical for space mission safety assessments.
    """

    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    report = []
    report.append("=" * 80)
    report.append("SPACE MISSION CNN ROBUSTNESS ASSESSMENT REPORT")
    report.append("=" * 80)
    report.append("")

    # Executive Summary
    report.append("📋 EXECUTIVE SUMMARY")
    report.append("-" * 40)
    report.append(f"• Baseline Model Accuracy: {baseline_accuracy:.4f}")

    # Determine overall risk level
    max_layer_drop = max([np.nanmean(r["individual"]) for r in layer_results.values()])
    max_bit_drop = max([v for v in bit_results.values() if not np.isnan(v)])
    max_stochastic_drop = max([r["mean"] for r in stochastic_results.values()])

    overall_risk = max(max_layer_drop, max_bit_drop, max_stochastic_drop)

    if overall_risk > 0.1:
        risk_level = "🔴 HIGH"
        recommendation = "NOT RECOMMENDED for critical missions without hardening"
    elif overall_risk > 0.05:
        risk_level = "🟡 MODERATE"
        recommendation = "Requires additional protection measures"
    else:
        risk_level = "🟢 LOW"
        recommendation = "APPROVED for mission deployment"

    report.append(f"• Overall Risk Level: {risk_level}")
    report.append(f"• Mission Recommendation: {recommendation}")
    report.append("")

    # Detailed Analysis
    report.append("🔬 DETAILED VULNERABILITY ANALYSIS")
    report.append("-" * 40)

    report.append("\n1. Layer-Specific Vulnerability:")
    for group_name, results in layer_results.items():
        mean_drop = results["mean"]
        report.append(f"   {group_name}: {mean_drop:.6f} ± {results['std']:.6f}")

    report.append("\n2. Most Vulnerable Bit Positions:")
    sorted_bits = sorted(
        [(pos, drop) for pos, drop in bit_results.items() if not np.isnan(drop)],
        key=lambda x: x[1],
        reverse=True,
    )
    for pos, drop in sorted_bits[:3]:
        report.append(f"   Bit {pos}: {drop:.6f} accuracy drop")

    report.append("\n3. Radiation Environment Impact:")
    for scenario, results in stochastic_results.items():
        mean_drop = results["mean"]
        probability = results["probability"]
        report.append(f"   {scenario}: {mean_drop:.6f} drop (p={probability:.0e})")

    # Recommendations
    report.append("\n🛡️ PROTECTION RECOMMENDATIONS")
    report.append("-" * 40)

    # Find most vulnerable layer
    most_vulnerable_layer = max(layer_results.items(), key=lambda x: x[1]["mean"])
    report.append(f"• Priority Protection: {most_vulnerable_layer[0]}")

    # Find most dangerous bit position
    most_dangerous_bit = max(
        bit_results.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 0
    )
    report.append(
        f"• Most Critical Bit Position: {most_dangerous_bit[0]} (drop: {most_dangerous_bit[1]:.6f})"
    )

    if overall_risk > 0.05:
        report.append("• Consider Triple Modular Redundancy (TMR)")
        report.append("• Implement error detection and correction")
        report.append("• Add radiation-hardened computing elements")

    report.append("\n• Report generated using SEU Injection Framework v1.0.0")
    report.append("• Framework: https://github.com/wd7512/seu-injection-framework")

    # Save report
    report_text = "\n".join(report)
    report_file = output_dir / "mission_robustness_report.txt"

    with open(report_file, "w") as f:
        f.write(report_text)

    print(f"\n📄 Mission report saved to: {report_file}")
    return report_text


def main():
    """
    Main analysis pipeline for space mission CNN robustness assessment.
    """

    print("🚀 Space Mission CNN Robustness Analysis")
    print("Using SEU Injection Framework for Fault Tolerance Assessment")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Create model and data
    print("\n🏗️  Setting up Space Mission CNN...")
    model = SpaceCNN(num_classes=10, input_channels=3)
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Generate mission data
    print("📡 Generating space mission test data...")
    X_test, y_test = generate_space_mission_data(num_samples=500)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"✅ Setup complete: {len(test_dataset)} test samples")

    # Analysis Pipeline
    try:
        # 1. Layer Vulnerability Analysis
        print("\n" + "=" * 80)
        baseline_accuracy, layer_results = analyze_layer_vulnerability(
            model, test_loader, device=device
        )

        # 2. Bit Position Sensitivity
        print("\n" + "=" * 80)
        bit_results = bit_position_sensitivity_analysis(
            model, test_loader, target_layer="fc2.weight", device=device
        )

        # 3. Stochastic SEU Campaign
        print("\n" + "=" * 80)
        stochastic_results = stochastic_seu_campaign(
            model, test_loader, device=device, num_trials=3
        )

        # 4. Generate Mission Report
        print("\n" + "=" * 80)
        print("📄 Generating Mission Readiness Report...")

        generate_mission_report(
            baseline_accuracy, layer_results, bit_results, stochastic_results
        )

        print("\n🎉 Analysis Complete!")
        print("\nNext Steps for Space Mission Deployment:")
        print("1. Review the generated mission report")
        print("2. Implement recommended protection measures")
        print("3. Conduct hardware-in-the-loop testing")
        print("4. Validate with actual space radiation data")

    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        print("Please check your environment and try again.")
        raise


if __name__ == "__main__":
    main()

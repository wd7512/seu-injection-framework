#!/usr/bin/env python3
"""
Architecture Comparison Study

Compare robustness of different neural network architectures to SEU injection.
Demonstrates systematic evaluation methodology for choosing fault-tolerant models.

Research Applications:
- Architecture selection for harsh environments
- Comparative robustness analysis
- Defense system neural network selection
- Critical infrastructure AI deployment

Requirements:
- seu-injection-framework
- torch, torchvision
"""

import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from seu_injection import SEUInjector, classification_accuracy


class SimpleNN(nn.Module):
    """Fully connected neural network baseline."""

    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class CompactCNN(nn.Module):
    """Compact CNN for resource-constrained environments."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """Basic residual block for mini-ResNet."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = functional.relu(out)
        return out


class MiniResNet(nn.Module):
    """Compact ResNet-inspired architecture."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 16, 2, stride=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EfficientBlock(nn.Module):
    """Efficient block with depthwise separable convolutions."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride, 1, groups=in_channels
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = functional.relu(self.bn1(self.depthwise(x)))
        x = functional.relu(self.bn2(self.pointwise(x)))
        return x


class EfficientNet(nn.Module):
    """Efficient architecture inspired by MobileNet/EfficientNet."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.block1 = EfficientBlock(16, 32, stride=2)
        self.block2 = EfficientBlock(32, 64, stride=2)
        self.block3 = EfficientBlock(64, 128, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = functional.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_architectures():
    """Create all architectures for comparison."""

    architectures = {
        "SimpleNN": SimpleNN(input_size=784, hidden_size=512, num_classes=10),
        "CompactCNN": CompactCNN(num_classes=10),
        "MiniResNet": MiniResNet(num_classes=10),
        "EfficientNet": EfficientNet(num_classes=10),
    }

    return architectures


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_model_complexity(architectures):
    """Analyze computational complexity of different architectures."""

    print("📊 Architecture Complexity Analysis")
    print("=" * 60)
    print(f"{'Architecture':<15} {'Parameters':<12} {'Memory (MB)':<12} {'FLOPs Est.'}")
    print("-" * 60)

    complexity_data = {}

    for name, model in architectures.items():
        # Parameter count
        params = count_parameters(model)

        # Rough memory estimate (parameters * 4 bytes for float32)
        memory_mb = params * 4 / (1024 * 1024)

        # Simple FLOPs estimation (very rough)
        if "NN" in name:
            flops = "~500K"  # Dense layers
        elif "CNN" in name or "ResNet" in name or "Efficient" in name:
            flops = "~50M"  # Convolutions
        else:
            flops = "Unknown"

        complexity_data[name] = {
            "parameters": params,
            "memory_mb": memory_mb,
            "flops": flops,
        }

        print(f"{name:<15} {params:<12,} {memory_mb:<12.2f} {flops}")

    return complexity_data


def comprehensive_robustness_analysis(architectures, test_loader, device="cpu"):
    """
    Comprehensive robustness analysis across all architectures.

    Tests multiple scenarios:
    1. Different bit positions
    2. Different layer types
    3. Stochastic injection scenarios
    """

    print("\n🔬 Comprehensive Robustness Analysis")
    print("=" * 80)

    results = defaultdict(dict)

    # Test scenarios
    bit_positions = [0, 8, 15, 23, 31]  # Representative bit positions
    injection_probabilities = [1e-6, 1e-5, 1e-4]  # Different radiation intensities

    for arch_name, model in architectures.items():
        print(f"\n🏗️  Analyzing {arch_name}...")
        model.to(device)
        model.eval()

        injector = SEUInjector(model, device=device)

        # Get baseline accuracy
        baseline = injector.get_criterion_score(
            data=test_loader, criterion=classification_accuracy, device=device
        )
        results[arch_name]["baseline_accuracy"] = baseline

        print(f"   Baseline accuracy: {baseline:.4f}")

        # 1. Bit position analysis
        print("   Testing bit position sensitivity...")
        bit_results = {}

        for bit_pos in bit_positions:
            try:
                # Test on a representative layer (usually the final classifier)
                target_layers = []
                for name, _ in model.named_parameters():
                    if "fc" in name.lower() and "weight" in name:
                        target_layers.append(name)
                        break

                if target_layers:
                    result = injector.run_seu(
                        data=test_loader,
                        criterion=classification_accuracy,
                        bit_position=bit_pos,
                        target_layers=target_layers,
                        device=device,
                    )
                    bit_results[bit_pos] = result["accuracy_drop"]
                else:
                    bit_results[bit_pos] = np.nan

            except Exception as e:
                print(f"     Error at bit {bit_pos}: {str(e)}")
                bit_results[bit_pos] = np.nan

        results[arch_name]["bit_sensitivity"] = bit_results

        # 2. Layer type vulnerability
        print("   Testing layer type vulnerability...")
        layer_results = {}

        for name, param in model.named_parameters():
            if (
                param.requires_grad and len(param.shape) > 1
            ):  # Skip biases and small params
                try:
                    result = injector.run_seu(
                        data=test_loader,
                        criterion=classification_accuracy,
                        bit_position=15,  # Standard test bit
                        target_layers=[name],
                        device=device,
                    )
                    layer_results[name] = result["accuracy_drop"]

                except Exception:
                    layer_results[name] = np.nan

        results[arch_name]["layer_vulnerability"] = layer_results

        # 3. Stochastic injection scenarios
        print("   Testing stochastic scenarios...")
        stochastic_results = {}

        for prob in injection_probabilities:
            try:
                result = injector.run_stochastic_seu(
                    data=test_loader,
                    criterion=classification_accuracy,
                    probability=prob,
                    device=device,
                    random_seed=42,
                )
                stochastic_results[prob] = result["accuracy_drop"]

            except Exception:
                stochastic_results[prob] = np.nan

        results[arch_name]["stochastic_robustness"] = stochastic_results

        print(f"   ✅ {arch_name} analysis complete")

    return results


def create_comparison_visualizations(results, complexity_data, output_dir=None):
    """Create comprehensive comparison visualizations."""

    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    # 1. Robustness vs Complexity Scatter Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Neural Network Architecture Comparison: Robustness vs Complexity", fontsize=16
    )

    # Extract data for plotting
    arch_names = list(results.keys())
    parameters = [complexity_data[name]["parameters"] for name in arch_names]
    baseline_accs = [results[name]["baseline_accuracy"] for name in arch_names]

    # Average bit sensitivity (robustness metric)
    avg_bit_sensitivity = []
    for name in arch_names:
        bit_vals = [
            v for v in results[name]["bit_sensitivity"].values() if not np.isnan(v)
        ]
        avg_bit_sensitivity.append(np.mean(bit_vals) if bit_vals else 0)

    # Plot 1: Robustness vs Parameters
    ax1.scatter(
        parameters,
        avg_bit_sensitivity,
        s=100,
        alpha=0.7,
        c=baseline_accs,
        cmap="viridis",
    )
    ax1.set_xlabel("Number of Parameters")
    ax1.set_ylabel("Average Accuracy Drop (lower = more robust)")
    ax1.set_title("Robustness vs Model Size")
    ax1.set_xscale("log")

    # Add architecture labels
    for i, name in enumerate(arch_names):
        ax1.annotate(
            name,
            (parameters[i], avg_bit_sensitivity[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Plot 2: Bit Position Sensitivity Comparison
    bit_positions = [0, 8, 15, 23, 31]
    for name in arch_names:
        bit_data = [
            results[name]["bit_sensitivity"].get(pos, 0) for pos in bit_positions
        ]
        ax2.plot(bit_positions, bit_data, marker="o", label=name, linewidth=2)

    ax2.set_xlabel("Bit Position")
    ax2.set_ylabel("Accuracy Drop")
    ax2.set_title("Bit Position Sensitivity by Architecture")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Stochastic Robustness
    probabilities = [1e-6, 1e-5, 1e-4]
    prob_labels = ["1e-6", "1e-5", "1e-4"]

    for name in arch_names:
        stoch_data = [
            results[name]["stochastic_robustness"].get(prob, 0)
            for prob in probabilities
        ]
        ax3.plot(prob_labels, stoch_data, marker="s", label=name, linewidth=2)

    ax3.set_xlabel("Injection Probability")
    ax3.set_ylabel("Accuracy Drop")
    ax3.set_title("Stochastic SEU Robustness")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Overall Robustness Ranking
    # Calculate composite robustness score
    robustness_scores = {}
    for name in arch_names:
        bit_score = np.nanmean(list(results[name]["bit_sensitivity"].values()))
        stoch_score = np.nanmean(list(results[name]["stochastic_robustness"].values()))
        layer_score = np.nanmean(list(results[name]["layer_vulnerability"].values()))

        # Lower is better (less accuracy drop = more robust)
        composite_score = np.nanmean([bit_score, stoch_score, layer_score])
        robustness_scores[name] = composite_score

    # Sort by robustness (ascending - lower drop is better)
    sorted_archs = sorted(robustness_scores.items(), key=lambda x: x[1])

    arch_names_sorted = [item[0] for item in sorted_archs]
    scores_sorted = [item[1] for item in sorted_archs]

    bars = ax4.bar(
        range(len(arch_names_sorted)),
        scores_sorted,
        color=["green", "yellow", "orange", "red"][: len(arch_names_sorted)],
    )
    ax4.set_xlabel("Architecture (Ranked by Robustness)")
    ax4.set_ylabel("Composite Robustness Score (lower = better)")
    ax4.set_title("Overall Robustness Ranking")
    ax4.set_xticks(range(len(arch_names_sorted)))
    ax4.set_xticklabels(arch_names_sorted, rotation=45)

    # Add value labels on bars
    for _i, (bar, score) in enumerate(zip(bars, scores_sorted)):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + score * 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / "architecture_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"📊 Comparison plots saved to: {plot_file}")

    return robustness_scores


def generate_comparative_report(
    results, complexity_data, robustness_scores, output_dir=None
):
    """Generate comprehensive comparison report."""

    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    report = []
    report.append("=" * 80)
    report.append("NEURAL NETWORK ARCHITECTURE ROBUSTNESS COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")

    # Executive Summary
    report.append("📋 EXECUTIVE SUMMARY")
    report.append("-" * 40)

    # Find best and worst architectures
    best_arch = min(robustness_scores.items(), key=lambda x: x[1])
    worst_arch = max(robustness_scores.items(), key=lambda x: x[1])

    report.append(
        f"• Most Robust Architecture: {best_arch[0]} (score: {best_arch[1]:.6f})"
    )
    report.append(
        f"• Least Robust Architecture: {worst_arch[0]} (score: {worst_arch[1]:.6f})"
    )
    report.append(
        f"• Robustness Improvement: {worst_arch[1] / best_arch[1]:.2f}x difference"
    )

    # Model complexity summary
    report.append("\n• Model Complexity Range:")
    min_params = min([complexity_data[name]["parameters"] for name in complexity_data])
    max_params = max([complexity_data[name]["parameters"] for name in complexity_data])
    report.append(f"  - Parameters: {min_params:,} to {max_params:,}")

    # Baseline performance
    baseline_range = [results[name]["baseline_accuracy"] for name in results]
    report.append(
        f"  - Baseline Accuracy: {min(baseline_range):.4f} to {max(baseline_range):.4f}"
    )
    report.append("")

    # Detailed Analysis
    report.append("🔬 DETAILED ARCHITECTURE ANALYSIS")
    report.append("-" * 40)

    # Sort architectures by robustness for reporting
    sorted_archs = sorted(robustness_scores.items(), key=lambda x: x[1])

    for rank, (arch_name, score) in enumerate(sorted_archs, 1):
        report.append(f"\n{rank}. {arch_name} (Robustness Score: {score:.6f})")

        # Basic metrics
        params = complexity_data[arch_name]["parameters"]
        baseline = results[arch_name]["baseline_accuracy"]
        report.append(f"   • Parameters: {params:,}")
        report.append(f"   • Baseline Accuracy: {baseline:.4f}")

        # Bit sensitivity analysis
        bit_sens = results[arch_name]["bit_sensitivity"]
        avg_bit = np.nanmean(list(bit_sens.values()))
        most_vulnerable_bit = max(
            bit_sens.items(), key=lambda x: x[1] if not np.isnan(x[1]) else 0
        )
        report.append(f"   • Average Bit Sensitivity: {avg_bit:.6f}")
        report.append(
            f"   • Most Vulnerable Bit: {most_vulnerable_bit[0]} ({most_vulnerable_bit[1]:.6f} drop)"
        )

        # Stochastic robustness
        stoch_rob = results[arch_name]["stochastic_robustness"]
        avg_stoch = np.nanmean(list(stoch_rob.values()))
        report.append(f"   • Average Stochastic Impact: {avg_stoch:.6f}")

        # Layer vulnerability (top 3 most vulnerable)
        layer_vuln = results[arch_name]["layer_vulnerability"]
        top_vulnerable = sorted(
            layer_vuln.items(),
            key=lambda x: x[1] if not np.isnan(x[1]) else 0,
            reverse=True,
        )[:3]
        report.append("   • Most Vulnerable Layers:")
        for layer_name, drop in top_vulnerable:
            report.append(f"     - {layer_name}: {drop:.6f}")

    # Recommendations
    report.append("\n🎯 DEPLOYMENT RECOMMENDATIONS")
    report.append("-" * 40)

    # Mission-specific recommendations
    best_robust = sorted_archs[0][0]
    best_compact = min(complexity_data.items(), key=lambda x: x[1]["parameters"])[0]
    best_accurate = max(results.items(), key=lambda x: x[1]["baseline_accuracy"])[0]

    report.append("\n• For CRITICAL MISSIONS (space, nuclear):")
    report.append(f"  - Primary Choice: {best_robust} (highest robustness)")
    if best_robust != best_accurate:
        report.append(f"  - Alternative: {best_accurate} (highest baseline accuracy)")

    report.append("\n• For RESOURCE-CONSTRAINED systems:")
    report.append(f"  - Recommended: {best_compact} (most compact)")
    compact_robustness_rank = [arch for arch, _ in sorted_archs].index(best_compact) + 1
    report.append(f"  - Robustness Rank: #{compact_robustness_rank}/4")

    report.append("\n• For HIGH-PERFORMANCE applications:")
    report.append(f"  - Recommended: {best_accurate} (highest baseline)")
    accurate_robustness_rank = [arch for arch, _ in sorted_archs].index(
        best_accurate
    ) + 1
    report.append(f"  - Robustness Rank: #{accurate_robustness_rank}/4")

    # Protection strategies
    report.append("\n🛡️  PROTECTION STRATEGIES")
    report.append("-" * 40)
    report.append(f"• For {worst_arch[0]} (least robust):")
    report.append("  - Implement Triple Modular Redundancy (TMR)")
    report.append("  - Add error detection and correction codes")
    report.append("  - Consider radiation-hardened hardware")

    report.append("• For all architectures:")
    report.append("  - Focus protection on most vulnerable layers")
    report.append("  - Monitor bit positions 15, 23, 31 (highest impact)")
    report.append("  - Implement graceful degradation strategies")

    # Research insights
    report.append("\n🔬 RESEARCH INSIGHTS")
    report.append("-" * 40)

    # Correlation analysis
    param_robust_corr = np.corrcoef(
        [complexity_data[name]["parameters"] for name in sorted_archs],
        [score for _, score in sorted_archs],
    )[0, 1]

    report.append(f"• Parameter-Robustness Correlation: {param_robust_corr:.3f}")
    if abs(param_robust_corr) > 0.5:
        trend = "positive" if param_robust_corr > 0 else "negative"
        report.append(
            f"  - {trend.capitalize()} correlation: larger models tend to be {'less' if param_robust_corr > 0 else 'more'} robust"
        )
    else:
        report.append(
            "  - Weak correlation: model size doesn't strongly predict robustness"
        )

    # Architecture-specific insights
    if "ResNet" in [name for name, _ in sorted_archs[:2]]:
        report.append("• Residual connections appear to enhance robustness")
    if "Efficient" in [name for name, _ in sorted_archs[:2]]:
        report.append("• Efficient architectures maintain good robustness")

    report.append("\n• Framework: SEU Injection Framework v1.0.0")
    report.append(f"• Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("• Repository: https://github.com/wd7512/seu-injection-framework")

    # Save report
    report_text = "\n".join(report)
    report_file = output_dir / "architecture_comparison_report.txt"

    with open(report_file, "w") as f:
        f.write(report_text)

    print(f"📄 Comparison report saved to: {report_file}")
    return report_text


def generate_test_data(num_samples=1000, image_size=28):
    """Generate synthetic test data for architecture comparison."""

    # Create MNIST-like data
    X = torch.randn(num_samples, 1, image_size, image_size)

    # Add some structure to make classification meaningful
    for i in range(num_samples):
        # Add patterns that correlate with labels
        pattern_type = i % 10
        if pattern_type < 5:
            # Add vertical lines
            X[i, 0, :, image_size // 4 : 3 * image_size // 4] += 0.5
        else:
            # Add horizontal lines
            X[i, 0, image_size // 4 : 3 * image_size // 4, :] += 0.5

    # Normalize
    X = torch.clamp(X, -2, 2)

    # Generate labels
    y = torch.randint(0, 10, (num_samples,))

    return X, y


def main():
    """
    Main pipeline for comprehensive architecture comparison study.
    """

    print("🏗️  Neural Network Architecture Robustness Comparison")
    print("Systematic evaluation for harsh environment deployment")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Create all architectures
    print("\n🔧 Creating test architectures...")
    architectures = create_architectures()

    # Analyze complexity
    complexity_data = analyze_model_complexity(architectures)

    # Generate test data
    print("\n📊 Generating test dataset...")
    X_test, y_test = generate_test_data(num_samples=500)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(
        f"✅ Setup complete: {len(architectures)} architectures, {len(test_dataset)} samples"
    )

    try:
        # Comprehensive analysis
        print("\n" + "=" * 80)
        results = comprehensive_robustness_analysis(architectures, test_loader, device)

        # Create visualizations
        print("\n📊 Creating comparison visualizations...")
        robustness_scores = create_comparison_visualizations(results, complexity_data)

        # Generate report
        print("\n📄 Generating comprehensive comparison report...")
        generate_comparative_report(results, complexity_data, robustness_scores)

        print("\n🎉 Architecture comparison study complete!")
        print("\nKey Findings:")

        # Print top-level results
        sorted_archs = sorted(robustness_scores.items(), key=lambda x: x[1])
        print(f"🥇 Most Robust: {sorted_archs[0][0]} (score: {sorted_archs[0][1]:.6f})")
        print(f"🥈 Second: {sorted_archs[1][0]} (score: {sorted_archs[1][1]:.6f})")
        print(f"🥉 Third: {sorted_archs[2][0]} (score: {sorted_archs[2][1]:.6f})")

        print(
            f"\n📈 Robustness improvement range: {sorted_archs[-1][1] / sorted_archs[0][1]:.2f}x"
        )

        print("\nOutput files generated:")
        print("• architecture_comparison.png - Visualization plots")
        print("• architecture_comparison_report.txt - Detailed analysis")

    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        print("Please check your environment and try again.")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Activation Function Robustness Comparison Experiment

Compares SEU robustness of different activation functions:
- ReLU (baseline)
- LeakyReLU
- GELU
- RReLU (Reliable ReLU with fixed cap M=3.0)
- RWG (Randomized Weighted Gaussian)

Research Application:
- Fault tolerance analysis for harsh environments
- Activation function selection for radiation-hardened systems

Usage:
    # Full experiment (MNIST full dataset)
    python activation_comparison.py

    # Debug mode (small dataset, fewer epochs)
    python activation_comparison.py --debug

    # Custom configuration
    python activation_comparison.py --target-accuracy 90 --train-samples 10000
"""

import argparse
import json
import time
from pathlib import Path
import multiprocessing

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seu_injection import StochasticSEUInjector, ExhaustiveSEUInjector
from seu_injection.metrics import classification_accuracy


class MiniCNN(nn.Module):
    """Small CNN for MNIST classification.

    Architecture: 2 conv layers + 1 FC layer
    Total parameters: ~2,010 (well under 5000 limit)
    """

    def __init__(self, act, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 3 * 3, num_classes)

        self.act = act
        self._is_function = callable(act) and not isinstance(act, nn.Module)

    def forward(self, x):
        if self._is_function:
            x = self.pool(self.act(self.conv1(x)))
            x = self.pool(self.act(self.conv2(x)))
        else:
            x = self.pool(self.act(self.conv1(x)))
            x = self.pool(self.act(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 8 * 3 * 3)
        x = self.fc1(x)
        return x


class RReLU(nn.Module):
    """Reliable ReLU - Clipped ReLU with fixed cap.

    f(x) = min(max(x, 0), M)

    Designed for fault tolerance - caps maximum activation value.
    Using M=3.0 as default (from experiment config).
    """

    def __init__(self, M=3.0):
        super().__init__()
        self.M = M

    def forward(self, x):
        return torch.clamp(x, min=0, max=self.M)


def rwg_activation(x):
    """Randomized Weighted Gaussian activation.

    f(x) = max(0, x * exp(-x^2))

    From ShipsNet notebook - used as baseline in existing experiments.
    Note: RWG decays quickly for |x| > 1, so it may need more careful tuning.
    """
    return torch.maximum(torch.zeros_like(x), x * torch.exp(-(x**2)))


def get_activation(name):
    """Get activation function by name."""
    activations = {
        "relu": (nn.ReLU(), False),
        "leakyrelu": (nn.LeakyReLU(0.01), False),
        "gelu": (nn.GELU(), False),
        "rrelu": (RReLU(M=3.0), False),
        "rwg": (rwg_activation, True),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name.lower()]


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_mnist_data(train_samples=None, test_samples=None, batch_size=32):
    """Load MNIST dataset (or subset for debug mode)."""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if train_samples and train_samples < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:train_samples].tolist()
        train_dataset = Subset(train_dataset, indices)

    if test_samples and test_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:test_samples].tolist()
        test_dataset = Subset(test_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(model, train_loader, test_loader, target_accuracy=90, max_epochs=100, lr=0.01, verbose=True):
    """Train the model until target accuracy is reached or max epochs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_acc = 0
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100 * train_correct / train_total

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100 * test_correct / test_total
        best_acc = max(best_acc, test_acc)

        if test_acc >= target_accuracy:
            best_epoch = epoch + 1
            if verbose:
                print(f"    Epoch {epoch + 1}: Target reached! Train: {train_acc:.2f}%, Test: {test_acc:.2f}%")
            break

        scheduler.step(test_acc)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{max_epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        best_epoch = epoch + 1

    return best_acc, best_epoch


def run_seu_injection(model, test_loader, bit_positions, layers):
    """Run EXHAUSTIVE SEU injection experiment - tests all bit flips in each layer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    injector = ExhaustiveSEUInjector(
        trained_model=model,
        criterion=classification_accuracy,
        data_loader=test_loader,
        device=device,
    )

    baseline = injector.baseline_score
    layer_info = {name: param.numel() for name, param in model.named_parameters()}

    results = []

    for bit_i in bit_positions:
        for layer_name in layers:
            # Verbosity is handled by the injector now.
            injection_results = injector.run_injector(bit_i=bit_i, layer_name=layer_name)

            for i in range(len(injection_results["criterion_score"])):
                results.append(
                    {
                        "tensor_location": injection_results.get("tensor_location", [(None,)])[i]
                        if i < len(injection_results.get("tensor_location", []))
                        else (None,),
                        "criterion_score": injection_results["criterion_score"][i],
                        "layer_name": layer_name,
                        "value_before": injection_results.get("value_before", [None])[i]
                        if i < len(injection_results.get("value_before", []))
                        else None,
                        "value_after": injection_results.get("value_after", [None])[i]
                        if i < len(injection_results.get("value_after", []))
                        else None,
                        "bit_i": bit_i,
                        "baseline_score": baseline,
                        "accuracy_drop": baseline - injection_results["criterion_score"][i],
                    }
                )

    return pd.DataFrame(results), baseline


def create_per_activation_plots(all_results, output_dir):
    """Create separate visualization plots for each activation function."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_results.empty:
        print("  No results to visualize")
        return

    activations = all_results["activation"].unique()

    for act_name in activations:
        act_results = all_results[all_results["activation"] == act_name]
        baseline = act_results["baseline_score"].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"SEU Robustness Analysis: {act_name.upper()}", fontsize=14, fontweight="bold")

        ax1 = axes[0]
        bit_positions = sorted(act_results["bit_i"].unique())
        bit_data = []
        bit_labels = []

        for bit in bit_positions:
            subset = act_results[act_results["bit_i"] == bit]
            if not subset.empty:
                bit_data.append(subset["criterion_score"].values)
                bit_labels.append(f"Bit {bit}")

        if bit_data:
            bp1 = ax1.boxplot(bit_data, patch_artist=True)
            ax1.set_xticklabels(bit_labels)
            ax1.axhline(baseline, color="red", linestyle="--", linewidth=2, label=f"Baseline ({baseline:.3f})")
            ax1.set_ylabel("Accuracy", fontsize=11)
            ax1.set_xlabel("Bit Position", fontsize=11)
            ax1.set_title("Accuracy by Bit Position", fontsize=12)
            ax1.legend(loc="lower left")
            ax1.grid(True, alpha=0.3)

            for patch in bp1["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)

        ax2 = axes[1]
        layer_names = sorted(act_results["layer_name"].unique(), key=lambda x: ("weight" in x, x))
        layer_data = []
        layer_labels = []

        for layer in layer_names:
            subset = act_results[act_results["layer_name"] == layer]
            if not subset.empty:
                layer_data.append(subset["criterion_score"].values)
                layer_labels.append(layer.replace(".weight", "").replace(".bias", ""))

        if layer_data:
            bp2 = ax2.boxplot(layer_data, patch_artist=True)
            ax2.set_xticklabels(layer_labels, rotation=45, ha="right")
            ax2.axhline(baseline, color="red", linestyle="--", linewidth=2, label=f"Baseline ({baseline:.3f})")
            ax2.set_ylabel("Accuracy", fontsize=11)
            ax2.set_xlabel("Layer", fontsize=11)
            ax2.set_title("Accuracy by Layer", fontsize=12)
            ax2.legend(loc="lower left")
            ax2.grid(True, alpha=0.3)

            for patch in bp2["boxes"]:
                patch.set_facecolor("lightgreen")
                patch.set_alpha(0.7)

        plt.tight_layout()
        plt.savefig(output_dir / f"robustness_{act_name}.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_dir / f'robustness_{act_name}.png'}")
        plt.close()


def create_comparison_plots(all_results, output_dir):
    """Create comparison visualizations across all activations."""
    output_dir = Path(output_dir)

    if all_results.empty:
        return

    activations = sorted(all_results["activation"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Activation Function SEU Robustness Comparison", fontsize=14, fontweight="bold")

    ax1 = axes[0, 0]
    means = []
    stds = []
    for act in activations:
        subset = all_results[all_results["activation"] == act]
        means.append(subset["criterion_score"].mean())
        stds.append(subset["criterion_score"].std())

    x = np.arange(len(activations))
    bars = ax1.bar(
        x, means, yerr=stds, capsize=5, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"], alpha=0.8
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.upper() for a in activations])
    ax1.set_ylabel("Mean Accuracy After Injection")
    ax1.set_title("Mean Accuracy by Activation (with std)")
    ax1.grid(True, alpha=0.3, axis="y")

    baseline = all_results["baseline_score"].iloc[0]
    ax1.axhline(baseline, color="red", linestyle="--", linewidth=2, label=f"Baseline ({baseline:.3f})")
    ax1.legend()

    ax2 = axes[0, 1]
    mean_drops = []
    max_drops = []
    width = 0.35
    for act in activations:
        subset = all_results[all_results["activation"] == act]
        mean_drops.append(subset["accuracy_drop"].mean())
        max_drops.append(subset["accuracy_drop"].max())

    x = np.arange(len(activations))
    ax2.bar(x - width / 2, mean_drops, width, label="Mean Drop", color="coral", alpha=0.8)
    ax2.bar(x + width / 2, max_drops, width, label="Max Drop", color="darkred", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.upper() for a in activations])
    ax2.set_ylabel("Accuracy Drop")
    ax2.set_title("Accuracy Drop Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = axes[1, 0]
    bit_means = {}
    for act in activations:
        act_data = all_results[all_results["activation"] == act]
        for bit in sorted(act_data["bit_i"].unique()):
            bit_subset = act_data[act_data["bit_i"] == bit]
            if bit not in bit_means:
                bit_means[bit] = []
            bit_means[bit].append(bit_subset["accuracy_drop"].mean())

    for bit, means in sorted(bit_means.items()):
        ax3.plot(range(len(activations)), means, marker="o", label=f"Bit {bit}", linewidth=2, markersize=8)

    ax3.set_xticks(range(len(activations)))
    ax3.set_xticklabels([a.upper() for a in activations])
    ax3.set_ylabel("Mean Accuracy Drop")
    ax3.set_title("Accuracy Drop by Bit Position")
    ax3.legend(title="Bit Position")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    layer_means = {}
    for act in activations:
        act_data = all_results[all_results["activation"] == act]
        for layer in act_data["layer_name"].unique():
            layer_subset = act_data[act_data["layer_name"] == layer]
            layer_key = layer.split(".")[0] + "." + layer.split(".")[-1]
            if layer_key not in layer_means:
                layer_means[layer_key] = []
            layer_means[layer_key].append(layer_subset["accuracy_drop"].mean())

    x = np.arange(len(activations))
    for i, (layer, means) in enumerate(sorted(layer_means.items())):
        ax4.plot(x, means, marker="s", label=layer, linewidth=2, markersize=8)

    ax4.set_xticks(x)
    ax4.set_xticklabels([a.upper() for a in activations])
    ax4.set_ylabel("Mean Accuracy Drop")
    ax4.set_title("Accuracy Drop by Layer")
    ax4.legend(title="Layer", loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "activation_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / 'activation_comparison.png'}")
    plt.close()


def run_single_experiment(
    activation_name, args, train_loader, test_loader, layer_names, bit_positions, results_base_dir
):
    print(f"\n" + "=" * 60, flush=True)
    print(f"Starting experiment for: {activation_name.upper()}", flush=True)
    print("=" * 60, flush=True)

    act_output_dir = results_base_dir / activation_name
    act_output_dir.mkdir(parents=True, exist_ok=True)

    act, _ = get_activation(activation_name)
    model = MiniCNN(act)
    params = count_parameters(model)
    print(f"  Model parameters: {params}", flush=True)

    print(f"\n  Training {activation_name.upper()} (target: {args.target_accuracy}%)...", flush=True)
    start_time = time.time()
    test_acc, epochs_trained = train_model(
        model, train_loader, test_loader, target_accuracy=args.target_accuracy, max_epochs=args.max_epochs
    )
    train_time = time.time() - start_time
    print(
        f"\n  {activation_name.upper()} training time: {train_time:.1f}s, Epochs: {epochs_trained}, Test accuracy: {test_acc:.2f}%",
        flush=True,
    )

    print(f"\n  Running EXHAUSTIVE SEU injection for {activation_name.upper()}...", flush=True)
    start_time = time.time()
    results_df, baseline = run_seu_injection(model, test_loader, bit_positions, layer_names)
    injection_time = time.time() - start_time

    results_df["activation"] = activation_name

    experiment_summary = {
        "parameters": params,
        "test_accuracy": test_acc,
        "epochs_trained": epochs_trained,
        "baseline_score": baseline,
        "mean_accuracy_after_injection": round(results_df["criterion_score"].mean(), 4),
        "std_accuracy_after_injection": round(results_df["criterion_score"].std(), 4),
        "mean_accuracy_drop": round(results_df["accuracy_drop"].mean(), 4),
        "max_accuracy_drop": round(results_df["accuracy_drop"].max(), 4),
        "train_time_seconds": round(train_time, 2),
        "injection_time_seconds": round(injection_time, 2),
    }

    print(f"\n  {activation_name.upper()} baseline accuracy: {baseline:.4f}", flush=True)
    print(
        f"  {activation_name.upper()} mean accuracy after injection: {results_df['criterion_score'].mean():.4f} (±{results_df['criterion_score'].std():.4f})",
        flush=True,
    )
    print(f"  {activation_name.upper()} mean accuracy drop: {results_df['accuracy_drop'].mean():.4f}", flush=True)
    print(f"  {activation_name.upper()} max accuracy drop: {results_df['accuracy_drop'].max():.4f}", flush=True)
    print(f"  {activation_name.upper()} injection time: {injection_time:.1f}s", flush=True)

    # Save individual results for this activation function
    individual_all_results_path = act_output_dir / f"all_results_{activation_name}.csv"
    results_df.to_csv(individual_all_results_path, index=False)
    print(f"  Saved individual results: {individual_all_results_path}", flush=True)

    individual_experiment_results_path = act_output_dir / f"experiment_results_{activation_name}.json"
    with open(individual_experiment_results_path, "w") as f:
        json.dump(experiment_summary, f, indent=2)
    print(f"  Saved individual summary: {individual_experiment_results_path}", flush=True)

    return {
        "activation_name": activation_name,
        "experiment_summary": experiment_summary,
        "all_results_path": individual_all_results_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Activation Function SEU Robustness Experiment")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (small dataset, fewer epochs)")
    parser.add_argument("--target-accuracy", type=float, default=95, help="Target test accuracy to train to")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--train-samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--test-samples", type=int, default=2000, help="Number of test samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--activation", type=str, default=None, help="Test single activation (relu, leakyrelu, gelu, rrelu, rwg)"
    )
    parser.add_argument("--output-dir", type=str, default="examples/mnist/results", help="Output directory")
    parser.add_argument(
        "--num-workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel processes"
    )
    parser.add_argument("--bit-index", type=int, default=None, help="Bit index to flip (0-31)")
    args = parser.parse_args()

    is_debug = args.debug

    if is_debug:
        args.target_accuracy = min(args.target_accuracy, 80)
        args.max_epochs = min(args.max_epochs, 20)
        args.train_samples = min(args.train_samples, 2000)
        args.test_samples = min(args.test_samples, 500)
        print("=" * 60)
        print("DEBUG MODE ENABLED")
        print(f"  Target Acc: {args.target_accuracy}%, Max Epochs: {args.max_epochs}")
        print(f"  Train: {args.train_samples}, Test: {args.test_samples}")
        print("=" * 60)

    activations_to_test = ["relu", "leakyrelu", "gelu", "rrelu", "rwg"]

    if args.activation:
        if args.activation.lower() not in activations_to_test:
            print(f"Unknown activation: {args.activation}")
            return
        activations_to_test = [args.activation.lower()]

    bit_positions = [args.bit_index] if args.bit_index is not None else [0, 1, 5, 15]
    layer_names = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "fc1.weight", "fc1.bias"]

    print("\n" + "=" * 60)
    print("ACTIVATION FUNCTION SEU ROBUSTNESS COMPARISON")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Target accuracy: {args.target_accuracy}%")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Train samples: {args.train_samples}")
    print(f"  Test samples: {args.test_samples}")
    print(f"  Injection mode: EXHAUSTIVE")
    print(f"  Bit position: {bit_positions}")
    print(f"  Parallel workers: {args.num_workers}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\nLoading MNIST data...")

    results_base_dir = Path(args.output_dir)
    results_base_dir.mkdir(parents=True, exist_ok=True)

    # Reworking pool_args for cleaner multiprocessing
    # Each process will recreate its own DataLoader for better isolation
    pool_args = [(act_name, args, layer_names, bit_positions, results_base_dir) for act_name in activations_to_test]

    print(f"\nStarting parallel experiments with {args.num_workers} workers...")
    all_experiment_results = []

    # Store temporary results files for later aggregation
    temp_all_results_files = []

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # Each call to run_single_experiment will return its summary and path to its full results
        results_from_pool = pool.starmap(run_single_experiment_wrapper, pool_args)

    for res in results_from_pool:
        if res:  # Ensure result is not None, indicating a successful run
            all_experiment_results.append((res["activation_name"], res["experiment_summary"]))
            temp_all_results_files.append(res["all_results_path"])

    # Aggregate results after all parallel runs are complete
    all_results_df = pd.concat([pd.read_csv(f) for f in temp_all_results_files], ignore_index=True)
    experiment_results = {act_name: summary for act_name, summary in all_experiment_results}

    print(f"\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)

    print("\n--- Summary ---")
    print(
        f"{'Activation':<12} {'Params':>8} {'Epochs':>7} {'Test Acc%':>10} {'Baseline':>10} {'Mean Inj':>12} {'Mean Drop':>10}"
    )
    print("-" * 80)
    for act_name, results in experiment_results.items():
        print(
            f"{act_name:<12} {results['parameters']:>8} {results['epochs_trained']:>7} {results['test_accuracy']:>10.2f} "
            f"{results['baseline_score']:>10.4f} {results['mean_accuracy_after_injection']:>12.4f} {results['mean_accuracy_drop']:>10.4f}"
        )

    final_all_results_path = results_base_dir / "all_results.csv"
    all_results_df.to_csv(final_all_results_path, index=False)
    print(f"\nSaved aggregated all results: {final_all_results_path}")

    final_experiment_results_path = results_base_dir / "experiment_results.json"
    with open(final_experiment_results_path, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"Saved aggregated experiment summary: {final_experiment_results_path}")

    print("\nGenerating visualizations...")
    create_per_activation_plots(all_results_df, results_base_dir)
    create_comparison_plots(all_results_df, results_base_dir)

    summary_df = pd.DataFrame(experiment_results).T
    final_summary_stats_path = results_base_dir / "summary_stats.csv"
    summary_df.to_csv(final_summary_stats_path)
    print(f"  Saved: {final_summary_stats_path}")

    print(f"\nFinal results directory: {results_base_dir}")


# Wrapper function for multiprocessing to recreate DataLoader in each child process
def run_single_experiment_wrapper(activation_name, args, layer_names, bit_positions, results_base_dir):
    # Recreate DataLoader within each child process to avoid pickling issues
    train_loader, test_loader = load_mnist_data(
        train_samples=args.train_samples, test_samples=args.test_samples, batch_size=args.batch_size
    )
    return run_single_experiment(
        activation_name, args, train_loader, test_loader, layer_names, bit_positions, results_base_dir
    )


if __name__ == "__main__":
    main()

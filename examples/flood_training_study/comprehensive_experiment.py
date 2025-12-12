#!/usr/bin/env python3
"""Comprehensive Flood Level Training Study for SEU Robustness

Extended experiments across multiple datasets, flood levels, and configurations.

Research Question:
    Does flood level training improve neural network robustness to SEUs?
    
Methodology:
    - Multiple datasets: moons, circles, blobs
    - Multiple flood levels: [0.05, 0.10, 0.15, 0.20, 0.30]
    - With/without dropout
    - Higher SEU sampling rate (15%)
    - Statistical validation
    
Author: SEU Injection Framework Research Team
Date: December 2025
"""

import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from seu_injection.core import StochasticSEUInjector
from seu_injection.metrics import classification_accuracy


# ============================================================================
# Flood Level Training Implementation
# ============================================================================

class FloodingLoss(nn.Module):
    """Implements flooding regularization."""
    
    def __init__(self, base_loss, flood_level=0.10):
        super().__init__()
        self.base_loss = base_loss
        self.flood_level = flood_level
    
    def forward(self, predictions, targets):
        loss = self.base_loss(predictions, targets)
        return torch.abs(loss - self.flood_level) + self.flood_level


# ============================================================================
# Model Architectures
# ============================================================================

def create_mlp_with_dropout():
    """MLP with 20% dropout."""
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


def create_mlp_without_dropout():
    """MLP without dropout."""
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


# ============================================================================
# Dataset Preparation
# ============================================================================

def prepare_dataset(dataset_name='moons', n_samples=2000, noise=0.3, random_state=42):
    """Prepare one of multiple datasets."""
    
    if dataset_name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_name == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif dataset_name == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, 
                         cluster_std=noise*5, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# Training Functions
# ============================================================================

def train_model(model, x_train, y_train, x_val, y_val, criterion, epochs=100, verbose=False):
    """Train model with given criterion."""
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    train_losses = []
    val_losses = []
    
    base_criterion = criterion.base_loss if hasattr(criterion, 'base_loss') else criterion
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation (always use base loss)
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = base_criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
        model.train()
        
        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Train: {loss.item():.4f}, Val: {val_loss.item():.4f}")
    
    model.eval()
    return model, train_losses, val_losses


def evaluate_baseline_accuracy(model, x_test, y_test):
    """Evaluate baseline accuracy."""
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()
    return accuracy


# ============================================================================
# SEU Injection Evaluation
# ============================================================================

def evaluate_seu_robustness(model, x_test, y_test, sampling_rate=0.15, bit_positions=None):
    """Evaluate SEU robustness with higher sampling rate."""
    
    if bit_positions is None:
        bit_positions = [31, 30, 23, 22, 0]  # Sign, exponent, mantissa
    
    injector = StochasticSEUInjector(
        trained_model=model,
        criterion=classification_accuracy,
        x=x_test,
        y=y_test,
    )
    
    baseline_acc = injector.baseline_score
    
    results = {}
    for bit_i in bit_positions:
        injection_results = injector.run_injector(bit_i=bit_i, p=sampling_rate)
        
        if len(injection_results["criterion_score"]) > 0:
            fault_scores = injection_results["criterion_score"]
            mean_acc = np.mean(fault_scores)
            accuracy_drop = baseline_acc - mean_acc
            critical_faults = sum(1 for score in fault_scores if (baseline_acc - score) > 0.1)
            critical_fault_rate = critical_faults / len(fault_scores)
            
            results[bit_i] = {
                "mean_accuracy": mean_acc,
                "accuracy_drop": accuracy_drop,
                "critical_fault_rate": critical_fault_rate,
                "num_injections": len(fault_scores),
            }
    
    return baseline_acc, results


# ============================================================================
# Comprehensive Experiment
# ============================================================================

def run_comprehensive_experiment():
    """Run experiments across multiple configurations."""
    
    print("="*80)
    print("COMPREHENSIVE FLOOD LEVEL TRAINING STUDY")
    print("="*80)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    datasets = ['moons', 'circles', 'blobs']
    flood_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]  # 0.0 = standard training
    dropout_configs = [True, False]
    sampling_rate = 0.15  # 15% sampling (up from 5%)
    
    all_results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(dataset_name)
        
        for use_dropout in dropout_configs:
            dropout_str = "with_dropout" if use_dropout else "no_dropout"
            print(f"\n--- Configuration: {dropout_str} ---")
            
            for flood_level in flood_levels:
                config_name = f"standard" if flood_level == 0.0 else f"flood_{flood_level}"
                print(f"\n  Testing {config_name}...")
                
                # Create model
                if use_dropout:
                    model = create_mlp_with_dropout()
                else:
                    model = create_mlp_without_dropout()
                
                # Create criterion
                base_criterion = nn.BCELoss()
                if flood_level > 0.0:
                    criterion = FloodingLoss(base_criterion, flood_level=flood_level)
                else:
                    criterion = base_criterion
                
                # Train
                model, train_losses, val_losses = train_model(
                    model, X_train, y_train, X_val, y_val, criterion, epochs=100, verbose=False
                )
                
                # Evaluate baseline
                baseline_acc = evaluate_baseline_accuracy(model, X_test, y_test)
                
                # Evaluate SEU robustness
                _, seu_results = evaluate_seu_robustness(
                    model, X_test, y_test, sampling_rate=sampling_rate
                )
                
                # Calculate overall metrics
                if seu_results:
                    mean_drop = np.mean([r["accuracy_drop"] for r in seu_results.values()])
                    mean_cfr = np.mean([r["critical_fault_rate"] for r in seu_results.values()])
                else:
                    mean_drop = 0.0
                    mean_cfr = 0.0
                
                result = {
                    'dataset': dataset_name,
                    'dropout': use_dropout,
                    'flood_level': flood_level,
                    'baseline_accuracy': baseline_acc,
                    'final_train_loss': train_losses[-1],
                    'final_val_loss': val_losses[-1],
                    'mean_accuracy_drop': mean_drop,
                    'mean_critical_fault_rate': mean_cfr,
                    'seu_by_bit': seu_results
                }
                
                all_results.append(result)
                
                print(f"    Baseline Acc: {baseline_acc:.3f}, "
                      f"Train Loss: {train_losses[-1]:.4f}, "
                      f"Val Loss: {val_losses[-1]:.4f}")
                print(f"    SEU Drop: {mean_drop:.3f}, CFR: {mean_cfr:.3f}")
    
    # Save results as JSON
    with open('data/comprehensive_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save results as CSV
    csv_headers = ['dataset', 'dropout', 'flood_level', 'baseline_accuracy', 
                   'final_train_loss', 'final_val_loss', 'mean_accuracy_drop', 
                   'mean_critical_fault_rate']
    with open('data/comprehensive_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for result in all_results:
            row = {k: result[k] for k in csv_headers}
            writer.writerow(row)
    
    print(f"\n{'='*80}")
    print("Results saved to:")
    print("  - data/comprehensive_results.json")
    print("  - data/comprehensive_results.csv")
    print(f"{'='*80}")
    
    return all_results


# ============================================================================
# Analysis and Visualization
# ============================================================================

def analyze_results(results):
    """Analyze and visualize comprehensive results."""
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    # Group by dataset
    for dataset in ['moons', 'circles', 'blobs']:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        
        print(f"\n{dataset.upper()}:")
        print("-" * 40)
        
        # With dropout
        with_dropout = [r for r in dataset_results if r['dropout']]
        without_dropout = [r for r in dataset_results if not r['dropout']]
        
        for config, label in [(with_dropout, "With Dropout"), (without_dropout, "Without Dropout")]:
            if not config:
                continue
            
            print(f"\n  {label}:")
            
            # Find standard (flood_level=0)
            standard = [r for r in config if r['flood_level'] == 0.0][0]
            
            print(f"    Standard: Acc={standard['baseline_accuracy']:.3f}, "
                  f"SEU Drop={standard['mean_accuracy_drop']:.3f}")
            
            # Compare flood levels
            for r in sorted(config, key=lambda x: x['flood_level']):
                if r['flood_level'] == 0.0:
                    continue
                
                acc_change = r['baseline_accuracy'] - standard['baseline_accuracy']
                drop_change = r['mean_accuracy_drop'] - standard['mean_accuracy_drop']
                improvement = -drop_change / standard['mean_accuracy_drop'] * 100 if standard['mean_accuracy_drop'] > 0 else 0
                
                print(f"    Flood {r['flood_level']:.2f}: "
                      f"Acc Δ={acc_change:+.3f}, "
                      f"Drop Δ={drop_change:+.3f}, "
                      f"Improve={improvement:+.1f}%")


if __name__ == "__main__":
    # Run comprehensive experiment
    results = run_comprehensive_experiment()
    
    # Analyze
    analyze_results(results)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("Results saved to: data/comprehensive_results.json")

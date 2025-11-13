# ShipsNet Fault Injection Experiments

This folder contains experiments re-creating the methodology from the 2025 paper, [A Framework for Developing Robust Machine Learning Models in Harsh Environments: A Review of CNN Design Choices](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars/), focusing on fault injection and robustness analysis for the ShipsNet dataset.

## Overview

The goal is to systematically evaluate the robustness of CNN models trained on the ShipsNet dataset under Single Event Upset (SEU) fault injection, following the experimental design and analysis described in the referenced publication.

## Dataset

- **ShipsNet**: A satellite image dataset for ship classification.
- Download and preprocessing instructions will be provided here.

## Methodology

- **Fault Model**: Single Event Upset (SEU) via targeted bit manipulation in model weights.
- **Injection Protocol**: Systematic and/or stochastic bit-flip injection across network layers.
- **Metrics**: Classification accuracy, degradation, and layer-wise vulnerability.
- **Analysis**: Compare baseline and post-injection performance, following the referenced 2025 paper.

## Experiment Plan

1. Baseline model training and evaluation
1. Systematic SEU injection (per-bit, per-layer)
1. Stochastic SEU injection (random sampling)
1. Robustness metrics calculation
1. Visualization and reporting

## Results

Results and analysis will be added here as experiments progress.

## References

- Dennis, W., & Pope, J. (2025). A Framework for Developing Robust Machine Learning Models in Harsh Environments: A Review of CNN Design Choices. [Link](https://research-information.bris.ac.uk/en/publications/a-framework-for-developing-robust-machine-learning-models-in-hars/)

"""
Metrics for evaluating neural network performance under fault injection.

This module provides comprehensive evaluation metrics for studying model
robustness under Single Event Upset (SEU) conditions.
"""

from .accuracy import (
    classification_accuracy,
    classification_accuracy_loader,
    multiclass_classification_accuracy,
)

__all__ = [
    "classification_accuracy",
    "classification_accuracy_loader",
    "multiclass_classification_accuracy",
]

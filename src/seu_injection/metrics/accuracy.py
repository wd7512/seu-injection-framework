"""
Accuracy metrics for neural network evaluation under SEU injection.

This module provides robust accuracy calculation functions that work with
both tensor inputs and DataLoader objects for comprehensive model evaluation.
"""

from typing import Optional, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score


def classification_accuracy_loader(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: Optional[Union[str, torch.device]] = None
) -> float:
    """
    Calculate classification accuracy using a PyTorch DataLoader.
    
    This function evaluates model accuracy across an entire dataset provided
    via DataLoader, handling batching and device placement automatically.
    
    Args:
        model: PyTorch model in evaluation mode
        data_loader: DataLoader containing (X, y) batches
        device: Computing device ('cpu', 'cuda', or torch.device)
        
    Returns:
        Classification accuracy as a float in [0, 1]
        
    Note:
        Automatically handles both binary and multiclass classification
        based on model output shape.
    """
    model.eval()
    if device:
        model = model.to(device)

    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            if device:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
            batch_pred = model(batch_X)
            y_pred_list.append(batch_pred)
            y_true_list.append(batch_y)

    y_pred_all = torch.cat(y_pred_list).cpu().numpy()
    y_true_all = torch.cat(y_true_list).cpu().numpy()

    return multiclass_classification_accuracy(y_true_all, y_pred_all)


def classification_accuracy(
    model: torch.nn.Module,
    X_tensor: Union[torch.Tensor, torch.utils.data.DataLoader],
    y_true: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 64
) -> float:
    """
    Calculate classification accuracy with automatic DataLoader detection.
    
    This function automatically detects whether the input is a tensor or
    DataLoader and routes to the appropriate evaluation method.
    
    Args:
        model: PyTorch model to evaluate
        X_tensor: Input tensor OR DataLoader (auto-detected)
        y_true: Target labels (ignored if X_tensor is DataLoader)
        device: Computing device for evaluation
        batch_size: Batch size for tensor evaluation (ignored for DataLoader)
        
    Returns:
        Classification accuracy as a float in [0, 1]
        
    Raises:
        ValueError: If DataLoader provided but y_true is also specified
        
    Example:
        >>> # Using tensors
        >>> acc = classification_accuracy(model, X_test, y_test)
        >>> # Using DataLoader  
        >>> acc = classification_accuracy(model, test_loader)
    """
    # Check if X_tensor is actually a DataLoader
    if hasattr(X_tensor, '__iter__') and hasattr(X_tensor, 'dataset'):
        # It's a DataLoader, use the loader function
        if y_true is not None:
            raise ValueError(
                "When using DataLoader, do not specify y_true separately. "
                "Labels should be included in the DataLoader."
            )
        return classification_accuracy_loader(model, X_tensor, device)

    # Handle tensor inputs
    if device:
        model = model.to(device)
        X_tensor = X_tensor.to(device)
        if y_true is not None:
            y_true = y_true.to(device)

    model.eval()
    y_pred_list = []
    y_true_list = []

    if batch_size is None:
        batch_size = len(X_tensor)

    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            end = start + batch_size
            batch_X = X_tensor[start:end]
            batch_y = y_true[start:end]
            batch_pred = model(batch_X)
            y_pred_list.append(batch_pred)
            y_true_list.append(batch_y)

    y_pred_all = torch.cat(y_pred_list).cpu().numpy()
    y_true_all = torch.cat(y_true_list).cpu().numpy()

    return multiclass_classification_accuracy(y_true_all, y_pred_all)


def multiclass_classification_accuracy(
    y_true: np.ndarray,
    model_output: np.ndarray
) -> float:
    """
    Calculate accuracy for both binary and multiclass classification.
    
    This function automatically detects the classification type based on
    model output shape and applies appropriate prediction logic.
    
    Args:
        y_true: True class labels
        model_output: Raw model outputs (logits or probabilities)
        
    Returns:
        Classification accuracy as a float in [0, 1]
        
    Note:
        - Binary classification: Uses midpoint thresholding between min/max labels
        - Multiclass: Uses argmax over output dimensions
    """
    if model_output.ndim == 1 or model_output.shape[1] == 1:
        # Binary classification case
        y_low = np.min(y_true)
        y_high = np.max(y_true)
        midpoint = (y_high + y_low) / 2

        y_pred = np.zeros_like(y_true) + y_low
        y_pred[model_output.flatten() >= midpoint] = y_high
    else:
        # Multiclass classification case
        y_pred = np.argmax(model_output, axis=1)

    return accuracy_score(y_true=y_true, y_pred=y_pred)

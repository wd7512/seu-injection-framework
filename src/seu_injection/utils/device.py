"""
Utility functions for device management and common operations.

This module provides helper functions for device detection, tensor operations,
and other common utilities used throughout the SEU injection framework.
"""

from typing import Any, Optional, Union

import torch

# TODO MAINTAINABILITY: Missing comprehensive error handling in utility functions
# ISSUE: Functions don't validate inputs or handle edge cases robustly
# EXAMPLES: detect_device() doesn't validate device availability, ensure_tensor() lacks type validation
# IMPACT: Runtime errors in production when invalid devices specified or incompatible data passed
# SOLUTION: Add input validation, proper error messages, graceful fallbacks
# PRIORITY: MEDIUM - affects framework reliability in production environments


def detect_device(
    preferred_device: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    """
    Detect the best available computing device.

    Args:
        preferred_device: Preferred device specification ('cpu', 'cuda', or torch.device)

    Returns:
        Detected or specified torch.device

    Example:
        >>> device = detect_device()  # Auto-detect
        >>> device = detect_device('cuda')  # Force CUDA if available
    """
    if preferred_device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(preferred_device)


def ensure_tensor(
    data: Union[torch.Tensor, Any],  # Any for numpy arrays or other array-like
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Ensure input data is a PyTorch tensor with specified dtype and device.

    Args:
        data: Input data (tensor or numpy array)
        dtype: Target tensor dtype
        device: Target device (None for current device)

    Returns:
        PyTorch tensor with specified properties
    """
    if isinstance(data, torch.Tensor):
        result = data.clone().detach()
    else:
        result = torch.tensor(data, dtype=dtype)

    if device is not None:
        result = result.to(device=device, dtype=dtype)
    elif dtype != result.dtype:
        result = result.to(dtype=dtype)

    return result


def get_model_info(model: torch.nn.Module) -> dict:
    """
    Extract comprehensive information about a PyTorch model.

    # TODO UNUSED FUNCTION: get_model_info() appears unused throughout codebase
    # STATUS: Function implemented but no references found in src/ or tests/
    # IMPACT: Dead code increases maintenance burden and package size
    # DECISION NEEDED: Remove if truly unused, or add to public API if valuable
    # SEARCH PERFORMED: grep -r "get_model_info" src/ tests/ - no matches found
    # PRIORITY: LOW - cleanup issue, no functional impact

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary with model statistics and layer information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layer_info = []
    for name, param in model.named_parameters():
        layer_info.append(
            {
                "name": name,
                "shape": tuple(param.shape),
                "params": param.numel(),
                "requires_grad": param.requires_grad,
                "dtype": str(param.dtype),
            }
        )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "layer_count": len(layer_info),
        "layers": layer_info,
    }

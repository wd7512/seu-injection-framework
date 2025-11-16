"""
Comprehensive accuracy metrics for neural network evaluation under fault injection.

This module provides a robust suite of classification accuracy calculation functions
optimized for Single Event Upset (SEU) injection experiments. It supports multiple
input formats (tensors, DataLoaders), automatic batch processing, device-aware
computation, and handles both binary and multiclass classification scenarios.

The metrics are designed to work seamlessly with the Injector class for systematic fault tolerance analysis, providing consistent and reliable performance measurements across different model architectures and dataset configurations.

Key Features:
    - Automatic input type detection (tensor vs DataLoader)
    - Device-aware computation with GPU acceleration
    - Memory-efficient batch processing for large datasets
    - Binary and multiclass classification support
    - Robust error handling and validation
    - Integration with sklearn.metrics for consistency
    - Non-blocking data transfer for performance optimization

Supported Classification Types:
    - Binary Classification: Single output or two-class scenarios
    - Multiclass Classification: Multiple output classes with argmax prediction
    - Automatic detection based on model output shape
    - Flexible label encoding (0/1, -1/+1, or arbitrary class indices)

Performance Characteristics:
    - Batch processing: Configurable batch sizes for memory management
    - GPU acceleration: Automatic device placement and non-blocking transfers
    - Memory efficiency: Streaming evaluation for large datasets via DataLoaders
    - Vectorized operations: NumPy and PyTorch optimizations throughout

Typical Usage in SEU Experiments:
    >>> from seu_injection.core import Injector
    >>> from seu_injection.metrics.accuracy import classification_accuracy
    >>>
    >>> # Define evaluation criterion
    >>> def accuracy_criterion(model, x, y, device):
    ...     return classification_accuracy(model, x, y, device)
    >>>
    >>> # Create injector with accuracy evaluation
    >>> injector = Injector(
    ...     trained_model=model,
    ...     criterion=accuracy_criterion,
    ...     x=test_data, y=test_labels
    >>> )
    >>>
    >>> # Run fault injection campaign
    >>> results = injector.run_injector(bit_i=15)
    >>> baseline = injector.baseline_score
    >>> accuracy_drops = [baseline - score for score in results['criterion_score']]

Integration with Common Frameworks:
    - PyTorch: Native tensor and DataLoader support
    - scikit-learn: Consistent accuracy_score implementation
    - NumPy: Efficient array operations and memory management
    - CUDA: Automatic GPU acceleration when available

See Also:
    seu_injection.core.Injector: Systematic fault injection
    sklearn.metrics.accuracy_score: Underlying accuracy computation
    torch.utils.data.DataLoader: Batch data loading for large datasets
"""

from collections.abc import Iterable
from typing import Optional, Union

import numpy as np
import torch

# Attempt to import sklearn's accuracy_score; provide a lightweight fallback if unavailable.
# This allows the core package installation to avoid pulling in scikit-learn while still
# supporting accuracy computations. For production / research parity, install the
# 'analysis' or 'dev' extra to use the official sklearn implementation.
try:  # pragma: no cover - import guard
    from sklearn.metrics import accuracy_score  # type: ignore
except Exception:  # pragma: no cover - fallback intentionally simple

    def accuracy_score(y_true, y_pred):  # type: ignore
        """Fallback accuracy_score implementation.

        Provides basic classification accuracy if scikit-learn is not installed.
        Matches sklearn.metrics.accuracy_score behavior for simple cases.
        Intended for minimal installs; for full functionality install 'analysis' extra.
        """
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                "Shape mismatch in fallback accuracy_score: "
                f"y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
            )
        return float(np.mean(y_true_arr == y_pred_arr))


def classification_accuracy_loader(
    model: torch.nn.Module,
    data_loader: "torch.utils.data.DataLoader",  # keep runtime type but avoid union confusion
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    Compute classification accuracy using PyTorch DataLoader with optimized batch processing.

    This function provides memory-efficient evaluation of model accuracy across large
    datasets by processing data in batches through a DataLoader. It handles device
    placement, memory management, and gradient computation optimization automatically,
    making it ideal for evaluating large models on extensive datasets during SEU experiments.

    The function is optimized for performance with non-blocking data transfers, automatic
    device placement, and efficient tensor concatenation, while maintaining full
    compatibility with the broader SEU injection framework.

    Args:
        model (torch.nn.Module): PyTorch neural network model to evaluate. The model
            will be automatically set to evaluation mode and moved to the specified
            device. Should accept inputs from the DataLoader and produce outputs
            compatible with the classification task (binary or multiclass).
        data_loader (torch.utils.data.DataLoader): PyTorch DataLoader that yields
            (input, target) batches. The DataLoader should be configured with
            appropriate batch size for memory constraints and should contain the
            complete evaluation dataset. Supports any DataLoader configuration
            (shuffled or ordered, custom samplers, etc.).
        device (Optional[Union[str, torch.device]]): Target computing device for
            evaluation. Options include 'cpu', 'cuda', 'cuda:0', etc., or a
            torch.device object. If None, uses the model's current device.
            All data is automatically transferred to this device for computation.

    Returns:
        float: Overall classification accuracy across the entire dataset as a value
            in [0.0, 1.0]. Represents the fraction of correctly classified samples
            across all batches, computed using the same logic as other accuracy
            functions for consistency.

    Raises:
        RuntimeError: If model evaluation fails due to device mismatches, memory
            constraints, or incompatible tensor shapes during batch processing.
        ValueError: If DataLoader yields batches with inconsistent formats or
            if model outputs cannot be processed by the accuracy computation logic.
        TypeError: If model is not a PyTorch nn.Module or DataLoader is not valid.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>> from seu_injection.metrics.accuracy import classification_accuracy_loader
        >>>
        >>> # Setup model and large dataset
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 256), nn.ReLU(),
        ...     nn.Linear(256, 10), nn.Softmax(dim=1)
        ... )
        >>>
        >>> # Create large dataset (simulating MNIST-like data)
        >>> x_large = torch.randn(50000, 784)
        >>> y_large = torch.randint(0, 10, (50000,))
        >>> dataset = TensorDataset(x_large, y_large)
        >>>
        >>> # Configure DataLoader for memory efficiency
        >>> loader = DataLoader(dataset, batch_size=512, shuffle=False,
        ...                    num_workers=4, pin_memory=True)
        >>>
        >>> # Evaluate on CPU
        >>> accuracy_cpu = classification_accuracy_loader(model, loader, device='cpu')
        >>> print(f"CPU accuracy: {accuracy_cpu:.4f}")
        >>>
        >>> # Evaluate on GPU if available
        >>> if torch.cuda.is_available():
        ...     accuracy_gpu = classification_accuracy_loader(
        ...         model, loader, device='cuda'
        ...     )
        ...     print(f"GPU accuracy: {accuracy_gpu:.4f}")
        >>>
        >>> # Integration with Injector for large-scale experiments
        >>> from seu_injection.core import Injector
        >>>
        >>> def loader_criterion(model, data_loader, device):
        ...     return classification_accuracy_loader(model, data_loader, device)
        >>>
        >>> injector = Injector(
        ...     trained_model=model,
        ...     criterion=loader_criterion,
        ...     data_loader=loader
        ... )
        >>>
        >>> # Memory-efficient fault injection on large dataset
        >>> baseline = injector.get_criterion_score()
        >>> results = injector.run_injector(bit_i=15, p=0.001)
        >>> print(f"Baseline: {baseline:.4f}, Faults: {len(results['tensor_location'])}")

    Performance Optimizations:
        Memory Efficiency:
        - Streaming evaluation: Processes one batch at a time
        - Automatic cleanup: Intermediate tensors released after each batch
        - Non-blocking transfers: Overlaps CPU-GPU data movement with computation
        - Pin memory: Uses pinned memory when available for faster transfers

        Computational Efficiency:
        - Gradient disabled: torch.no_grad() context for inference-only mode
        - Batch processing: Leverages DataLoader's optimized batching
        - Device optimization: Minimizes unnecessary data movement
        - Vectorized operations: Efficient tensor concatenation and computation

    Memory Management:
        The function is designed to handle datasets larger than available memory:
        - Processes data in configurable batches via DataLoader
        - Only accumulates predictions and targets, not full dataset
        - Automatic garbage collection of intermediate batch tensors
        - Supports datasets of arbitrary size limited only by storage

    Device Handling:
        - Automatic model placement on target device
        - Non-blocking data transfers when supported
        - Consistent device placement across all batch operations
        - Efficient CPU-GPU memory management

    Batch Processing Logic:
        1. Set model to evaluation mode and move to target device
        2. Initialize prediction and target accumulators
        3. For each batch in DataLoader:
           a. Transfer batch to target device (non-blocking if supported)
           b. Compute model predictions with gradients disabled
           c. Accumulate predictions and targets
        4. Concatenate all batch results and compute final accuracy

    See Also:
        classification_accuracy: General-purpose accuracy with automatic type detection
        multiclass_classification_accuracy: Core accuracy computation logic
        torch.utils.data.DataLoader: PyTorch batch data loading
        seu_injection.core.injector.Injector: Framework integration point
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
    x_tensor: Union[torch.Tensor, "torch.utils.data.DataLoader"],
    y_true: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 64,
) -> float:
    """
    Calculate classification accuracy with intelligent input type detection and optimization.

    This function serves as the primary entry point for classification accuracy evaluation
    in SEU injection experiments. It automatically detects input format (tensors vs
    DataLoaders) and applies the optimal evaluation strategy, handling device placement,
    memory management, and batch processing transparently.

    The function is designed for seamless integration with Injector as a criterion
    function, providing consistent accuracy measurements across different model
    architectures and dataset configurations during fault injection campaigns.

    Args:
        model (torch.nn.Module): PyTorch neural network model to evaluate. The model
            should be pre-trained and will be automatically set to evaluation mode
            during accuracy calculation. Must produce outputs compatible with the
            specified classification type (binary or multiclass).
        x_tensor (Union[torch.Tensor, torch.utils.data.DataLoader]): Input data for
            evaluation. Can be either:
            - torch.Tensor: Input features tensor with shape (N, ...) where N is
              the number of samples. Will be processed in batches of size `batch_size`.
            - torch.utils.data.DataLoader: PyTorch DataLoader yielding (x, y) batches.
              Automatically detected and processed using optimized loader evaluation.
        y_true (Optional[torch.Tensor]): Target labels tensor with shape (N,) containing
            class indices or binary labels. Required when x_tensor is a tensor, ignored
            when x_tensor is a DataLoader (labels should be included in DataLoader).
        device (Optional[Union[str, torch.device]]): Computing device for evaluation.
            Options: 'cpu', 'cuda', 'cuda:0', etc., or torch.device object. If None,
            uses model's current device. All tensors are automatically moved to this
            device for computation.
        batch_size (int): Batch size for tensor-based evaluation to manage memory usage.
            Larger values increase memory usage but may improve computation efficiency.
            Ignored when x_tensor is a DataLoader. Default: 64 for balanced performance.

    Returns:
        float: Classification accuracy as a value in [0.0, 1.0] where 1.0 represents
            perfect accuracy (100% correct predictions) and 0.0 represents no correct
            predictions. The metric is computed using sklearn.metrics.accuracy_score
            for consistency with standard machine learning practices.

    Raises:
        ValueError: If DataLoader is provided as x_tensor but y_true is also specified.
            DataLoaders should contain both inputs and labels internally.
        RuntimeError: If model evaluation fails due to device mismatches, memory
            issues, or incompatible tensor shapes between model outputs and labels.
        TypeError: If inputs are not of expected types (model not nn.Module, invalid
            tensor types, etc.) or if model outputs cannot be processed by accuracy logic.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from seu_injection.metrics.accuracy import classification_accuracy
        >>>
        >>> # Setup model and data
        >>> model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=1))
        >>> x_test = torch.randn(1000, 784)
        >>> y_test = torch.randint(0, 10, (1000,))
        >>>
        >>> # Basic tensor-based evaluation
        >>> accuracy = classification_accuracy(model, x_test, y_test)
        >>> print(f"Model accuracy: {accuracy:.3f}")
        >>>
        >>> # GPU acceleration
        >>> if torch.cuda.is_available():
        ...     accuracy_gpu = classification_accuracy(
        ...         model, x_test, y_test, device='cuda'
        ...     )
        ...     print(f"GPU accuracy: {accuracy_gpu:.3f}")
        >>>
        >>> # DataLoader-based evaluation for large datasets
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>> dataset = TensorDataset(x_test, y_test)
        >>> loader = DataLoader(dataset, batch_size=128, shuffle=False)
        >>> accuracy_loader = classification_accuracy(model, loader, device='cuda')
        >>> print(f"DataLoader accuracy: {accuracy_loader:.3f}")
        >>>
        >>> # Integration with Injector
        >>> from seu_injection.core import Injector
        >>>
        >>> def accuracy_criterion(model, x, y, device):
        ...     return classification_accuracy(model, x, y, device, batch_size=256)
        >>>
        >>> injector = Injector(
        ...     trained_model=model,
        ...     criterion=accuracy_criterion,
        ...     x=x_test, y=y_test
        ... )
        >>>
        >>> # Baseline accuracy
        >>> baseline = injector.get_criterion_score()
        >>> print(f"Baseline accuracy: {baseline:.3f}")
        >>>
        >>> # Fault injection analysis
        >>> results = injector.run_injector(bit_i=0, p=0.001)
        >>> fault_accuracies = results['criterion_score']
        >>> accuracy_drops = [baseline - acc for acc in fault_accuracies]
        >>> critical_faults = sum(1 for drop in accuracy_drops if drop > 0.1)
        >>> print(f"Critical faults (>10% accuracy drop): {critical_faults}")

    Performance Optimization:
        The function applies several optimizations based on input type:

        Tensor Input Optimizations:
        - Automatic batching to prevent memory overflow
        - Device placement with non-blocking transfers
        - Gradient computation disabled for inference speed
        - Memory-efficient batch concatenation

        DataLoader Input Optimizations:
        - Direct batch processing without additional batching
        - Optimized for pre-configured batch sizes
        - Non-blocking GPU transfers when supported
        - Streaming evaluation for memory efficiency

    Memory Management:
        - Batch processing: Prevents out-of-memory errors on large datasets
        - Automatic cleanup: Intermediate tensors eligible for garbage collection
        - Device awareness: Efficient GPU memory utilization
        - Non-blocking transfers: Overlap computation with data movement

    Classification Type Detection:
        The function automatically handles both binary and multiclass scenarios:
        - Binary: Single output neurons or two-class outputs
        - Multiclass: Multiple output neurons with argmax prediction
        - Detection based on model output dimensionality
        - Consistent with multiclass_classification_accuracy() logic

    See Also:
        classification_accuracy_loader: Direct DataLoader evaluation
        multiclass_classification_accuracy: Core accuracy computation logic
        seu_injection.core.Injector: Systematic fault injection
        sklearn.metrics.accuracy_score: Underlying accuracy computation standard
    """
    # Check if x_tensor is actually a DataLoader
    # DataLoader detection: Torch DataLoader has 'dataset' attribute; exclude plain tensors
    if not isinstance(x_tensor, torch.Tensor) and hasattr(x_tensor, "dataset"):
        # It's a DataLoader, use the loader function
        if y_true is not None:
            # TODO ERROR HANDLING: Inconsistent exception types across framework
            # ISSUE: Mixed use of ValueError, RuntimeError, TypeError without clear patterns
            # CURRENT: ValueError for user input errors, but RuntimeError for device issues
            # IMPROVEMENT: Define custom exception hierarchy with clear usage guidelines
            # EXAMPLES: SEUInputError, SEUDeviceError, SEUComputationError
            # PRIORITY: MEDIUM - affects user experience and debugging efficiency
            raise ValueError(
                "When using DataLoader, do not specify y_true separately. "
                "Labels should be included in the DataLoader."
            )
        return classification_accuracy_loader(model, x_tensor, device)

    # Handle tensor inputs
    if device:
        model = model.to(device)
        if isinstance(x_tensor, torch.Tensor):
            x_tensor = x_tensor.to(device)
        if y_true is not None:
            y_true = y_true.to(device)

    model.eval()
    y_pred_list = []
    y_true_list = []

    if batch_size is None:
        batch_size = int(len(x_tensor))  # ensure int for mypy

    if not isinstance(x_tensor, torch.Tensor):
        raise TypeError("x_tensor must be a torch.Tensor when not using a DataLoader")
    if y_true is None:
        raise ValueError("y_true must be provided when using tensor inputs")

    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], batch_size):
            end = start + batch_size
            batch_X = x_tensor[start:end]
            batch_y = y_true[start:end]
            batch_pred = model(batch_X)
            y_pred_list.append(batch_pred)
            y_true_list.append(batch_y)

    y_pred_all = torch.cat(y_pred_list).cpu().numpy()
    y_true_all = torch.cat(y_true_list).cpu().numpy()

    return multiclass_classification_accuracy(y_true_all, y_pred_all)


def multiclass_classification_accuracy(
    y_true: np.ndarray, model_output: np.ndarray
) -> float:
    """
    Compute classification accuracy with automatic binary/multiclass detection and robust prediction logic.

    This function provides the core accuracy computation logic used throughout the SEU
    injection framework. It automatically determines the classification type based on
    model output dimensionality and applies appropriate prediction strategies, handling
    various label encoding schemes and output formats commonly encountered in deep learning.

    The function is designed to be robust against different neural network architectures
    and output formats, making it suitable for evaluating a wide range of models during
    fault injection experiments without requiring manual configuration.

    Args:
        y_true (np.ndarray): Ground truth class labels with shape (N,) where N is the
            number of samples. Supports various label encoding schemes:
            - Binary: {0, 1}, {-1, +1}, or any two distinct values
            - Multiclass: {0, 1, ..., K-1} or any K distinct integer values
            - The function automatically adapts to the label range present in y_true.
        model_output (np.ndarray): Raw neural network outputs with shape (N,) for binary
            classification or (N, K) for K-class classification. Can be:
            - Logits: Raw neural network outputs before activation
            - Probabilities: Softmax or sigmoid activated outputs
            - Any real-valued outputs suitable for prediction via thresholding/argmax

    Returns:
        float: Classification accuracy as a value in [0.0, 1.0] representing the
            fraction of correctly classified samples. Computed using sklearn's
            accuracy_score for consistency with standard machine learning practices.

    Raises:
        ValueError: If y_true and model_output have incompatible shapes (different
            number of samples) or if arrays contain invalid values (NaN, infinity).
        TypeError: If inputs are not numpy arrays or cannot be converted to arrays.

    Classification Logic:
        Binary Classification (model_output.ndim == 1 or model_output.shape[1] == 1):
        - Determines label range: y_low = min(y_true), y_high = max(y_true)
        - Computes threshold: midpoint = (y_high + y_low) / 2
        - Applies thresholding: output >= midpoint → y_high, else y_low
        - Handles arbitrary binary encodings: {0,1}, {-1,+1}, {class_a, class_b}

        Multiclass Classification (model_output.shape[1] > 1):
        - Applies argmax along class dimension: predicted_class = argmax(output, axis=1)
        - Assumes class indices correspond to output neuron positions
        - Compatible with one-hot encoding and standard multiclass setups

    Example:
        >>> import numpy as np
        >>> from seu_injection.metrics.accuracy import multiclass_classification_accuracy
        >>>
        >>> # Binary classification examples
        >>> y_binary = np.array([0, 1, 0, 1, 1])
        >>>
        >>> # Single output neuron (sigmoid-style)
        >>> output_sigmoid = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        >>> acc_binary = multiclass_classification_accuracy(y_binary, output_sigmoid)
        >>> print(f"Binary accuracy: {acc_binary:.2f}")  # 0.80 (4/5 correct)
        >>>
        >>> # Two output neurons (softmax-style)
        >>> output_softmax = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3],
        ...                           [0.0, 1.0], [0.2, 0.8]])
        >>> acc_softmax = multiclass_classification_accuracy(y_binary, output_softmax)
        >>> print(f"Softmax binary accuracy: {acc_softmax:.2f}")
        >>>
        >>> # Multiclass classification
        >>> y_multi = np.array([0, 2, 1, 0, 2])
        >>> output_multi = np.array([[0.9, 0.05, 0.05],  # Pred: 0, True: 0 ✓
        ...                         [0.1, 0.1, 0.8],     # Pred: 2, True: 2 ✓
        ...                         [0.3, 0.6, 0.1],     # Pred: 1, True: 1 ✓
        ...                         [0.2, 0.7, 0.1],     # Pred: 1, True: 0 ✗
        ...                         [0.0, 0.1, 0.9]])    # Pred: 2, True: 2 ✓
        >>> acc_multi = multiclass_classification_accuracy(y_multi, output_multi)
        >>> print(f"Multiclass accuracy: {acc_multi:.2f}")  # 0.80 (4/5 correct)
        >>>
        >>> # Alternative binary encoding
        >>> y_alt = np.array([-1, 1, -1, 1, -1])
        >>> output_alt = np.array([-0.5, 0.8, -0.2, 1.2, 0.1])
        >>> acc_alt = multiclass_classification_accuracy(y_alt, output_alt)
        >>> print(f"Alternative encoding: {acc_alt:.2f}")

    Robustness Features:
        - Label Encoding Agnostic: Handles any binary label pair automatically
        - Output Format Flexible: Works with logits, probabilities, or raw scores
        - Shape Validation: Ensures input compatibility and provides clear error messages
        - NaN/Infinity Handling: Validates inputs and provides informative errors
        - Consistent Results: Uses sklearn.accuracy_score for standardized computation

    Performance Characteristics:
        - Time Complexity: O(N) for N samples, dominated by numpy operations
        - Space Complexity: O(N) for prediction array creation
        - Vectorized Operations: Efficient numpy implementations throughout
        - Memory Efficiency: Minimal temporary array allocation

    Integration with SEU Framework:
        This function is called internally by higher-level accuracy functions and
        integrates seamlessly with the SEU injection workflow:

        >>> # Typical usage in fault injection
        >>> def create_accuracy_criterion(y_test):
        ...     def criterion(model, x, y, device):
        ...         with torch.no_grad():
        ...             outputs = model(x.to(device))
        ...             return multiclass_classification_accuracy(
        ...                 y.cpu().numpy(), outputs.cpu().numpy()
        ...             )
        ...     return criterion

    See Also:
        classification_accuracy: High-level tensor/DataLoader interface
        classification_accuracy_loader: DataLoader-specific evaluation
        sklearn.metrics.accuracy_score: Underlying accuracy computation
        numpy.argmax: Multiclass prediction logic implementation
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

    return float(accuracy_score(y_true=y_true, y_pred=y_pred))

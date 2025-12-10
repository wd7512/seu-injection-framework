"""Core SEU injection functionality.

This module provides the abstract BaseInjector class and its concrete implementations (ExhaustiveSEUInjector and
StochasticSEUInjector) for systematic and stochastic fault injection in PyTorch neural networks to study robustness in
harsh environments.

# TODO PRODUCTION READINESS: Major architectural improvements needed per PRODUCTION_READINESS_PLAN.md
# WORKING DOCUMENTS CONVERTED TO TODOS (can be archived):
#   - PIPELINE_FIX_URGENT.md -> Converted to TODOs in pyproject.toml, CI workflow, run_tests.py
#   - COVERAGE_FIX_SUMMARY.md -> Implementation complete, coverage threshold fixed
#   - DOCS_REVIEW_COMPLETE.md -> Temporary review file, content extracted
#   - USER_EXPERIENCE_IMPROVEMENT_PLAN.md -> Key items extracted to code TODOs
#   - Parts of PRODUCTION_READINESS_PLAN.md -> Architectural TODOs extracted here
# IMPLEMENTATION PRIORITIES:
# TYPE SAFETY: Add comprehensive type hints throughout class (currently missing)
# API COMPLEXITY: High learning curve - need simplified "quick analysis" functions
# ERROR HANDLING: Limited custom exception types for better user experience
# DEVICE MANAGEMENT: Inconsistent string vs torch.device object handling
# PERFORMANCE: GPU operations could be optimized further for large models
# EXTENSIBILITY: Rigid architecture - need plugin system for custom metrics/strategies
# DOCUMENTATION: Missing comprehensive docstrings for production API
# VALIDATION: Input validation using Pydantic models for better error messages
# PRIORITY: HIGH - Core user-facing API needs enhancement for v1.0 release
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Any, Union

import numpy as np
import torch

from ..bitops import bitflip_float32_optimized


class BaseInjector(ABC):
    """Abstract base class for SEU fault injection in PyTorch models.

    Supports systematic and stochastic bit-flip injection to evaluate model robustness.
    Device-aware operation and flexible evaluation via user-supplied criterion.

    Notes:
        - All injections are reversible; model is unchanged after each run.
        - Use either (x, y) or data_loader for evaluation, not both.
        - Model parameters must be float32 tensors for bit-flip operations.
        - Device is auto-detected if not specified.

    Example:
        >>> injector = ExhaustiveSEUInjector(model, criterion, x=x, y=y)
        >>> results = injector.run_injector(bit_i=15)
        >>> print(injector.baseline_score)

    """

    def __init__(
        self,
        trained_model: torch.nn.Module,
        criterion: Callable[..., float],
        device: Union[str, torch.device, None] = None,
        x: Union[torch.Tensor, np.ndarray, None] = None,
        y: Union[torch.Tensor, np.ndarray, None] = None,
        data_loader: Union[torch.utils.data.DataLoader, None] = None,
    ) -> None:
        # TODO API COMPLEXITY: Constructor requires too much domain knowledge per improvement plans
        # ISSUES:
        #   - Users must understand criterion functions (no sensible defaults)
        #   - Device management confusing (string vs torch.device)
        #   - Mutually exclusive x/y vs data_loader not clear from signature
        #   - No convenience constructors for common scenarios
        # IMPROVEMENTS NEEDED:
        #   - Add ExhaustiveSEUInjector.from_model(model) with classification_accuracy default
        #   - Add ExhaustiveSEUInjector.quick_setup(model, test_data) for simple cases
        #   - Better error messages when x/y and data_loader both provided
        #   - Auto-detect device if None provided (already implemented)
        # PRIORITY: MEDIUM - Affects new user onboarding
        """Initialize injector with model, criterion, device, and data.

        Args:
            trained_model: PyTorch model to inject faults into.
            criterion: Function to evaluate model performance.
            device: Target device ('cpu', 'cuda', etc.). Auto-detects if None.
            x: Input data tensor (optional).
            y: Target labels tensor (optional).
            data_loader: DataLoader for evaluation (optional).

        Raises:
            ValueError: If both data_loader and (x, y) are provided, or neither.

        """
        # Device detection and setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model setup
        self.criterion = criterion  # type: ignore[assignment]
        self.model = trained_model.to(self.device)
        self.model.eval()

        # Initialize optional data attributes for type checking
        self.X: Union[torch.Tensor, None] = None
        self.y: Union[torch.Tensor, None] = None

        # Data setup - validate mutually exclusive options
        self.use_data_loader = False

        print(f"Testing a forward pass on {self.device}...")

        if data_loader:
            if x is not None or y is not None:
                raise ValueError(
                    "Cannot pass both a dataloader and x and y values. Use either data_loader OR (x, y), not both."
                )

            self.use_data_loader = True
            self.data_loader = data_loader
            self.baseline_score = criterion(self.model, self.data_loader, device=self.device)

        else:
            # Handle tensor conversion with proper validation
            # Predeclare attributes for mypy (Optional tensors)
            if x is not None:
                if isinstance(x, torch.Tensor):
                    self.X = x.clone().detach().to(device=self.device, dtype=torch.float32)
                else:
                    self.X = torch.tensor(x, dtype=torch.float32, device=self.device)
            if y is not None:
                if isinstance(y, torch.Tensor):
                    self.y = y.clone().detach().to(device=self.device, dtype=torch.float32)
                else:
                    self.y = torch.tensor(y, dtype=torch.float32, device=self.device)

            # Validate that we have valid data
            if self.X is None and self.y is None:
                raise ValueError("Must provide either data_loader or at least one of X, y")

            self.baseline_score = criterion(self.model, self.X, self.y, self.device)

        print(f"Baseline Criterion Score: {self.baseline_score}")

        self._layer_names = [name for name, _ in self.model.named_parameters()]

    def run_injector(self, bit_i: int, layer_name: Union[str, None] = None, **kwargs) -> dict[str, list[Any]]:
        """Run the fault injection process.

        Args:
            bit_i (int): Bit position to flip (0-31).
            layer_name (Optional[str]): Name of the layer to target (None for all layers).
            **kwargs: Additional arguments for the injection process.

        Returns:
            dict[str, list[Any]]: Results of the injection process, including affected tensor locations and scores.

        Raises:
            ValueError: If `bit_i` is out of range or `layer_name` is invalid.
        """
        if bit_i not in range(33):
            raise ValueError(f"bit_i must be in [0, 32], got {bit_i}")

        if layer_name is not None and layer_name not in self._layer_names:
            print(f"WARNING - layer '{layer_name}' missing. Skipping...")

        self.model.eval()
        return self._run_injector_impl(bit_i, layer_name, **kwargs)

    @abstractmethod
    def _run_injector_impl(self, bit_i: int, layer_name: Union[str, None], **kwargs) -> dict[str, list[Any]]: ...

    def _get_criterion_score(self) -> float:
        """Evaluate model performance using the configured criterion.

        Returns:
            float: Current model score.

        """
        if self.use_data_loader:
            return float(self.criterion(self.model, self.data_loader, device=self.device))
        else:
            return float(self.criterion(self.model, self.X, self.y, device=self.device))

    def _iterate_layers(
        self, layer_name: Union[str, None]
    ) -> Generator[tuple[str, torch.nn.Parameter], None, None]:
        """Iterate through model layers, optionally filtering by name.

        Args:
            layer_name: Name of specific layer to target (None for all layers).

        Yields:
            tuple: (layer_name, parameter_tensor) pairs.

        """
        for current_layer_name, tensor in self.model.named_parameters():
            # Skip layer if specific layer requested and this isn't it
            if layer_name and layer_name != current_layer_name:
                continue
            yield current_layer_name, tensor

    def _prepare_tensor_for_injection(self, tensor: torch.nn.Parameter) -> tuple[torch.Tensor, np.ndarray]:
        """Prepare a tensor for injection by cloning and converting to numpy.

        Args:
            tensor: The parameter tensor to prepare.

        Returns:
            tuple: (original_tensor, tensor_cpu) where original_tensor is a clone
                   and tensor_cpu is a numpy array on CPU.

        """
        original_tensor = tensor.data.clone()
        tensor_cpu = original_tensor.cpu().numpy()
        return original_tensor, tensor_cpu

    def _inject_and_evaluate(
        self,
        tensor: torch.nn.Parameter,
        idx: tuple,
        original_tensor: torch.Tensor,
        original_val: float,
        bit_i: int,
    ) -> tuple[float, float]:
        """Inject a fault at a specific location, evaluate, and restore.

        Args:
            tensor: The parameter tensor to inject into.
            idx: The index location for injection.
            original_tensor: The original tensor values for restoration.
            original_val: The original value at the injection location.
            bit_i: The bit position to flip (0-31).

        Returns:
            tuple: (criterion_score, seu_val) where criterion_score is the model
                   performance after injection and seu_val is the injected value.

        """
        # Perform bitflip
        seu_val = bitflip_float32_optimized(original_val, bit_i, inplace=False)

        # Inject fault
        tensor.data[idx] = torch.tensor(seu_val, device=self.device, dtype=tensor.dtype)

        # Evaluate model
        criterion_score = self._get_criterion_score()

        # Restore original value
        tensor.data[idx] = original_tensor[idx]

        return criterion_score, seu_val

    def _record_injection_result(
        self,
        results: dict[str, list[Any]],
        idx: tuple,
        criterion_score: float,
        layer_name: str,
        original_val: float,
        seu_val: float,
    ) -> None:
        """Record the results of a single injection.

        Args:
            results: The results dictionary to update.
            idx: The index location of the injection.
            criterion_score: The model performance score after injection.
            layer_name: The name of the layer that was injected.
            original_val: The original parameter value.
            seu_val: The value after injection.

        """
        results["tensor_location"].append(idx)
        results["criterion_score"].append(criterion_score)
        results["layer_name"].append(layer_name)
        results["value_before"].append(original_val)
        results["value_after"].append(seu_val)

    def _initialize_results(self) -> dict[str, list[Any]]:
        """Initialize the results dictionary structure.

        Returns:
            dict: Empty results dictionary with required keys.

        """
        return {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": [],
        }

    @abstractmethod
    def _get_injection_indices(self, tensor_shape: tuple, **kwargs) -> np.ndarray:
        """Get the indices where injections should be performed.

        This method defines the injection strategy (exhaustive vs. stochastic).

        Args:
            tensor_shape: The shape of the tensor to inject into.
            **kwargs: Additional strategy-specific parameters.

        Returns:
            np.ndarray: Array of indices where injections should occur.
                       Shape: (N, len(tensor_shape)) where N is the number of injections.

        """
        ...

"""
Core SEU injection functionality.

This module provides the abstract BaseInjector class and its concrete implementations (ExhaustiveSEUInjector and StochasticSEUInjector) for systematic and stochastic fault injection
in PyTorch neural networks to study robustness in harsh environments.

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
from typing import Any, Callable, Optional, Union

import numpy as np
import torch


class BaseInjector(ABC):
    """
    Injector for PyTorch neural networks.

    This class provides comprehensive fault injection capabilities to study neural network
    robustness under radiation-induced bit flips. It supports both systematic exhaustive
    injection across all model parameters and stochastic probabilistic injection for
    large-scale analysis.

    The injector operates by temporarily modifying model parameters through bit-flip
    operations, evaluating model performance using provided criteria, and collecting
    detailed results for analysis. All injections are reversible, leaving the original
    model unchanged.

    Key Features:
        - Systematic injection across all targetable layers
        - Stochastic sampling for large model analysis
        - Flexible criterion-based evaluation
        - Device-aware operation (CPU/CUDA)
        - Comprehensive result tracking
        - Layer-specific targeting support

    Attributes:
        model (torch.nn.Module): The PyTorch model under test
        criterion (callable): Evaluation function for measuring model performance
        device (torch.device): Computing device (CPU/CUDA)
        baseline_score (float): Model performance without fault injection
        x (torch.Tensor): Input data tensor for evaluation
        y (torch.Tensor): Target labels for evaluation
        data_loader (DataLoader): Alternative data source for evaluation

    Example:
        >>> import torch
        >>> from seu_injection.core import ExhaustiveSEUInjector, StochasticSEUInjector
        >>> from seu_injection.metrics import accuracy_top1
        >>>
        >>> # Setup model and data
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(784, 128),
        ...     torch.nn.ReLU(),
        ...     torch.nn.Linear(128, 10)
        ... )
        >>> x = torch.randn(100, 784)
        >>> y = torch.randint(0, 10, (100,))
        >>>
        >>> # Systematic (exhaustive) injection
        >>> injector = ExhaustiveSEUInjector(
        ...     trained_model=model,
        ...     criterion=accuracy_top1,
        ...     x=x, y=y
        ... )
        >>> results = injector.run_injector(bit_i=15)
        >>> print(f"Baseline accuracy: {injector.baseline_score:.3f}")
        >>> print(f"Injected {len(results['criterion_score'])} faults")
        >>>
        >>> # Stochastic injection
        >>> injector = StochasticSEUInjector(
        ...     trained_model=model,
        ...     criterion=accuracy_top1,
        ...     x=x, y=y
        ... )
        >>> results = injector.run_injector(bit_i=15, p=0.01)
        >>> print(f"Injected {len(results['criterion_score'])} faults (stochastic)")

    See Also:
        ExhaustiveSEUInjector: Systematic fault injection across all parameters
        StochasticSEUInjector: Probabilistic fault injection sampling
        get_criterion_score: Manual criterion evaluation
    """

    def __init__(
        self,
        trained_model: torch.nn.Module,
        criterion: Callable[..., float],
        device: Optional[Union[str, torch.device]] = None,
        x: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y: Optional[Union[torch.Tensor, np.ndarray]] = None,
        data_loader: Optional[torch.utils.data.DataLoader] = None,
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
        """
        Initialize the SEU injector with model, data, and evaluation criterion.

        Sets up the fault injection environment by configuring the target model,
        evaluation data, and performance criterion. Automatically detects optimal
        computing device and establishes baseline performance metrics.

        Args:
            trained_model (torch.nn.Module): PyTorch neural network model to inject
                faults into. Should be pre-trained and in evaluation mode. The model
                parameters must be float32 tensors for bit-flip operations.
            criterion (callable): Function to evaluate model performance after fault
                injection. Should accept (model, x, y, device) and return a numeric
                score. Higher scores typically indicate better performance.
            device (Optional[Union[str, torch.device]]): Computing device for operations.
                Options: 'cpu', 'cuda', 'cuda:0', etc., or torch.device object.
                If None, automatically selects CUDA if available, otherwise CPU.
            x (Optional[Union[torch.Tensor, np.ndarray]]): Input data tensor for model
                evaluation. Shape should match model's expected input. Mutually
                exclusive with data_loader parameter. Will be moved to target device.
            y (Optional[Union[torch.Tensor, np.ndarray]]): Target labels tensor for
                supervised evaluation. Must have same batch size as x. Mutually
                exclusive with data_loader parameter. Will be moved to target device.
                x and y parameters. Useful for large datasets that don't fit in memory.

        Raises:
            ValueError: If both data_loader and (x, y) are provided simultaneously,
                or if neither data source is provided.
            TypeError: If model parameters are not float32 tensors.
            RuntimeError: If CUDA is specified but not available.

        Example:
            >>> # Basic usage with tensor data
            >>> injector = ExhaustiveSEUInjector(
            ...     trained_model=model,
            ...     criterion=accuracy_top1,
            ...     x=test_images,
            ...     y=test_labels
            ... )
            >>>
            >>> # Usage with DataLoader for large datasets
            >>> injector = ExhaustiveSEUInjector(
            ...     trained_model=model,
            ...     criterion=accuracy_top1,
            ...     data_loader=test_loader,
            ...     device='cuda'
            ... )

        Note:
            This implementation assumes float32 precision for IEEE 754 bit manipulation.
            Other precisions (float16, float64) will be supported in future versions.
            The model is automatically moved to the specified device and set to
            evaluation mode for consistent injection results.
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
        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

        # Data setup - validate mutually exclusive options
        self.use_data_loader = False

        print(f"Testing a forward pass on {self.device}...")

        if data_loader:
            if x is not None or y is not None:
                raise ValueError(
                    "Cannot pass both a dataloader and x and y values. "
                    "Use either data_loader OR (x, y), not both."
                )

            self.use_data_loader = True
            self.data_loader = data_loader
            self.baseline_score = criterion(
                self.model, self.data_loader, device=self.device
            )

        else:
            # Handle tensor conversion with proper validation
            # Predeclare attributes for mypy (Optional tensors)
            if x is not None:
                if isinstance(x, torch.Tensor):
                    self.X = (
                        x.clone().detach().to(device=self.device, dtype=torch.float32)
                    )
                else:
                    self.X = torch.tensor(x, dtype=torch.float32, device=self.device)
            if y is not None:
                if isinstance(y, torch.Tensor):
                    self.y = (
                        y.clone().detach().to(device=self.device, dtype=torch.float32)
                    )
                else:
                    self.y = torch.tensor(y, dtype=torch.float32, device=self.device)

            # Validate that we have valid data
            if self.X is None and self.y is None:
                raise ValueError(
                    "Must provide either data_loader or at least one of X, y"
                )

            self.baseline_score = criterion(self.model, self.X, self.y, self.device)

        print(f"Baseline Criterion Score: {self.baseline_score}")

    @abstractmethod
    def run_injector(
        self, bit_i: int, layer_name: Optional[str] = None, **kwargs
    ) -> dict[str, list[Any]]:
        pass

    def _get_criterion_score(self) -> float:
        """
        Evaluate current model performance using the configured criterion function.

        This method provides on-demand evaluation of the model's current state using
        the criterion function specified during initialization. It handles both
        tensor-based and DataLoader-based evaluation automatically based on the
        configured data source.

        The method is used internally during fault injection campaigns but can also
        be called directly for manual performance assessment at any time. It ensures
        consistent evaluation methodology across all injection experiments.

        Returns:
            float: Current model performance score as computed by the criterion function.
                The interpretation depends on the specific criterion:
                - Accuracy metrics: Higher values indicate better performance (0.0-1.0)
                - Loss metrics: Lower values indicate better performance (0.0+)
                - Custom metrics: Interpretation depends on implementation

        Raises:
            RuntimeError: If model evaluation fails due to data/device mismatches,
                insufficient memory, or criterion function errors.
            ValueError: If the criterion function returns non-numeric results.

        Example:
            >>> # Manual evaluation during analysis
            >>> injector = ExhaustiveSEUInjector(model, accuracy_top1, x=data, y=labels)
            >>> baseline = injector.get_criterion_score()
            >>> print(f"Baseline accuracy: {baseline:.3f}")
            >>>
            >>> # Check performance after manual model modifications
            >>> with torch.no_grad():
            ...     model.classifier.weight.data *= 0.5  # Simulate parameter corruption
            >>> corrupted_score = injector.get_criterion_score()
            >>> print(f"Performance drop: {baseline - corrupted_score:.3f}")
            >>>
            >>> # Restore and verify
            >>> with torch.no_grad():
            ...     model.classifier.weight.data *= 2.0  # Restore
            >>> restored_score = injector.get_criterion_score()
            >>> assert abs(restored_score - baseline) < 1e-6

        Performance:
            Computational cost equals one forward pass through the model plus criterion
            evaluation overhead. For typical scenarios:
            - Small models (<1M params): <10ms on GPU
            - Medium models (1-100M params): 10-100ms on GPU
            - Large models (>100M params): 100ms-1s on GPU

            Memory usage scales with batch size and model size, identical to normal
            inference requirements.

        See Also:
            __init__: Criterion function specification and requirements
            ExhaustiveSEUInjector: Systematic injection with automatic evaluation
            StochasticSEUInjector: Statistical injection with automatic evaluation
        """
        if self.use_data_loader:
            return float(
                self.criterion(self.model, self.data_loader, device=self.device)
            )
        else:
            return float(self.criterion(self.model, self.X, self.y, device=self.device))

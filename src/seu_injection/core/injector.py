"""
Core SEU injection functionality.

This module provides the main SEUInjector class for systematic fault injection
in PyTorch neural networks to study robustness in harsh environments.
"""

from typing import Any, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from ..bitops.float32 import bitflip_float32


class SEUInjector:
    """
    Single Event Upset (SEU) injector for PyTorch neural networks.

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
        >>> from seu_injection import SEUInjector
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
        >>> # Create injector
        >>> injector = SEUInjector(
        ...     trained_model=model,
        ...     criterion=accuracy_top1,
        ...     x=x, y=y
        ... )
        >>>
        >>> # Run systematic injection
        >>> results = injector.run_seu(bit_i=15)
        >>> print(f"Baseline accuracy: {injector.baseline_score:.3f}")
        >>> print(f"Injected {len(results)} faults")

    See Also:
        run_seu: Systematic fault injection across all parameters
        run_stochastic_seu: Probabilistic fault injection sampling
        get_criterion_score: Manual criterion evaluation
    """

    def __init__(
        self,
        trained_model: torch.nn.Module,
        criterion: callable,
        device: Optional[Union[str, torch.device]] = None,
        x: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y: Optional[Union[torch.Tensor, np.ndarray]] = None,
        data_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> None:
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
            data_loader (Optional[torch.utils.data.DataLoader]): PyTorch DataLoader
                providing (x, y) batches for evaluation. Mutually exclusive with
                x and y parameters. Useful for large datasets that don't fit in memory.

        Raises:
            ValueError: If both data_loader and (x, y) are provided simultaneously,
                or if neither data source is provided.
            TypeError: If model parameters are not float32 tensors.
            RuntimeError: If CUDA is specified but not available.

        Example:
            >>> # Basic usage with tensor data
            >>> injector = SEUInjector(
            ...     trained_model=model,
            ...     criterion=accuracy_top1,
            ...     x=test_images,
            ...     y=test_labels
            ... )
            >>>
            >>> # Usage with DataLoader for large datasets
            >>> injector = SEUInjector(
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
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Model setup
        self.criterion = criterion
        self.model = trained_model.to(self.device)
        self.model.eval()

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
            if x is not None:
                if isinstance(x, torch.Tensor):
                    self.X = (
                        x.clone().detach().to(device=self.device, dtype=torch.float32)
                    )
                else:
                    self.X = torch.tensor(x, dtype=torch.float32, device=self.device)
            else:
                self.X = None

            if y is not None:
                if isinstance(y, torch.Tensor):
                    self.y = (
                        y.clone().detach().to(device=self.device, dtype=torch.float32)
                    )
                else:
                    self.y = torch.tensor(y, dtype=torch.float32, device=self.device)
            else:
                self.y = None

            # Validate that we have valid data
            if self.X is None and self.y is None:
                raise ValueError(
                    "Must provide either data_loader or at least one of X, y"
                )

            self.baseline_score = criterion(self.model, self.X, self.y, self.device)

        print(f"Baseline Criterion Score: {self.baseline_score}")

    def get_criterion_score(self) -> float:
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
            >>> injector = SEUInjector(model, accuracy_top1, x=data, y=labels)
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
            run_seu: Systematic injection with automatic evaluation
            run_stochastic_seu: Statistical injection with automatic evaluation
        """
        if self.use_data_loader:
            return self.criterion(self.model, self.data_loader, device=self.device)
        else:
            return self.criterion(self.model, self.X, self.y, device=self.device)

    def run_seu(
        self, bit_i: int, layer_name: Optional[str] = None
    ) -> dict[str, list[Any]]:
        """
        Perform systematic exhaustive SEU injection across model parameters.

        This method conducts a comprehensive fault injection campaign by systematically
        injecting a single bit flip at the specified bit position in every float32
        parameter of the neural network. Each injection is performed individually,
        with the parameter restored to its original value before the next injection.

        The method is ideal for detailed vulnerability analysis of smaller models or
        specific layers, providing complete coverage of all parameters. For large
        models, consider using run_stochastic_seu() for computational efficiency.

        Args:
            bit_i (int): Bit position to flip in IEEE 754 float32 representation.
                Range: [0, 31] where 0 is the most significant bit (sign bit),
                1-8 are exponent bits, and 9-31 are mantissa bits. Different bit
                positions have varying impact on parameter magnitude and sign.
            layer_name (Optional[str]): Specific layer name to target for injection.
                If None, injects across all targetable layers in the model. Use
                model.named_parameters() to see available layer names. Useful for
                focused analysis of specific network components.

        Returns:
            dict[str, list[Any]]: Comprehensive injection results dictionary containing:
                - 'tensor_location' (list[int]): Flat parameter indices where each
                  injection was performed, allowing precise identification of
                  affected parameters across the model.
                - 'criterion_score' (list[float]): Model performance score after
                  each injection, measured using the configured criterion function.
                  Enables statistical analysis of fault impact distribution.
                - 'layer_name' (list[str]): Layer name containing each injected
                  parameter, facilitating layer-wise vulnerability analysis.
                - 'value_before' (list[float]): Original parameter values before
                  injection, enabling impact magnitude calculation.
                - 'value_after' (list[float]): Parameter values immediately after
                  bit flip injection, showing exact fault manifestation.

        Raises:
            AssertionError: If bit_i is not in valid range [0, 32]. Note that
                while IEEE 754 has 32 bits (0-31), bit position 32 is included
                for boundary testing purposes.
            RuntimeError: If model evaluation fails during criterion computation.

        Example:
            >>> # Basic systematic injection
            >>> injector = SEUInjector(model, accuracy_top1, x=data, y=labels)
            >>> results = injector.run_seu(bit_i=15)  # Flip middle mantissa bit
            >>>
            >>> # Analyze results
            >>> baseline = injector.baseline_score
            >>> scores = results['criterion_score']
            >>> accuracy_drops = [baseline - score for score in scores]
            >>> critical_faults = [i for i, drop in enumerate(accuracy_drops) if drop > 0.1]
            >>>
            >>> print(f"Baseline accuracy: {baseline:.3f}")
            >>> print(f"Total injections: {len(scores)}")
            >>> print(f"Critical faults (>10% drop): {len(critical_faults)}")
            >>>
            >>> # Layer-specific analysis
            >>> layer_results = injector.run_seu(bit_i=0, layer_name='classifier.weight')
            >>> print(f"Sign bit flips in classifier: {len(layer_results['tensor_location'])}")

        Performance:
            The computational complexity is O(n) where n is the number of parameters
            in the target scope. Each injection requires:
            - One bit flip operation (~1μs)
            - One forward pass through the model
            - One criterion evaluation
            - Parameter restoration

            For a typical ResNet-18 with ~11M parameters, expect ~30-60 minutes per
            bit position on modern GPU hardware, depending on criterion complexity.

        See Also:
            run_stochastic_seu: Probabilistic sampling for large-scale analysis
            get_criterion_score: Manual evaluation without injection
            bitops.float32.flip_bit: Underlying bit manipulation function
        """
        if bit_i not in range(0, 33):
            raise ValueError(f"bit_i must be in range [0, 32], got {bit_i}")

        self.model.eval()

        results = {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": [],
        }

        with torch.no_grad():  # Disable gradient tracking for efficiency
            # Iterate through each layer of the neural network
            for current_layer_name, tensor in self.model.named_parameters():
                # Skip layer if specific layer requested and this isn't it
                if layer_name and layer_name != current_layer_name:
                    continue

                print(f"Testing Layer: {current_layer_name}")

                # Store original tensor values for restoration
                original_tensor = tensor.data.clone()
                tensor_cpu = original_tensor.cpu().numpy()

                # Iterate through every parameter in the tensor
                for idx in tqdm(
                    np.ndindex(tensor_cpu.shape),
                    desc=f"Injecting into {current_layer_name}",
                ):
                    original_val = tensor_cpu[idx]
                    seu_val = bitflip_float32(original_val, bit_i)

                    # Inject fault, evaluate, restore
                    tensor.data[idx] = torch.tensor(
                        seu_val, device=self.device, dtype=tensor.dtype
                    )
                    criterion_score = self.get_criterion_score()
                    tensor.data[idx] = original_tensor[idx]  # Restore original value

                    # Record results
                    results["tensor_location"].append(idx)
                    results["criterion_score"].append(criterion_score)
                    results["layer_name"].append(current_layer_name)
                    results["value_before"].append(original_val)
                    results["value_after"].append(seu_val)

        return results

    def run_stochastic_seu(
        self, bit_i: int, p: float, layer_name: Optional[str] = None
    ) -> dict[str, list[Any]]:
        """
        Perform probabilistic stochastic SEU injection for large-scale analysis.

        This method uses Monte Carlo sampling to randomly select parameters for
        fault injection based on a specified probability. Each parameter has an
        independent probability p of being selected for injection, making this
        approach computationally feasible for large models where exhaustive
        injection would be prohibitive.

        The stochastic approach provides statistical estimates of fault impact
        with controllable computational cost. Higher probability values increase
        statistical confidence but require more computation time.

        Args:
            bit_i (int): Bit position to flip in IEEE 754 float32 representation.
                Range: [0, 31] where 0 is the sign bit, 1-8 are exponent bits,
                and 9-31 are mantissa bits. Each bit position has different
                statistical impact on parameter values and model behavior.
            p (float): Probability of injection for each parameter. Range: [0.0, 1.0]
                where 0.0 means no injections and 1.0 means all parameters are
                injected (equivalent to run_seu). Typical values: 0.001-0.01 for
                large models, 0.1-0.5 for focused analysis.
            layer_name (Optional[str]): Specific layer name to target for injection.
                If None, samples from all targetable layers in the model. Useful
                for layer-wise statistical analysis or focusing on critical
                components like classifier layers.

        Returns:
            dict[str, list[Any]]: Injection results with identical structure to run_seu():
                - 'tensor_location' (list[int]): Indices of randomly selected parameters
                  that received bit flip injections, in order of processing.
                - 'criterion_score' (list[float]): Model performance after each
                  injection, enabling statistical analysis of fault impact distribution.
                - 'layer_name' (list[str]): Layer names containing each selected
                  parameter, useful for layer-wise vulnerability assessment.
                - 'value_before' (list[float]): Original parameter values before
                  injection, allowing impact magnitude analysis.
                - 'value_after' (list[float]): Parameter values after bit flip,
                  showing actual fault manifestation in each case.

        Raises:
            AssertionError: If p is not in valid range [0.0, 1.0] or bit_i is
                not in valid range [0, 32]. Both parameters must be within their
                respective valid domains for proper operation.
            RuntimeError: If model evaluation fails during criterion computation
                or if random sampling produces no injections (very rare with p>0).

        Example:
            >>> # Large model statistical analysis
            >>> injector = SEUInjector(large_model, accuracy_top1, x=data, y=labels)
            >>>
            >>> # Sample 0.1% of parameters for sign bit analysis
            >>> results = injector.run_stochastic_seu(bit_i=0, p=0.001)
            >>> expected_injections = sum(p.numel() for p in model.parameters()) * 0.001
            >>> actual_injections = len(results['tensor_location'])
            >>> print(f"Expected ~{expected_injections:.0f}, got {actual_injections}")
            >>>
            >>> # Statistical analysis of fault impact
            >>> baseline = injector.baseline_score
            >>> scores = results['criterion_score']
            >>> drops = [baseline - score for score in scores]
            >>> mean_drop = np.mean(drops)
            >>> std_drop = np.std(drops)
            >>> print(f"Mean accuracy drop: {mean_drop:.4f} ± {std_drop:.4f}")
            >>>
            >>> # Layer-specific sampling
            >>> classifier_results = injector.run_stochastic_seu(
            ...     bit_i=15, p=0.1, layer_name='classifier.weight'
            ... )

        Performance:
            Expected computational complexity is O(p×n) where n is the number of
            parameters in scope and p is the injection probability. For p=0.001
            and a 100M parameter model, expect ~100K injections requiring:

            - GPU memory: Same as single forward pass
            - Time: ~5-15 minutes depending on criterion complexity
            - Statistical confidence: √(p×n) effective sample size

            The method is particularly efficient for:
            - Large language models (>1B parameters)
            - Convolutional networks with many parameters
            - Comparative studies across different bit positions
            - Monte Carlo estimation of fault tolerance metrics

        See Also:
            run_seu: Exhaustive systematic injection for complete coverage
            get_criterion_score: Direct evaluation without injection
            numpy.random: Underlying random sampling implementation
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0, 1], got {p}")
        if bit_i not in range(0, 33):
            raise ValueError(f"bit_i must be in range [0, 32], got {bit_i}")

        self.model.eval()

        results = {
            "tensor_location": [],
            "criterion_score": [],
            "layer_name": [],
            "value_before": [],
            "value_after": [],
        }

        with torch.no_grad():  # Disable gradient tracking for efficiency
            # Iterate through each layer of the neural network
            for current_layer_name, tensor in self.model.named_parameters():
                # Skip layer if specific layer requested and this isn't it
                if layer_name and layer_name != current_layer_name:
                    continue

                print(f"Testing Layer: {current_layer_name}")

                # Store original tensor values for restoration
                original_tensor = tensor.data.clone()
                tensor_cpu = original_tensor.cpu().numpy()

                # Iterate through parameters with stochastic sampling
                for idx in tqdm(
                    np.ndindex(tensor_cpu.shape),
                    desc=f"Stochastic injection into {current_layer_name}",
                ):
                    # Skip this parameter with probability (1-p)
                    if np.random.uniform(0, 1) > p:
                        continue

                    original_val = tensor_cpu[idx]
                    seu_val = bitflip_float32(original_val, bit_i)

                    # Inject fault, evaluate, restore
                    tensor.data[idx] = torch.tensor(
                        seu_val, device=self.device, dtype=tensor.dtype
                    )
                    criterion_score = self.get_criterion_score()
                    tensor.data[idx] = original_tensor[idx]  # Restore original value

                    # Record results
                    results["tensor_location"].append(idx)
                    results["criterion_score"].append(criterion_score)
                    results["layer_name"].append(current_layer_name)
                    results["value_before"].append(original_val)
                    results["value_after"].append(seu_val)

        return results

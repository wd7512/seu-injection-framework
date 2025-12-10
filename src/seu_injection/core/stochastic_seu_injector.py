"""Stochastic SEU Injector Module.

This module provides the `StochasticSEUInjector` class, which performs random bit flips in model parameters to evaluate
statistical robustness under fault injection scenarios.
"""

from typing import Any, Union

import numpy as np
import torch
from tqdm import tqdm

from .base_injector import BaseInjector


class StochasticSEUInjector(BaseInjector):
    """Stochastic SEU injector for PyTorch models.

    Randomly flips bits in float32 weights across all layers (or a specified layer),
    evaluating model performance after each injection.

    Notes:
        - Use for statistical fault analysis in large models.
        - Injection probability p controls sample size and efficiency.
        - All injections are reversible; model is restored after each run.

    Example:
        >>> injector = StochasticSEUInjector(model, criterion, x=data, y=labels)
        >>> results = injector.run_injector(bit_i=15, p=0.01)
        >>> print(len(results['criterion_score']))

    """

    def _get_injection_indices(self, tensor_shape: tuple, **kwargs) -> np.ndarray:
        """Get stochastically selected indices for injection.

        Args:
            tensor_shape: Shape of the tensor to inject into.
            **kwargs: Must include 'p' (probability) and optionally 'run_at_least_one_injection'.

        Returns:
            np.ndarray: Randomly selected indices based on probability p.

        Raises:
            ValueError: If p is not in [0.0, 1.0].

        """
        p = kwargs.get("p", 0.0)
        run_at_least_one_injection = kwargs.get("run_at_least_one_injection", True)

        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0, 1], got {p}")

        # Build a boolean mask for stochastic selection
        injection_mask = np.random.random(tensor_shape) < p

        # Check if at least one injection will occur
        if run_at_least_one_injection and not injection_mask.any() and np.prod(tensor_shape) > 0:
            # If no injections selected and we need at least one, pick one randomly
            random_idx = tuple(np.random.randint(0, dim) for dim in tensor_shape)
            injection_mask[random_idx] = True

        # Get indices where injections should occur
        return np.argwhere(injection_mask)

    def _run_injector_impl(self, bit_i: int, layer_name: Union[str, None] = None, **kwargs) -> dict[str, list[Any]]:
        """Randomly inject faults into model parameters using probability p.

        Args:
            bit_i (int): Bit position to flip (0-31).
            layer_name (Optional[str]): Layer to target (None for all).
            p (float, via kwargs): Probability of injection for each parameter (0.0-1.0).
            run_at_least_one_injection (bool, via kwargs): If True, ensure at least one injection per layer
                even if p is very small. Default is True to prevent empty results in smoke tests.

        Returns:
            dict[str, list[Any]]: Results including tensor locations, scores, layer names, values before/after.

        Raises:
            ValueError: If p is not in [0.0, 1.0] or bit_i is not in [0, 32].
            RuntimeError: If model evaluation fails.

        Notes:
            - Efficient for large models and statistical analysis.
            - All injections are reversible; model is restored after each run.

        """
        results = self._initialize_results()

        with torch.no_grad():  # Disable gradient tracking for efficiency
            # Iterate through each layer of the neural network
            for current_layer_name, tensor in self._iterate_layers(layer_name):
                print(f"Testing Layer: {current_layer_name}")

                # Prepare tensor for injection
                original_tensor, tensor_cpu = self._prepare_tensor_for_injection(tensor)

                # Get indices for injection (stochastic strategy)
                injection_indices = self._get_injection_indices(tensor_cpu.shape, **kwargs)

                # Perform injections for selected indices
                for idx_array in tqdm(
                    injection_indices,
                    desc=f"Stochastic injection into {current_layer_name}",
                ):
                    idx = tuple(idx_array)
                    original_val = tensor_cpu[idx]

                    # Inject fault, evaluate, and restore
                    criterion_score, seu_val = self._inject_and_evaluate(
                        tensor, idx, original_tensor, original_val, bit_i
                    )

                    # Record results
                    self._record_injection_result(
                        results, idx, criterion_score, current_layer_name, original_val, seu_val
                    )

        return results

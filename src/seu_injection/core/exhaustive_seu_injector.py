"""Exhaustive SEU Injector Module.

This module provides the `ExhaustiveSEUInjector` class, which systematically flips bits in model parameters to evaluate
robustness under exhaustive fault injection scenarios.
"""

from typing import Any, Union

import numpy as np
import torch
from tqdm import tqdm

from .base_injector import BaseInjector


class ExhaustiveSEUInjector(BaseInjector):
    """Exhaustive SEU injector for PyTorch models.

    Systematically flips each bit in float32 weights across all layers (or a specified layer),
    evaluating model performance after each injection.

    Notes:
        - Use for detailed vulnerability analysis of small models or specific layers.
        - For large models, use StochasticSEUInjector for efficiency.
        - All injections are reversible; model is restored after each run.

    Example:
        >>> injector = ExhaustiveSEUInjector(model, criterion, x=data, y=labels)
        >>> results = injector.run_injector(bit_i=15)
        >>> print(len(results['criterion_score']))

    """

    def _get_injection_indices(self, tensor_shape: tuple, **kwargs) -> np.ndarray:
        """Get all indices for exhaustive injection.

        Args:
            tensor_shape: Shape of the tensor to inject into.
            **kwargs: Unused for exhaustive strategy.

        Returns:
            np.ndarray: All possible indices in the tensor.

        """
        # Generate all indices exhaustively
        all_indices = list(np.ndindex(tensor_shape))
        return np.array([idx for idx in all_indices], dtype=object)

    def _run_injector_impl(self, bit_i: int, layer_name: Union[str, None] = None, **kwargs) -> dict[str, list[Any]]:
        """Perform systematic SEU injection across model parameters.

        Flips a single bit at the specified position in every float32 parameter of the model (or a specific layer),
        evaluates the model, and restores the original value.

        Args:
            bit_i (int): Bit position to flip (0-31).
            layer_name (Optional[str]): Layer to target (None for all).

        Returns:
            dict[str, list[Any]]: Results including tensor locations, scores, layer names, values before/after.

        Raises:
            AssertionError: If bit_i is not in [0, 32].
            RuntimeError: If model evaluation fails.

        Notes:
            - For large models, this is computationally expensive.
            - All injections are reversible; model is restored after each run.

        """
        results = self._initialize_results()

        with torch.no_grad():  # Disable gradient tracking for efficiency
            # Iterate through each layer of the neural network
            for current_layer_name, tensor in self._iterate_layers(layer_name):
                print(f"Testing Layer: {current_layer_name}")

                # Prepare tensor for injection
                original_tensor, tensor_cpu = self._prepare_tensor_for_injection(tensor)

                # Get indices for injection (exhaustive strategy)
                injection_indices = self._get_injection_indices(tensor_cpu.shape, **kwargs)

                # Perform injections for all indices
                for idx in tqdm(
                    injection_indices,
                    desc=f"Injecting into {current_layer_name}",
                ):
                    # Convert to tuple if needed
                    if not isinstance(idx, tuple):
                        idx = tuple(idx)

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

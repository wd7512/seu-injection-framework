"""Stochastic SEU Injector Module.

This module provides the `StochasticSEUInjector` class, which performs random bit flips in model parameters to evaluate
statistical robustness under fault injection scenarios.
"""

import numpy as np

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

        # Use per-instance RNG for reproducibility, fall back to local if none set
        rng = self._rng if self._rng is not None else np.random.default_rng()

        # Build a boolean mask for stochastic selection
        injection_mask = rng.random(tensor_shape) < p

        # Check if at least one injection will occur
        if run_at_least_one_injection and not injection_mask.any() and np.prod(tensor_shape) > 0:
            # If no injections selected and we need at least one, pick one randomly
            random_idx = tuple(rng.integers(0, dim) for dim in tensor_shape)
            injection_mask[random_idx] = True

        # Get indices where injections should occur
        return np.argwhere(injection_mask)

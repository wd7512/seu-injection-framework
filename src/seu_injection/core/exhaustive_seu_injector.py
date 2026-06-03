"""Exhaustive SEU Injector Module.

This module provides the `ExhaustiveSEUInjector` class, which systematically flips bits in model parameters to evaluate
robustness under exhaustive fault injection scenarios.
"""

import warnings

import numpy as np

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
            **kwargs: Unused for exhaustive strategy. If provided, a warning is issued.

        Returns:
            np.ndarray: All possible indices in the tensor.
                       Shape: (N, len(tensor_shape)).

        Notes:
            Uses ``np.argwhere(np.ones(...))`` which materialises all indices
            in memory (O(N) on total parameter count — ~176 MB for ResNet-18).
            For extreme models (>100M params) consider returning ``np.ndindex(tensor_shape)``
            instead for O(1) memory, at the cost of: (a) changing the return type to
            ``Iterable[tuple]`` (weaker contract), (b) losing tqdm progress info without
            an explicit ``total=``, and (c) inconsistent return types across strategies
            (stochastic is always eager).

        """
        if kwargs:
            warnings.warn(
                f"ExhaustiveSEUInjector ignores extra kwargs: {set(kwargs.keys())}. "
                f"These parameters are only used by StochasticSEUInjector.",
                UserWarning,
                stacklevel=2,
            )
        # Build exhaustive indices without materializing intermediate Python list
        # Using argwhere(ones(...)) is O(N) memory but avoids O(N) tuple overhead
        return np.argwhere(np.ones(tensor_shape, dtype=bool))

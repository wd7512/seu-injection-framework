"""
Core SEU injection functionality.

This module provides the main SEUInjector class for systematic fault injection
in PyTorch neural networks to study robustness in harsh environments.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from ..bitops.float32 import bitflip_float32


class SEUInjector:
    """
    Single Event Upset (SEU) injector for PyTorch neural networks.
    
    This class provides systematic and stochastic fault injection capabilities
    to study neural network robustness under radiation-induced bit flips.
    
    Attributes:
        model: The PyTorch model under test
        criterion: Evaluation function for measuring model performance
        device: Computing device (CPU/CUDA)
        baseline_score: Model performance without fault injection
    """

    def __init__(
        self,
        trained_model: torch.nn.Module,
        criterion: callable,
        device: Optional[Union[str, torch.device]] = None,
        X: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y: Optional[Union[torch.Tensor, np.ndarray]] = None,
        data_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> None:
        """
        Initialize the SEU injector.
        
        Args:
            trained_model: PyTorch model to inject faults into
            criterion: Function to evaluate model performance
            device: Computing device ('cpu', 'cuda', or torch.device)
            X: Input data tensor (mutually exclusive with data_loader)
            y: Target labels tensor (mutually exclusive with data_loader)  
            data_loader: PyTorch DataLoader (mutually exclusive with X, y)
            
        Raises:
            ValueError: If both data_loader and X/y are provided
            
        Note:
            This implementation assumes float32 precision. Other precisions
            will be supported in future versions.
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
            if X is not None or y is not None:
                raise ValueError(
                    "Cannot pass both a dataloader and X and y values. "
                    "Use either data_loader OR (X, y), not both."
                )

            self.use_data_loader = True
            self.data_loader = data_loader
            self.baseline_score = criterion(self.model, self.data_loader, device=self.device)

        else:
            # Handle tensor conversion with proper validation
            if X is not None:
                if isinstance(X, torch.Tensor):
                    self.X = X.clone().detach().to(device=self.device, dtype=torch.float32)
                else:
                    self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
            else:
                self.X = None

            if y is not None:
                if isinstance(y, torch.Tensor):
                    self.y = y.clone().detach().to(device=self.device, dtype=torch.float32)
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
        Evaluate current model performance using the configured criterion.
        
        Returns:
            Current criterion score (e.g., accuracy, loss)
        """
        if self.use_data_loader:
            return self.criterion(self.model, self.data_loader, device=self.device)
        else:
            return self.criterion(self.model, self.X, self.y, device=self.device)

    def run_seu(
        self,
        bit_i: int,
        layer_name: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Perform exhaustive SEU injection across model parameters.
        
        This method systematically injects a bit flip at the specified bit position
        in every parameter of the neural network (or specified layer).
        
        Args:
            bit_i: Bit position to flip (0-31, where 0 is MSB)
            layer_name: Optional layer name to target (None for all layers)
            
        Returns:
            Dictionary containing injection results:
                - tensor_location: Parameter indices where injections occurred
                - criterion_score: Performance after each injection
                - layer_name: Name of layer containing each parameter
                - value_before: Original parameter values
                - value_after: Parameter values after bit flip
                
        Raises:
            AssertionError: If bit_i is not in valid range [0, 32]
        """
        assert bit_i in range(0, 33), f"bit_i must be in range [0, 32], got {bit_i}"

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
                for idx in tqdm(np.ndindex(tensor_cpu.shape),
                               desc=f"Injecting into {current_layer_name}"):

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
        self,
        bit_i: int,
        p: float,
        layer_name: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Perform stochastic SEU injection with probability sampling.
        
        This method randomly samples parameters for fault injection based on
        probability p, making it suitable for large models where exhaustive
        injection would be computationally prohibitive.
        
        Args:
            bit_i: Bit position to flip (0-31, where 0 is MSB)
            p: Probability of injection for each parameter [0.0, 1.0]
            layer_name: Optional layer name to target (None for all layers)
            
        Returns:
            Dictionary containing injection results (same format as run_seu)
            
        Raises:
            AssertionError: If p is not in valid range [0, 1] or bit_i invalid
        """
        assert 0.0 <= p <= 1.0, f"Probability p must be in [0, 1], got {p}"
        assert bit_i in range(0, 33), f"bit_i must be in range [0, 32], got {bit_i}"

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
                for idx in tqdm(np.ndindex(tensor_cpu.shape),
                               desc=f"Stochastic injection into {current_layer_name}"):

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

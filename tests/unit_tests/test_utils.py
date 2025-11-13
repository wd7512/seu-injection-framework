"""
Basic tests for utils module to improve coverage.
"""

import pytest

from seu_injection.utils.device import detect_device, ensure_tensor


class TestUtilsModule:
    """Basic tests for utils functionality."""

    def test_utils_import(self):
        """Test that utils module can be imported."""
        # This test simply ensures the __init__.py file is executed
        # and any imports work correctly

        # The import should have been done at module level
        # This test just validates no import errors occurred
        assert True  # If we get here, imports worked

    def test_utils_module_structure(self):
        """Test utils module structure."""
        import seu_injection.utils as utils_module

        # TODO TEST QUALITY: Weak assertion patterns reduce test value
        # ISSUE: Tests use trivial assertions that don't validate functionality
        # CURRENT: assert utils_module is not None, assert hasattr(utils_module, "__name__")
        # PROBLEM: These assertions would be true for any Python module
        # IMPROVEMENT: Test actual utils functionality, expected exports, API contracts
        # PRIORITY: MEDIUM - affects test suite quality and coverage meaningfulness

        # Module should exist
        assert utils_module is not None

        # Should have expected attributes (even if empty)
        assert hasattr(utils_module, "__name__")

    def test_detect_device_cpu(self):
        """Test detect_device function with CPU."""
        import torch

        # Test default device detection
        device = detect_device()
        assert isinstance(device, torch.device)

        # Test explicit CPU device
        cpu_device = detect_device("cpu")
        assert cpu_device.type == "cpu"

        # Test torch.device input
        torch_device = detect_device(torch.device("cpu"))
        assert torch_device.type == "cpu"

    def test_ensure_tensor_functionality(self):
        """Test ensure_tensor function."""
        import numpy as np
        import torch

        # Test with torch tensor input
        tensor_input = torch.tensor([1.0, 2.0, 3.0])
        result = ensure_tensor(tensor_input)
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, tensor_input)

        # Test with numpy array input
        numpy_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = ensure_tensor(numpy_input)
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor(numpy_input))

        # Test with list input
        list_input = [1.0, 2.0, 3.0]
        result = ensure_tensor(list_input)
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor(list_input))

    def test_ensure_tensor_device_and_dtype(self):
        """Test ensure_tensor with specific device and dtype."""
        import torch

        # Test with specific dtype
        input_data = [1, 2, 3]  # integers
        result = ensure_tensor(input_data, dtype=torch.float64)
        assert result.dtype == torch.float64

        # Test with specific device (CPU)
        cpu_device = torch.device("cpu")
        result = ensure_tensor(input_data, device=cpu_device)
        assert result.device.type == "cpu"

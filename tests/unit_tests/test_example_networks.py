"""
Additional tests for tests/fixtures/example_networks.py to complete coverage.

These tests focus on error handling and edge cases not covered by integration tests.
"""

import pytest
import torch

from tests.fixtures.example_networks import get_example_network


class TestExampleNetworks:
    """Test suite for example networks module to complete coverage."""

    def test_unsupported_network_type(self):
        """Test error handling for unsupported network types."""
        with pytest.raises(ValueError, match="Network 'invalid' not implemented"):
            get_example_network(net_name="invalid")

    def test_get_example_network_without_training(self):
        """Test getting untrained networks."""
        model, X_train, X_test, y_train, y_test, train_fn, eval_fn = (
            get_example_network(net_name="nn", train=False)
        )

        assert model is not None
        assert isinstance(model, torch.nn.Module)
        assert X_train.shape[1] == 2  # 2D input for moons dataset
        assert len(X_train) > len(X_test)  # More training than test data
        assert callable(train_fn)
        assert callable(eval_fn)

    def test_get_example_network_with_training(self):
        """Test getting trained networks (quick training for test)."""
        model, X_train, X_test, y_train, y_test, train_fn, eval_fn = (
            get_example_network(
                net_name="nn",
                train=True,
                epochs=1,  # Quick training
            )
        )

        assert model is not None
        # Model should have been trained (parameters changed from initialization)
        # We can't easily test this directly, but we can test it doesn't error
        accuracy = eval_fn(model, X_test, y_test)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_different_data_split_params(self):
        """Test different data splitting parameters."""
        model, X_train, X_test, y_train, y_test, _, _ = get_example_network(
            net_name="nn", test_size=0.2, random_state=42
        )

        # With test_size=0.2, we should have 80% training, 20% test
        total_samples = len(X_train) + len(X_test)
        expected_test_size = int(total_samples * 0.2)

        # Allow some tolerance due to rounding
        assert abs(len(X_test) - expected_test_size) <= 1

    def test_all_supported_network_types(self):
        """Test all supported network types can be created."""
        supported_types = ["nn", "cnn", "rnn"]

        for net_type in supported_types:
            model, _, _, _, _, _, _ = get_example_network(net_name=net_type)
            assert model is not None
            assert isinstance(model, torch.nn.Module)

    def test_network_forward_pass(self):
        """Test that networks can perform forward passes."""
        model, X_train, _, _, _, _, _ = get_example_network(net_name="nn")

        # Test forward pass works
        with torch.no_grad():
            output = model(X_train[:5])  # Test with first 5 samples
            assert output.shape == (5, 1)  # Should output single value per sample
            assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output [0,1]

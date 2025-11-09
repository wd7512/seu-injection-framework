import numpy as np
import pytest
import torch

# Import from the new seu_injection package
from seu_injection.metrics.accuracy import (
    classification_accuracy,
    classification_accuracy_loader,
    multiclass_classification_accuracy,
)


class TestCriterionFunctions:
    """Test suite for criterion/metric functions."""

    def test_classification_accuracy_basic(self, simple_model, sample_data, device):
        """Test basic classification accuracy computation."""
        X, y = sample_data

        # Move data to device
        X = X.to(device)
        y = y.to(device)
        simple_model = simple_model.to(device)

        accuracy = classification_accuracy(simple_model, X, y, device)

        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0, (
            f"Accuracy should be between 0 and 1, got {accuracy}"
        )

    def test_classification_accuracy_perfect_model(self, device):
        """Test accuracy computation with a perfect model."""
        # Create a simple dataset where output should be perfect
        X = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=device)
        y = torch.tensor([[1.0], [0.0]], dtype=torch.float32, device=device)

        # Create a model that should predict perfectly
        model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid()).to(
            device
        )

        # Set weights to make perfect predictions
        with torch.no_grad():
            model[0].weight.data = torch.tensor([[1.0, -1.0]], device=device)
            model[0].bias.data = torch.tensor([0.0], device=device)

        accuracy = classification_accuracy(model, X, y, device)

        # Should be perfect or very close
        assert accuracy > 0.9, (
            f"Perfect model should have high accuracy, got {accuracy}"
        )

    def test_classification_accuracy_with_batching(self, simple_model, device):
        """Test accuracy computation with different batch sizes."""
        # Create larger dataset to test batching
        torch.manual_seed(42)
        X = torch.randn(200, 2, dtype=torch.float32, device=device)
        y = torch.randint(0, 2, (200, 1), dtype=torch.float32, device=device)

        simple_model = simple_model.to(device)

        # Test with different batch sizes
        accuracy_32 = classification_accuracy(simple_model, X, y, device, batch_size=32)
        accuracy_64 = classification_accuracy(simple_model, X, y, device, batch_size=64)
        accuracy_all = classification_accuracy(
            simple_model, X, y, device, batch_size=None
        )

        # Results should be identical regardless of batch size
        assert abs(accuracy_32 - accuracy_64) < 1e-6, (
            "Batch size should not affect accuracy"
        )
        assert abs(accuracy_32 - accuracy_all) < 1e-6, (
            "Batch size should not affect accuracy"
        )

    def test_classification_accuracy_loader(
        self, simple_model, sample_dataloader, device
    ):
        """Test accuracy computation using DataLoader."""
        simple_model = simple_model.to(device)

        accuracy = classification_accuracy_loader(
            simple_model, sample_dataloader, device
        )

        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0, (
            f"Accuracy should be between 0 and 1, got {accuracy}"
        )

    def test_multiclass_classification_accuracy_binary(self):
        """Test multiclass accuracy function with binary classification."""
        # Test binary classification (single output)
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred_probs = np.array([0.2, 0.8, 0.1, 0.9, 0.6])  # Single column output

        accuracy = multiclass_classification_accuracy(y_true, y_pred_probs)

        # Manual calculation: predictions [0, 1, 0, 1, 1] vs true [0, 1, 0, 1, 1] = 100%
        assert accuracy == 1.0, f"Expected accuracy 1.0, got {accuracy}"

    def test_multiclass_classification_accuracy_multiclass(self):
        """Test multiclass accuracy function with multi-class classification."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred_probs = np.array(
            [
                [0.8, 0.1, 0.1],  # Predicts class 0 ✓
                [0.2, 0.7, 0.1],  # Predicts class 1 ✓
                [0.1, 0.1, 0.8],  # Predicts class 2 ✓
                [0.6, 0.3, 0.1],  # Predicts class 0 ✓
                [0.1, 0.8, 0.1],  # Predicts class 1 ✓
            ]
        )

        accuracy = multiclass_classification_accuracy(y_true, y_pred_probs)

        assert accuracy == 1.0, f"Expected perfect accuracy, got {accuracy}"

    def test_multiclass_classification_accuracy_imperfect(self):
        """Test multiclass accuracy with some incorrect predictions."""
        y_true = np.array([0, 1, 2, 0])
        y_pred_probs = np.array(
            [
                [0.8, 0.1, 0.1],  # Predicts class 0 ✓
                [0.7, 0.2, 0.1],  # Predicts class 0 ✗ (should be 1)
                [0.1, 0.1, 0.8],  # Predicts class 2 ✓
                [0.6, 0.3, 0.1],  # Predicts class 0 ✓
            ]
        )

        accuracy = multiclass_classification_accuracy(y_true, y_pred_probs)

        # 3 out of 4 correct = 0.75
        assert accuracy == 0.75, f"Expected accuracy 0.75, got {accuracy}"

    def test_binary_classification_midpoint_logic(self):
        """Test binary classification midpoint calculation logic."""
        # Test with different label ranges
        y_true_01 = np.array([0, 1, 0, 1])
        y_pred = np.array([0.3, 0.7, 0.2, 0.8])

        accuracy_01 = multiclass_classification_accuracy(y_true_01, y_pred)

        # Midpoint between 0 and 1 is 0.5
        # Predictions: [0, 1, 0, 1] vs true [0, 1, 0, 1] = 100%
        assert accuracy_01 == 1.0

        # Test with different label range (-1, 1)
        y_true_neg = np.array([-1, 1, -1, 1])
        accuracy_neg = multiclass_classification_accuracy(y_true_neg, y_pred)

        # Midpoint between -1 and 1 is 0
        # Predictions [0.3, 0.7, 0.2, 0.8] all >= 0, so mapped to [1, 1, 1, 1]
        # vs true [-1, 1, -1, 1] = 2/4 = 50% accuracy
        assert accuracy_neg == 0.5

    def test_criterion_edge_cases(self, device):
        """Test criterion functions with edge cases."""
        # Test with very small model and data
        tiny_model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid()).to(
            device
        )
        X_tiny = torch.tensor([[1.0]], dtype=torch.float32, device=device)
        y_tiny = torch.tensor([[1.0]], dtype=torch.float32, device=device)

        accuracy = classification_accuracy(tiny_model, X_tiny, y_tiny, device)
        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0

    def test_criterion_consistency_tensor_vs_loader(
        self, simple_model, sample_data, sample_dataloader, device
    ):
        """Test that tensor-based and loader-based accuracy give same results."""
        X, y = sample_data
        X = X.to(device)
        y = y.to(device)
        simple_model = simple_model.to(device)

        accuracy_tensor = classification_accuracy(simple_model, X, y, device)
        accuracy_loader = classification_accuracy_loader(
            simple_model, sample_dataloader, device
        )

        # Should be very close (allowing for small floating point differences)
        assert abs(accuracy_tensor - accuracy_loader) < 1e-4, (
            f"Tensor and loader accuracies differ: {accuracy_tensor} vs {accuracy_loader}"
        )

    def test_classification_accuracy_dataloader_with_y_true_error(
        self, simple_model, sample_dataloader, device
    ):
        """Test that providing y_true with DataLoader raises ValueError."""
        simple_model = simple_model.to(device)
        dummy_y = torch.tensor([1, 0], device=device)

        with pytest.raises(
            ValueError, match="When using DataLoader, do not specify y_true separately"
        ):
            classification_accuracy(simple_model, sample_dataloader, dummy_y, device)

    def test_classification_accuracy_no_device(self, simple_model, sample_data):
        """Test accuracy computation without device specification."""
        X, y = sample_data

        accuracy = classification_accuracy(simple_model, X, y, device=None)
        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0

    def test_classification_accuracy_batch_size_none(
        self, simple_model, sample_data, device
    ):
        """Test accuracy computation with batch_size=None."""
        X, y = sample_data
        X = X.to(device)
        y = y.to(device)
        simple_model = simple_model.to(device)

        accuracy = classification_accuracy(simple_model, X, y, device, batch_size=None)
        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0

    def test_classification_accuracy_loader_no_device(
        self, simple_model, sample_dataloader
    ):
        """Test loader accuracy computation without device specification."""
        accuracy = classification_accuracy_loader(
            simple_model, sample_dataloader, device=None
        )
        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0

    def test_multiclass_classification_accuracy_1d_output(self):
        """Test multiclass accuracy with 1D model output (edge case)."""
        y_true = np.array([0, 1, 0, 1])
        # 1D output that should be treated as binary
        model_output = np.array([0.3, 0.7, 0.2, 0.8])

        accuracy = multiclass_classification_accuracy(y_true, model_output)
        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0

    def test_multiclass_classification_accuracy_single_column(self):
        """Test multiclass accuracy with single column output."""
        y_true = np.array([0, 1, 0, 1])
        # 2D output with single column (binary case)
        model_output = np.array([[0.3], [0.7], [0.2], [0.8]])

        accuracy = multiclass_classification_accuracy(y_true, model_output)
        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0

    def test_classification_accuracy_dataloader_return_path(
        self, simple_model, sample_dataloader, device
    ):
        """Test that DataLoader path returns classification_accuracy_loader result."""
        simple_model = simple_model.to(device)

        # This should trigger the return path at line 100
        accuracy = classification_accuracy(
            simple_model, sample_dataloader, device=device
        )
        assert isinstance(accuracy, (float, np.floating))
        assert 0.0 <= accuracy <= 1.0

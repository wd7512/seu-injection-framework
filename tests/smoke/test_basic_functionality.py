# Smoke tests - quick validation that basic functionality works
import numpy as np
import torch


def test_basic_imports():
    """Test that all framework modules can be imported without errors."""
    try:
        from seu_injection import classification_accuracy
        from seu_injection.bitops.float32 import (
            binary_to_float32,
            bitflip_float32,
            float32_to_binary,
        )
        from seu_injection.core import ExhaustiveSEUInjector as Injector
        from seu_injection.metrics.accuracy import multiclass_classification_accuracy

        # Basic validation that classes/functions exist
        assert Injector is not None
        assert classification_accuracy is not None
        assert bitflip_float32 is not None

    except ImportError as e:
        assert False, f"Import failed: {e}"


def test_basic_bitflip_functionality():
    """Quick test that bitflip operations work."""
    from seu_injection.bitops.float32 import bitflip_float32

    # Simple bitflip test
    result = bitflip_float32(1.0, 0)
    assert result == -1.0, "Basic bitflip test failed"

    # Array test
    values = np.array([1.0, 2.0], dtype=np.float32)
    result = bitflip_float32(values, 0)
    expected = np.array([-1.0, -2.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected, err_msg="Array bitflip test failed")


def test_basic_model_creation():
    """Test that we can create simple models for testing."""
    # Create a minimal model
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())

    # Test forward pass
    x = torch.randn(5, 2)
    output = model(x)

    assert output.shape == (5, 1), "Model output shape incorrect"
    assert torch.all((output >= 0) & (output <= 1)), "Sigmoid output should be in [0,1]"


def test_basic_injector_creation():
    """Test that Injector can be created and initialized."""
    from seu_injection import classification_accuracy
    from seu_injection.core import ExhaustiveSEUInjector as Injector

    # Create simple model and data
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    X = torch.randn(10, 2)
    y = torch.randint(0, 2, (10, 1)).float()

    # Create injector - should not raise errors
    injector = Injector(
        trained_model=model, criterion=classification_accuracy, x=X, y=y
    )

    assert injector.model is not None
    assert injector.baseline_score is not None
    assert isinstance(injector.baseline_score, (float, np.floating))


def test_basic_criterion_functionality():
    """Test that criterion functions work."""
    from seu_injection.metrics.accuracy import multiclass_classification_accuracy

    # Simple binary classification test
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])

    accuracy = multiclass_classification_accuracy(y_true, y_pred)
    assert accuracy == 1.0, "Perfect prediction should give 100% accuracy"


def test_device_compatibility():
    """Test basic device (CPU/CUDA) compatibility."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and move to device
    model = torch.nn.Linear(2, 1)
    model = model.to(device)

    # Create data and move to device
    x = torch.randn(5, 2, device=device)
    output = model(x)

    assert output.device == device, "Model output should be on correct device"


def test_example_networks_import():
    """Test that example networks module can be imported."""
    try:
        # Ensure the repository root is in the path for direct execution
        import os
        import sys

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # The fixtures module should now be importable as a proper package
        from tests.fixtures.example_networks import get_example_network

        # Test that function exists
        assert get_example_network is not None

        # Quick test of network creation (without training)
        model, X_train, X_test, y_train, y_test, train_fn, eval_fn = (
            get_example_network(net_name="nn", train=False)
        )

        assert model is not None
        assert X_train is not None
        assert len(X_train.shape) == 2  # Should be 2D tensor

    except ImportError as e:
        assert False, f"Example networks import failed: {e}"


def test_framework_version_info():
    """Test basic framework information and structure."""
    import seu_injection

    # Framework should exist as a package
    assert seu_injection is not None

    # Key modules should be accessible
    assert hasattr(seu_injection, "__version__")


def test_dependencies_available():
    """Test that key dependencies are available."""
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import torch
        import tqdm

        # Check versions are reasonable
        assert hasattr(torch, "__version__")
        assert hasattr(np, "__version__")

    except ImportError as e:
        assert False, f"Required dependency missing: {e}"


def test_performance_smoke():
    """Basic performance smoke test - ensure operations complete quickly."""
    import time

    from seu_injection.bitops.float32 import bitflip_float32

    # Test should complete quickly
    start_time = time.time()

    # Perform a bunch of operations
    for _ in range(100):
        bitflip_float32(1.0, 15)

    end_time = time.time()
    duration = end_time - start_time

    # Should complete in well under a second
    assert duration < 1.0, (
        f"Performance smoke test too slow: {duration}s for 100 operations"
    )


if __name__ == "__main__":
    """Run smoke tests directly for quick validation."""
    print("Running smoke tests...")

    tests = [
        test_basic_imports,
        test_basic_bitflip_functionality,
        test_basic_model_creation,
        test_basic_injector_creation,
        test_basic_criterion_functionality,
        test_device_compatibility,
        test_example_networks_import,
        test_framework_version_info,
        test_dependencies_available,
        test_performance_smoke,
    ]

    for test in tests:
        try:
            test()
            print(f"OK {test.__name__}")
        except Exception as e:
            print(f"FAIL {test.__name__}: {e}")

    print("Smoke tests complete!")

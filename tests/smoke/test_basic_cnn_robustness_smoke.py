"""
Smoke test for basic_cnn_robustness.py example.

This test performs a quick validation to ensure the example can import
and basic functions work without errors. It's designed for fast execution
in the smoke test suite.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

# Add examples directory to path
examples_dir = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(examples_dir))


def test_basic_cnn_robustness_imports():
    """Test that basic_cnn_robustness can be imported without errors."""
    try:
        import basic_cnn_robustness

        assert basic_cnn_robustness is not None
        print("SUCCESS: basic_cnn_robustness imports successfully")
    except ImportError as e:
        raise AssertionError(f"Failed to import basic_cnn_robustness: {e}") from e


def test_basic_cnn_robustness_functions_exist():
    """Test that all required functions exist in the example."""
    import basic_cnn_robustness

    required_functions = [
        "create_simple_cnn",
        "prepare_data",
        "train_model",
        "run_baseline_analysis",
        "analyze_sign_bit_vulnerability",
        "analyze_layer_vulnerability",
        "analyze_bit_position_sensitivity",
        "create_visualizations",
        "main",
    ]

    for func_name in required_functions:
        assert hasattr(basic_cnn_robustness, func_name), (
            f"Missing function: {func_name}"
        )
        assert callable(getattr(basic_cnn_robustness, func_name)), (
            f"Not callable: {func_name}"
        )

    print("SUCCESS: All required functions exist and are callable")


def test_basic_cnn_robustness_minimal_execution():
    """Test basic execution with minimal parameters (smoke test)."""
    import basic_cnn_robustness

    # Test individual components quickly

    # 1. Test model creation
    model = basic_cnn_robustness.create_simple_cnn()
    assert isinstance(model, torch.nn.Module)
    print("SUCCESS: Model creation works")

    # 2. Test data preparation
    X_train, X_test, y_train, y_test = basic_cnn_robustness.prepare_data()
    assert all(isinstance(x, torch.Tensor) for x in [X_train, X_test, y_train, y_test])
    print("SUCCESS: Data preparation works")

    # 3. Test very quick training (1 epoch)
    trained_model = basic_cnn_robustness.train_model(model, X_train, y_train, epochs=1)
    assert trained_model is not None
    print("SUCCESS: Model training works")

    # 4. Test baseline analysis setup
    injector, baseline_acc = basic_cnn_robustness.run_baseline_analysis(
        trained_model,
        X_test[:20],
        y_test[:20],  # Use only 20 samples for speed
    )
    assert injector is not None
    assert isinstance(baseline_acc, float)
    print("SUCCESS: Baseline analysis setup works")

    print(
        "SUCCESS: SMOKE TEST PASSED: basic_cnn_robustness example basic functionality verified"
    )


def test_basic_cnn_robustness_seu_framework_integration():
    """Test that the example properly integrates with SEU framework."""
    # Test imports work
    from seu_injection.core import ExhaustiveSEUInjector
    from seu_injection.metrics import classification_accuracy

    assert ExhaustiveSEUInjector is not None
    assert classification_accuracy is not None
    print("SUCCESS: SEU framework integration confirmed")


if __name__ == "__main__":
    # Run smoke tests
    test_basic_cnn_robustness_imports()
    test_basic_cnn_robustness_functions_exist()
    test_basic_cnn_robustness_minimal_execution()
    test_basic_cnn_robustness_seu_framework_integration()

    print("\nSUCCESS: ALL SMOKE TESTS PASSED - basic_cnn_robustness example is ready!")

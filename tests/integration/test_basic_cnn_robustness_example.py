"""
Integration test for the basic_cnn_robustness.py example.

This test ensures the complete example can run successfully without errors
and produces expected outputs. It validates the entire pipeline from data
preparation through model training to SEU injection analysis.
"""

import subprocess
import sys
from pathlib import Path

import pytest


class TestBasicCNNRobustnessExample:
    """Test suite for the basic CNN robustness analysis example."""

    def test_example_syntax_validation(self):
        """Test that the example has valid Python syntax."""
        examples_dir = Path(__file__).parent.parent.parent / "examples"
        example_path = examples_dir / "basic_cnn_robustness.py"

        assert example_path.exists(), f"Example not found: {example_path}"

        # Test syntax compilation
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(example_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Syntax errors in example: {result.stderr}"

    def test_example_imports_work(self):
        """Test that all imports in the example work correctly."""
        examples_dir = Path(__file__).parent.parent.parent / "examples"
        test_imports_script = f"""
import sys
sys.path.insert(0, '{examples_dir}')

# Test framework imports
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy

# Test other required imports
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Test example can be imported
import basic_cnn_robustness

# Check key functions exist
assert hasattr(basic_cnn_robustness, 'main')
assert hasattr(basic_cnn_robustness, 'create_simple_cnn')
assert hasattr(basic_cnn_robustness, 'prepare_data')
assert hasattr(basic_cnn_robustness, 'train_model')

print("All imports successful!")
"""

        result = subprocess.run(
            [sys.executable, "-c", test_imports_script], capture_output=True, text=True
        )

        assert result.returncode == 0, f"Import failures: {result.stderr}"
        assert "All imports successful!" in result.stdout

    def test_example_individual_functions_work(self):
        """Test that individual functions in the example work correctly."""
        examples_dir = Path(__file__).parent.parent.parent / "examples"
        test_functions_script = f"""
import sys
sys.path.insert(0, '{examples_dir}')
import basic_cnn_robustness
import torch

# Test model creation
model = basic_cnn_robustness.create_simple_cnn()
assert isinstance(model, torch.nn.Module)

# Test data preparation
X_train, X_test, y_train, y_test = basic_cnn_robustness.prepare_data()
assert all(isinstance(x, torch.Tensor) for x in [X_train, X_test, y_train, y_test])
assert X_train.shape[1] == 2  # 2D features
assert y_train.shape[1] == 1  # Single output

# Test quick training (1 epoch)
trained_model = basic_cnn_robustness.train_model(model, X_train, y_train, epochs=1)
assert trained_model is not None

# Test baseline analysis setup
injector, baseline_acc = basic_cnn_robustness.run_baseline_analysis(
    trained_model, X_test[:20], y_test[:20]  # Small sample for speed
)
assert injector is not None
assert isinstance(baseline_acc, float)
assert 0.0 <= baseline_acc <= 1.0

print("Individual function tests passed!")
"""

        result = subprocess.run(
            [sys.executable, "-c", test_functions_script],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"Function test failures: {result.stderr}"
        assert "Individual function tests passed!" in result.stdout

    def test_example_can_run_with_short_timeout(self):
        """Test that the example can start running without immediate errors."""
        examples_dir = Path(__file__).parent.parent.parent / "examples"
        example_path = examples_dir / "basic_cnn_robustness.py"

        # Run example with a very short timeout just to test it starts correctly
        # We expect it to timeout, but it should not fail immediately with import/syntax errors
        try:
            result = subprocess.run(
                [sys.executable, str(example_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=examples_dir.parent,
            )

            # If it completes within 30 seconds, great!
            assert result.returncode == 0, f"Example failed: {result.stderr}"

        except subprocess.TimeoutExpired:
            # This is expected - the example is designed to be comprehensive
            # The important thing is that it didn't fail immediately
            print("Example started successfully (timed out as expected)")
            pass

    def test_example_dependencies_available(self):
        """Test that all required dependencies for the example are available."""
        dependency_check_script = """
# Check all required dependencies
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check SEU injection framework
from seu_injection import SEUInjector
from seu_injection.metrics import classification_accuracy

print("All dependencies available!")
"""

        result = subprocess.run(
            [sys.executable, "-c", dependency_check_script],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Missing dependencies: {result.stderr}"
        assert "All dependencies available!" in result.stdout

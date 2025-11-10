# Test configuration for pytest
# TODO DIRECTORY STRUCTURE: Testing organization improvements per USER_EXPERIENCE_IMPROVEMENT_PLAN.md
# RESOLVED: testing/ vs tests/ directory confusion has been cleaned up
# CURRENT: Single tests/ directory with clear organization:
#   - tests/unit/ - Individual component testing
#   - tests/integration/ - End-to-end workflow testing
#   - tests/smoke/ - Quick validation testing
#   - tests/benchmarks/ - Performance testing
# IMPROVEMENT OPPORTUNITIES:
#   - Consider tests/fixtures/ subdirectory for shared test models/data
#   - Add tests/regression/ for catching performance regressions
#   - Move example_networks.py to tests/fixtures/models.py for clarity
# PRIORITY: LOW - Current structure is functional, improvements can wait

import os
import sys

import torch

# Add the framework to the Python path for all tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Test fixtures and common utilities
import pytest


@pytest.fixture
def device():
    """Fixture to provide the best available device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_model():
    """Fixture to provide a simple neural network for testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1),
        torch.nn.Sigmoid(),
    )
    # Initialize weights for reproducible tests
    torch.manual_seed(42)
    for layer in model:
        if hasattr(layer, "weight"):
            torch.nn.init.normal_(layer.weight, 0, 0.1)
            if hasattr(layer, "bias"):
                torch.nn.init.zeros_(layer.bias)
    return model


@pytest.fixture
def sample_data():
    """Fixture to provide sample input data and labels."""
    torch.manual_seed(42)
    X = torch.randn(100, 2, dtype=torch.float32)
    y = torch.randint(0, 2, (100, 1), dtype=torch.float32)
    return X, y


@pytest.fixture
def sample_dataloader(sample_data):
    """Fixture to provide a sample DataLoader."""
    X, y = sample_data
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

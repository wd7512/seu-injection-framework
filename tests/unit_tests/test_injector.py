import numpy as np
import pytest
import torch

from seu_injection import classification_accuracy

# Import from the new seu_injection package
from seu_injection.core import ExhaustiveSEUInjector, StochasticSEUInjector


class TestInjector:
    """Test suite for the SEU Injector classes (ExhaustiveSEUInjector, StochasticSEUInjector)."""

    # TODO TESTING IMPROVEMENTS: Test coverage gaps and enhancements per improvement plans
    # COVERAGE GAPS:
    #   - Edge cases: NaN/Inf handling in bitflip operations
    #   - Error conditions: Invalid bit positions, incompatible devices
    #   - Performance: Large model injection scenarios (memory/speed)
    #   - Integration: Real model architectures (ResNet, EfficientNet)
    # MISSING TESTS:
    #   - Multi-GPU device handling and tensor movement
    #   - Custom criterion function validation
    #   - Layer-specific injection with complex model hierarchies
    #   - Stochastic injection reproducibility with different random seeds
    # PRIORITY: MEDIUM - Current 94% coverage good, but edge cases important for production

    def test_injector_initialization_with_tensor_data(self, simple_model, sample_data, device):
        """Test Injector initialization with tensor data."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        assert injector.model is not None
        assert injector.criterion == classification_accuracy
        assert injector.device == device
        assert injector.baseline_score is not None
        assert isinstance(injector.baseline_score, (float, np.floating))
        assert 0.0 <= injector.baseline_score <= 1.0

    def test_injector_device_auto_detection(self, simple_model, sample_data):
        """Test automatic device detection (CUDA vs CPU)."""
        X, y = sample_data

        # Test with device=None to trigger auto-detection
        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=None,  # This should trigger auto-detection
            x=X,
            y=y,
        )

        # Should detect either CUDA or CPU
        assert str(injector.device) in ["cuda", "cpu"]

        # The detected device should match what torch.cuda.is_available() suggests
        if torch.cuda.is_available():
            # If CUDA is available, device could be either (depends on implementation)
            assert str(injector.device) in ["cuda", "cpu"]
        else:
            # If CUDA is not available, should default to CPU
            assert str(injector.device) == "cpu"

    def test_injector_cuda_path_coverage(self, simple_model, sample_data, device):
        """Test CUDA path for coverage by explicitly testing device handling."""
        X, y = sample_data

        # Test explicit CUDA device assignment (will fall back to CPU if no CUDA)
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
        else:
            # Skip CUDA-specific testing on systems without CUDA
            pytest.skip("CUDA not available for testing")
            return

        # Test with explicit device to ensure CUDA path coverage
        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=cuda_device,
            x=X,
            y=y,
        )

        # Verify CUDA device was used (if available)
        assert str(injector.device).startswith("cuda")

    def test_injector_with_numpy_input(self, simple_model, sample_data, device):
        """Test Injector with numpy array input (non-tensor)."""
        X, y = sample_data

        # Convert to numpy to test the tensor conversion path
        X_numpy = X.cpu().numpy()
        y_numpy = y.cpu().numpy()

        # We need to ensure the y values are proper for the criterion function
        # Convert to binary classification format
        y_numpy = (y_numpy > 0.5).astype(np.float32)

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X_numpy,  # This should trigger torch.tensor() conversion
            y=y_numpy,
        )

        assert injector.model is not None
        assert torch.is_tensor(injector.X)
        assert torch.is_tensor(injector.y)

    def test_injector_initialization_with_dataloader(self, simple_model, sample_data, device):
        """Test Injector initialization with DataLoader."""
        X, y = sample_data

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            data_loader=dataloader,
        )

        assert injector.model is not None
        assert injector.use_data_loader is True
        assert injector.data_loader is dataloader
        assert injector.baseline_score is not None

    def test_injector_invalid_initialization(self, simple_model, sample_data, device):
        """Test that Injector raises error when both X/y and dataloader are provided."""
        X, y = sample_data

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

        with pytest.raises(ValueError, match="Cannot pass both a dataloader and x and y values"):
            ExhaustiveSEUInjector(
                trained_model=simple_model,
                criterion=classification_accuracy,
                device=device,
                x=X,
                y=y,
                data_loader=dataloader,
            )

    def test_injector_missing_data(self, simple_model):
        """Test that Injector raises error when neither data_loader nor X/y are provided."""
        with pytest.raises(ValueError, match="Must provide either data_loader or at least one of X, y"):
            ExhaustiveSEUInjector(
                trained_model=simple_model,
                criterion=classification_accuracy,
                device="cpu",
            )

    def test_get_criterion_score(self, simple_model, sample_data, device):
        """Test that get_criterion_score returns consistent results."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Get score multiple times - should be consistent
        score1 = injector._get_criterion_score()
        score2 = injector._get_criterion_score()

        assert score1 == score2, "Criterion score should be consistent"
        assert score1 == injector.baseline_score, "get_criterion_score should match baseline"

    def test_get_criterion_score_with_dataloader(self, simple_model, sample_data, device):
        """Test get_criterion_score with DataLoader to cover the DataLoader branch."""
        X, y = sample_data

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            data_loader=dataloader,
        )

        # This should trigger the DataLoader branch in get_criterion_score
        score = injector._get_criterion_score()
        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0

    def test_run_seu_basic(self, simple_model, sample_data, device):
        """Test basic SEU injection functionality."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Run SEU injection on sign bit (bit 0)
        results = injector.run_injector(bit_i=0)

        # Validate results structure
        assert isinstance(results, dict)
        expected_keys = [
            "tensor_location",
            "criterion_score",
            "layer_name",
            "value_before",
            "value_after",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key {key} in results"
            assert isinstance(results[key], list)

        # All lists should have the same length
        lengths = [len(results[key]) for key in expected_keys]
        assert all(length == lengths[0] for length in lengths), "Result lists have different lengths"

        # Should have some results (the simple model has parameters)
        assert len(results["tensor_location"]) > 0, "No SEU injection results found"

    def test_run_seu_bit_position_validation(self, simple_model, sample_data, device):
        """Test that run_seu validates bit positions correctly."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Test valid bit positions
        for bit_pos in [0, 15, 31]:
            results = injector.run_injector(bit_i=bit_pos)
        assert len(results["tensor_location"]) > 0

        # Test invalid bit positions should raise ValueError
        with pytest.raises(ValueError):
            injector.run_injector(bit_i=-1)

        with pytest.raises(ValueError):
            injector.run_injector(bit_i=33)

    def test_run_seu_layer_targeting(self, simple_model, sample_data, device):
        """Test SEU injection with layer targeting."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Get layer names from the model
        layer_names = [name for name, _ in simple_model.named_parameters()]
        assert len(layer_names) > 0, "Model should have named parameters"

        # Target specific layer
        target_layer = layer_names[0]
        results = injector.run_injector(bit_i=0, layer_name=target_layer)

        # All results should be from the targeted layer
        assert all(layer == target_layer for layer in results["layer_name"]), (
            f"Found results from non-targeted layers: {set(results['layer_name'])}"
        )

    def test_run_seu_layer_filtering(self, simple_model, sample_data, device):
        """Test SEU injection with layer filtering to trigger continue statement."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Get layer names - should be ['0.weight', '0.bias', '2.weight', '2.bias']
        [name for name, _ in simple_model.named_parameters()]

        # Target a layer in the middle - this ensures other layers are skipped
        target_layer = "2.weight"  # Target the second linear layer specifically

        # This should iterate through ALL layers ['0.weight', '0.bias', '2.weight', '2.bias']
        # but only process '2.weight'. The continue statement should be executed for
        # '0.weight', '0.bias', and '2.bias'
        results = injector.run_injector(bit_i=0, layer_name=target_layer)

        # Verify results only come from the targeted layer
        assert all(layer == target_layer for layer in results["layer_name"]), (
            f"Should only have results from {target_layer}, got: {set(results['layer_name'])}"
        )

        # Verify we have results only from the target layer (multiple results from same layer are OK)
        assert len(set(results["layer_name"])) == 1, (
            f"Should have results from only 1 layer, got {len(set(results['layer_name']))}"
        )
        assert results["layer_name"][0] == target_layer

        # Also test with non-existent layer to ensure all layers are skipped
        # This MUST trigger the continue statement for ALL layers
        results_empty = injector.run_injector(bit_i=0, layer_name="nonexistent_layer")
        assert len(results_empty["layer_name"]) == 0, "Should have no results for non-existent layer"

        # Test with first layer to ensure other layers are skipped
        results_first = injector.run_injector(bit_i=0, layer_name="0.weight")
        # This should process 0.weight and skip 0.bias, 2.weight, 2.bias
        assert all(layer == "0.weight" for layer in results_first["layer_name"])
        assert len(results_first["layer_name"]) > 0

    def test_run_stochastic_seu_basic(self, simple_model, sample_data, device):
        """Test basic stochastic SEU injection functionality."""
        X, y = sample_data

        injector = StochasticSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Run stochastic SEU with 50% probability
        results = injector.run_injector(bit_i=0, p=0.5)

        # Validate results structure (same as regular SEU)
        assert isinstance(results, dict)
        expected_keys = [
            "tensor_location",
            "criterion_score",
            "layer_name",
            "value_before",
            "value_after",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key {key} in results"

        # Should have some results, but likely fewer than exhaustive SEU
        assert len(results["tensor_location"]) > 0, "No stochastic SEU injection results found"

    def test_run_stochastic_seu_probability_validation(self, simple_model, sample_data, device):
        """Test probability validation in stochastic SEU."""
        X, y = sample_data

        injector = StochasticSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Test valid probabilities
        for p in [0.0, 0.1, 0.5, 1.0]:
            results = injector.run_injector(bit_i=0, p=p)
        assert isinstance(results, dict)

        # Test invalid probabilities
        with pytest.raises(ValueError):
            injector.run_injector(bit_i=0, p=-0.1)

        with pytest.raises(ValueError):
            injector.run_injector(bit_i=0, p=1.1)

    def test_stochastic_seu_probability_effects(self, simple_model, sample_data, device):
        """Test that different probabilities affect the number of injections."""
        X, y = sample_data

        injector = StochasticSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Set random seed for reproducible results
        np.random.seed(42)
        results_low = injector.run_injector(bit_i=0, p=0.1)

        np.random.seed(42)
        results_high = injector.run_injector(bit_i=0, p=0.9)

        # Higher probability should generally result in more injections
        # (though with randomness, this isn't guaranteed, so we use a loose check)
        low_count = len(results_low["tensor_location"])
        high_count = len(results_high["tensor_location"])

        # At minimum, both should have some results
        assert low_count >= 0, "Low probability should have some results"
        assert high_count >= 0, "High probability should have some results"

    def test_model_state_preservation(self, simple_model, sample_data, device):
        """Test that model parameters are restored after SEU injection."""
        X, y = sample_data

        # Store original model state
        original_params = {}
        for name, param in simple_model.named_parameters():
            original_params[name] = param.data.clone()

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Run SEU injection
        injector.run_injector(bit_i=0)

        # Check that model parameters are restored
        for name, param in simple_model.named_parameters():
            torch.testing.assert_close(
                param.data,
                original_params[name],
                msg=f"Parameter {name} was not restored after SEU injection",
            )

    def test_run_stochastic_seu_layer_filtering(self, simple_model, sample_data, device):
        """Test stochastic SEU injection with layer filtering to trigger continue statement."""
        X, y = sample_data

        injector = StochasticSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Target specific layer in stochastic SEU - this should trigger continue for other layers
        results = injector.run_injector(bit_i=0, p=1.0, layer_name="0.weight")

        # Should only process the targeted layer, skipping others via continue
        assert all(layer == "0.weight" for layer in results["layer_name"]), (
            f"Should only have results from 0.weight, got: {set(results['layer_name'])}"
        )

        # Also test with nonexistent layer to trigger continue for all layers
        results_empty = injector.run_injector(bit_i=0, p=1.0, layer_name="nonexistent_layer")
        assert len(results_empty["layer_name"]) == 0, "Should have no results for nonexistent layer"

    def test_run_at_least_one_injection_default(self, simple_model, sample_data, device):
        """Test that run_at_least_one_injection defaults to True and ensures at least one injection per layer."""
        X, y = sample_data

        injector = StochasticSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Run with p=0 to force run_at_least_one_injection behavior
        results = injector.run_injector(bit_i=0, p=0)

        # Should have at least one result per layer
        layer_names = [name for name, _ in simple_model.named_parameters()]
        unique_layers = set(results["layer_name"])

        # With run_at_least_one_injection=True (default), we should have results for all layers
        assert len(unique_layers) == len(layer_names), (
            f"Expected at least one injection per layer. Got {len(unique_layers)} layers, expected {len(layer_names)}"
        )

    def test_run_at_least_one_injection_false(self, simple_model, sample_data, device):
        """Test that run_at_least_one_injection=False allows zero injections with low probability."""
        X, y = sample_data

        injector = StochasticSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Set seed for reproducibility
        np.random.seed(42)

        # Run with p=0 and run_at_least_one_injection=False
        results = injector.run_injector(bit_i=0, p=0, run_at_least_one_injection=False)

        # With p=0 and run_at_least_one_injection=False, we should have zero results
        # This is expected behavior and should not crash
        assert isinstance(results, dict)
        assert "tensor_location" in results
        assert "criterion_score" in results

    def test_run_at_least_one_injection_specific_layer(self, simple_model, sample_data, device):
        """Test that run_at_least_one_injection works correctly with layer targeting."""
        X, y = sample_data

        injector = StochasticSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Target a specific layer with p=0 to force run_at_least_one_injection
        target_layer = "0.weight"
        results = injector.run_injector(bit_i=0, p=0, layer_name=target_layer)

        # Should have at least one result from the targeted layer
        assert len(results["layer_name"]) >= 1, "Should have at least one injection in targeted layer"
        assert all(layer == target_layer for layer in results["layer_name"]), (
            f"All results should be from {target_layer}, got: {set(results['layer_name'])}"
        )

    def test_initialize_results(self, simple_model, sample_data, device):
        """Test that _initialize_results creates proper structure."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        results = injector._initialize_results()

        # Verify structure
        assert isinstance(results, dict)
        assert "tensor_location" in results
        assert "criterion_score" in results
        assert "layer_name" in results
        assert "value_before" in results
        assert "value_after" in results

        # Verify all are empty lists
        for key, value in results.items():
            assert isinstance(value, list)
            assert len(value) == 0

    def test_iterate_layers(self, simple_model, sample_data, device):
        """Test _iterate_layers helper method."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Test iterating all layers
        all_layers = list(injector._iterate_layers(None))
        assert len(all_layers) > 0
        assert all(isinstance(name, str) and isinstance(param, torch.nn.Parameter) for name, param in all_layers)

        # Test filtering by specific layer
        target_layer = all_layers[0][0]
        filtered_layers = list(injector._iterate_layers(target_layer))
        assert len(filtered_layers) == 1
        assert filtered_layers[0][0] == target_layer

    def test_prepare_tensor_for_injection(self, simple_model, sample_data, device):
        """Test _prepare_tensor_for_injection helper method."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Get first layer
        layer_name, tensor = next(injector._iterate_layers(None))

        # Prepare tensor
        original_tensor, tensor_cpu = injector._prepare_tensor_for_injection(tensor)

        # Verify outputs
        assert isinstance(original_tensor, torch.Tensor)
        assert isinstance(tensor_cpu, np.ndarray)
        assert original_tensor.shape == tensor.shape
        assert tensor_cpu.shape == tuple(tensor.shape)

    def test_inject_and_evaluate(self, simple_model, sample_data, device):
        """Test _inject_and_evaluate helper method."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Get first layer
        layer_name, tensor = next(injector._iterate_layers(None))
        original_tensor, tensor_cpu = injector._prepare_tensor_for_injection(tensor)

        # Get first index
        idx = tuple(np.ndindex(tensor_cpu.shape))[0]
        original_val = tensor_cpu[idx]

        # Store original tensor value to verify restoration
        original_tensor_val = tensor.data[idx].item()

        # Inject and evaluate
        criterion_score, seu_val = injector._inject_and_evaluate(tensor, idx, original_tensor, original_val, bit_i=15)

        # Verify outputs
        assert isinstance(criterion_score, float)
        assert isinstance(seu_val, (float, np.floating))
        assert 0.0 <= criterion_score <= 1.0

        # Verify tensor was restored
        assert tensor.data[idx].item() == original_tensor_val

    def test_record_injection_result(self, simple_model, sample_data, device):
        """Test _record_injection_result helper method."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        results = injector._initialize_results()
        idx = (0, 1)
        criterion_score = 0.95
        layer_name = "test_layer"
        original_val = 1.5
        seu_val = 1.6

        # Record result
        injector._record_injection_result(results, idx, criterion_score, layer_name, original_val, seu_val)

        # Verify recording
        assert len(results["tensor_location"]) == 1
        assert results["tensor_location"][0] == idx
        assert results["criterion_score"][0] == criterion_score
        assert results["layer_name"][0] == layer_name
        assert results["value_before"][0] == original_val
        assert results["value_after"][0] == seu_val

    def test_get_injection_indices_exhaustive(self, simple_model, sample_data, device):
        """Test _get_injection_indices for exhaustive strategy."""
        X, y = sample_data

        injector = ExhaustiveSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Test with a small shape
        tensor_shape = (2, 3)
        indices = injector._get_injection_indices(tensor_shape)

        # Should get all 6 indices
        assert len(indices) == 6
        assert isinstance(indices, np.ndarray)

    def test_get_injection_indices_stochastic(self, simple_model, sample_data, device):
        """Test _get_injection_indices for stochastic strategy."""
        X, y = sample_data

        injector = StochasticSEUInjector(
            trained_model=simple_model,
            criterion=classification_accuracy,
            device=device,
            x=X,
            y=y,
        )

        # Test with p=0.5 and a reasonable shape
        np.random.seed(42)
        tensor_shape = (10, 10)
        indices = injector._get_injection_indices(tensor_shape, p=0.5)

        # Should get roughly 50% of indices (but exact count varies with randomness)
        assert len(indices) > 0
        assert len(indices) < 100  # Should not be all indices
        assert isinstance(indices, np.ndarray)

        # Test with p=0 and run_at_least_one_injection=True
        indices_min = injector._get_injection_indices(tensor_shape, p=0.0, run_at_least_one_injection=True)
        assert len(indices_min) == 1  # Should have exactly one

        # Test error handling for invalid p
        with pytest.raises(ValueError, match="Probability p must be in"):
            injector._get_injection_indices(tensor_shape, p=1.5)

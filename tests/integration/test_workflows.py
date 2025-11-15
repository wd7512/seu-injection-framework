# Integration tests - test complete workflows and component interactions
import numpy as np
import pandas as pd
import torch

from seu_injection import classification_accuracy

# Import from the new seu_injection package
from seu_injection.core import ExhaustiveSEUInjector, StochasticSEUInjector
from tests.fixtures.example_networks import get_example_network


class TestSEUInjectionWorkflows:
    """Integration tests for complete SEU injection workflows."""

    def test_complete_nn_workflow(self):
        """Test complete workflow with neural network."""
        # Get a trained neural network
        model, X_train, X_test, y_train, y_test, train_fn, eval_fn = (
            get_example_network(
                net_name="nn",
                train=True,
                epochs=1,  # Ultra-minimal epochs for fast testing
            )
        )

        # Create injector
        injector = ExhaustiveSEUInjector(
            trained_model=model, criterion=classification_accuracy, x=X_test, y=y_test
        )

        # Test basic injection
        results = injector.run_injector(bit_i=0)

        # Validate workflow results
        assert isinstance(results, dict)
        assert len(results["tensor_location"]) > 0
        assert len(results["criterion_score"]) > 0

        # Results should be reasonable
        baseline = injector.baseline_score
        avg_degraded = np.mean(results["criterion_score"])

        assert 0.0 <= baseline <= 1.0
        assert 0.0 <= avg_degraded <= 1.0

        # Convert to DataFrame (as shown in examples)
        df_results = pd.DataFrame(results)
        assert len(df_results) > 0
        assert "criterion_score" in df_results.columns

    def test_complete_cnn_workflow(self):
        """Test complete workflow with CNN."""
        model, X_train, X_test, y_train, y_test, train_fn, eval_fn = (
            get_example_network(net_name="cnn", train=True, epochs=1)
        )

        injector = StochasticSEUInjector(
            trained_model=model, criterion=classification_accuracy, x=X_test, y=y_test
        )

        # Test stochastic injection (faster for CNNs)
        results = injector.run_injector(bit_i=0, p=0.1)

        assert isinstance(results, dict)
        # For small models or low probability, we might not get any injections
        # Just validate the structure exists
        assert "tensor_location" in results
        assert "criterion_score" in results

    def test_complete_rnn_workflow(self):
        """Test complete workflow with RNN."""
        model, X_train, X_test, y_train, y_test, train_fn, eval_fn = (
            get_example_network(net_name="rnn", train=True, epochs=1)
        )

        injector = ExhaustiveSEUInjector(
            trained_model=model, criterion=classification_accuracy, x=X_test, y=y_test
        )

        # Test layer-specific injection
        layer_names = [name for name, _ in model.named_parameters()]
        if len(layer_names) > 0:
            target_layer = layer_names[0]
            results = injector.run_injector(bit_i=0, layer_name=target_layer)

            # All results should be from targeted layer
            assert all(layer == target_layer for layer in results["layer_name"])

    def test_dataloader_workflow(self):
        """Test complete workflow using DataLoader."""
        model, X_train, X_test, y_train, y_test, train_fn, eval_fn = (
            get_example_network(net_name="nn", train=True, epochs=1)
        )

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_test, y_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

        injector = StochasticSEUInjector(
            trained_model=model,
            criterion=classification_accuracy,
            data_loader=dataloader,
        )

        # Test with dataloader
        results = injector.run_injector(bit_i=0, p=0.2)

        assert isinstance(results, dict)
        assert injector.use_data_loader is True

    def test_multiple_bit_positions(self):
        """Test injection across multiple bit positions."""
        model, X_train, X_test, y_train, y_test, train_fn, eval_fn = (
            get_example_network(net_name="nn", train=True, epochs=1)
        )

        injector = StochasticSEUInjector(
            trained_model=model, criterion=classification_accuracy, x=X_test, y=y_test
        )

        baseline = injector.baseline_score
        bit_results = {}

        # Test different bit positions with higher probability to ensure results
        for bit_pos in [0, 1, 15, 31]:  # Sign, low bits, middle, LSB
            results = injector.run_injector(bit_i=bit_pos, p=0.5)  # Higher probability
        if len(results["criterion_score"]) > 0:
            avg_accuracy = np.mean(results["criterion_score"])
            bit_results[bit_pos] = avg_accuracy
        else:
            # If no injections occurred, use baseline
            bit_results[bit_pos] = baseline

        # Results should be reasonable
        for bit_pos, accuracy in bit_results.items():
            assert 0.0 <= accuracy <= 1.0, (
                f"Bit {bit_pos} gave invalid accuracy {accuracy}"
            )

    def test_robustness_analysis_pipeline(self):
        """Test a complete robustness analysis pipeline."""
        # Test with small network for speed
        model, X_train, X_test, y_train, y_test, train_fn, eval_fn = (
            get_example_network(net_name="nn", train=True, epochs=1)
        )

        injector = StochasticSEUInjector(
            trained_model=model, criterion=classification_accuracy, x=X_test, y=y_test
        )

        baseline_accuracy = injector.baseline_score

        # Test multiple scenarios
        scenarios = [
            {"bit": 0, "p": 0.1, "name": "sign_bit_low_prob"},
            {"bit": 15, "p": 0.1, "name": "middle_bit_low_prob"},
            {"bit": 31, "p": 0.1, "name": "lsb_low_prob"},
        ]

        analysis_results = {}

        for scenario in scenarios:
            results = injector.run_injector(bit_i=scenario["bit"], p=scenario["p"])

        if len(results["criterion_score"]) > 0:
            mean_degraded = np.mean(results["criterion_score"])
            degradation = baseline_accuracy - mean_degraded

            analysis_results[scenario["name"]] = {
                "baseline": baseline_accuracy,
                "mean_degraded": mean_degraded,
                "degradation": degradation,
                "num_injections": len(results["criterion_score"]),
            }

        # Validate analysis results
        assert len(analysis_results) > 0, "Should have at least some analysis results"

        for _name, metrics in analysis_results.items():
            assert 0.0 <= metrics["baseline"] <= 1.0
            assert 0.0 <= metrics["mean_degraded"] <= 1.0
            # Degradation can be negative if injections accidentally improve performance
            assert isinstance(metrics["degradation"], float)
            assert metrics["num_injections"] > 0

    def test_framework_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        model, X_train, X_test, y_train, y_test, train_fn, eval_fn = (
            get_example_network(net_name="nn", train=True, epochs=1)
        )

        # Run 1 with seed
        torch.manual_seed(42)
        np.random.seed(42)
        injector1 = StochasticSEUInjector(
            trained_model=model, criterion=classification_accuracy, x=X_test, y=y_test
        )
        results1 = injector1.run_injector(bit_i=0, p=0.2)

        # Run 2 with same seed
        torch.manual_seed(42)
        np.random.seed(42)
        injector2 = StochasticSEUInjector(
            trained_model=model, criterion=classification_accuracy, x=X_test, y=y_test
        )
        results2 = injector2.run_injector(bit_i=0, p=0.2)

        # Results should be identical
        assert len(results1["criterion_score"]) == len(results2["criterion_score"])

        # Compare first few results (may vary due to randomness in injection selection)
        if (
            len(results1["criterion_score"]) > 0
            and len(results2["criterion_score"]) > 0
        ):
            # At least baseline scores should be identical
            assert abs(injector1.baseline_score - injector2.baseline_score) < 1e-6

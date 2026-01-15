"""Integration tests for core module."""

import torch
import numpy as np
import pytest
from src.core import (
    BatchFeaturizer,
    MolecularGraphBatch,
    SimplifiedGNN,
    ModelConfig,
    PreprocessingPipeline,
)


class TestFeaturizerToModel:
    """Test featurizer output works with model."""

    def test_featurizer_to_model_pipeline(self):
        """Test end-to-end: SMILES -> Featurizer -> Model -> Predictions."""
        featurizer = BatchFeaturizer(num_hops=2, num_workers=1)
        config = ModelConfig(hidden_dim=64, output_dim=1, num_shells=2)
        model = SimplifiedGNN(config)
        model.eval()

        smiles = ["CCO", "CC", "c1ccccc1"]  # ethanol, ethane, benzene
        targets = np.array([[1.0], [2.0], [3.0]])

        batch = featurizer.featurize(smiles, targets)

        with torch.no_grad():
            predictions = model(batch)

        assert predictions.shape == (3, 1)
        assert torch.isfinite(predictions).all()

    def test_with_preprocessing_pipeline(self):
        """Test model with preprocessing applied."""
        featurizer = BatchFeaturizer(num_hops=2, num_workers=1)

        smiles = ["CCO", "CC", "CCC", "CCCC", "CCCCC"]
        targets = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])

        # First featurize to get atomic numbers for SAE fitting
        batch = featurizer.featurize(smiles, targets)

        # Extract atomic numbers per molecule for preprocessing pipeline fit
        atomic_numbers_list = []
        for i in range(batch.num_molecules):
            start = batch.ptr[i].item()
            end = batch.ptr[i + 1].item()
            atom_nums = batch.atom_types[start:end].tolist()
            atomic_numbers_list.append(atom_nums)

        # Fit preprocessing pipeline with SAE
        pipeline = PreprocessingPipeline.fit(
            atomic_numbers_list=atomic_numbers_list,
            targets=targets,
            apply_sae=True,
            sae_subtasks=[0],
            apply_scaling=True,
        )

        # Create model and run forward pass
        config = ModelConfig(hidden_dim=64, output_dim=1, num_shells=2)
        model = SimplifiedGNN(config)
        model.eval()

        with torch.no_grad():
            predictions = model(batch)

        assert predictions.shape == (5, 1)
        assert torch.isfinite(predictions).all()

    def test_batch_device_consistency(self):
        """Test batch and model on same device."""
        featurizer = BatchFeaturizer(num_hops=2, num_workers=1)
        config = ModelConfig(hidden_dim=64, output_dim=1, num_shells=2)
        model = SimplifiedGNN(config)
        model.eval()

        smiles = ["CCO", "CC"]
        targets = np.array([[1.0], [2.0]])

        batch = featurizer.featurize(smiles, targets)

        # Ensure both are on CPU
        model = model.to("cpu")
        batch = batch.to("cpu")

        with torch.no_grad():
            predictions = model(batch)

        # Verify output device matches
        assert predictions.device.type == "cpu"
        assert batch.device.type == "cpu"


class TestModelTraining:
    """Test model can be trained."""

    def test_single_training_step(self):
        """Test single gradient update."""
        featurizer = BatchFeaturizer(num_hops=2, num_workers=1)
        config = ModelConfig(hidden_dim=64, output_dim=1, num_shells=2)
        model = SimplifiedGNN(config)
        model.train()

        smiles = ["CCO", "CC", "CCC"]
        targets = np.array([[1.0], [2.0], [3.0]])

        batch = featurizer.featurize(smiles, targets)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Forward pass
        predictions = model(batch)

        # Compute loss (MSE)
        loss = torch.nn.functional.mse_loss(predictions, batch.targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify loss is positive (finite and > 0)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_multiple_epochs(self):
        """Test training for multiple epochs."""
        featurizer = BatchFeaturizer(num_hops=2, num_workers=1)
        config = ModelConfig(hidden_dim=64, output_dim=1, num_shells=2)
        model = SimplifiedGNN(config)
        model.train()

        smiles = ["CCO", "CC", "CCC", "CCCC", "c1ccccc1"]
        targets = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        batch = featurizer.featurize(smiles, targets)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        num_epochs = 10

        for _ in range(num_epochs):
            # Forward pass
            predictions = model(batch)

            # Compute loss
            loss = torch.nn.functional.mse_loss(predictions, batch.targets)
            losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        initial_loss = losses[0]
        final_loss = losses[-1]

        # Verify loss decreases (at least 10% improvement)
        improvement = (initial_loss - final_loss) / initial_loss
        assert improvement >= 0.10, (
            f"Loss did not decrease by at least 10%: "
            f"initial={initial_loss:.4f}, final={final_loss:.4f}, "
            f"improvement={improvement:.2%}"
        )

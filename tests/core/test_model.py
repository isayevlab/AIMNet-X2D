"""Tests for SimplifiedGNN model."""

import torch
import pytest

from src.core.model import SimplifiedGNN
from src.core.model_config import ModelConfig
from src.core.batch import MolecularGraphBatch


def create_sample_batch() -> MolecularGraphBatch:
    """Create a sample batch for testing."""
    return MolecularGraphBatch(
        atom_types=torch.tensor([6, 6, 8, 6, 7, 1, 1, 1], dtype=torch.int32),
        degrees=torch.tensor([2, 3, 1, 2, 1, 1, 1, 1], dtype=torch.int32),
        hybridizations=torch.tensor([2, 2, 3, 2, 3, 0, 0, 0], dtype=torch.int32),
        hydrogen_counts=torch.tensor([2, 1, 0, 2, 2, 0, 0, 0], dtype=torch.int32),
        batch_idx=torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int64),
        ptr=torch.tensor([0, 3, 8], dtype=torch.int64),
        edge_indices=[
            torch.tensor([[0, 1, 1, 2, 3, 4, 5, 6, 7],
                          [1, 0, 2, 1, 4, 3, 3, 4, 4]], dtype=torch.int64),
            torch.tensor([[0, 2, 3, 5, 6, 7],
                          [2, 0, 5, 3, 3, 3]], dtype=torch.int64),
        ],
        targets=torch.tensor([[1.5], [2.3]]),
        num_molecules=2,
    )


def create_default_config(output_dim: int = 1) -> ModelConfig:
    """Create a default model config for testing."""
    return ModelConfig(
        hidden_dim=64,
        output_dim=output_dim,
        num_shells=2,  # Matches number of edge_indices in sample batch
        num_message_passing_layers=2,
        embedding_dim=32,
        dropout=0.05,
        ffn_num_layers=2,
        attention_num_heads=2,
    )


class TestSimplifiedGNN:
    """Tests for SimplifiedGNN model."""

    def test_model_creation(self):
        """Test that model can be created from config."""
        config = create_default_config()
        model = SimplifiedGNN(config)

        assert model is not None
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, 'config')
        assert model.config == config

    def test_forward_pass(self):
        """Test that forward produces correct shape [num_molecules, output_dim]."""
        config = create_default_config(output_dim=1)
        model = SimplifiedGNN(config)
        model.eval()

        batch = create_sample_batch()

        with torch.no_grad():
            output = model(batch)

        # Output should be [num_molecules, output_dim]
        assert output.shape == (batch.num_molecules, config.output_dim)
        assert output.shape == (2, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_pass_multitask(self):
        """Test that forward works with output_dim > 1 (multitask)."""
        output_dim = 5
        config = create_default_config(output_dim=output_dim)
        model = SimplifiedGNN(config)
        model.eval()

        batch = create_sample_batch()

        with torch.no_grad():
            output = model(batch)

        # Output should be [num_molecules, output_dim]
        assert output.shape == (batch.num_molecules, output_dim)
        assert output.shape == (2, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self):
        """Test that gradients propagate through model."""
        config = create_default_config(output_dim=1)
        model = SimplifiedGNN(config)
        model.train()

        batch = create_sample_batch()

        # Forward pass
        output = model(batch)

        # Compute simple loss and backward
        loss = output.sum()
        loss.backward()

        # Check that gradients flow to embeddings and layers
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients, "No gradients found in model parameters"

    def test_parameter_count(self):
        """Test that model has reasonable parameter count."""
        config = create_default_config()
        model = SimplifiedGNN(config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Check that we have some parameters
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All should be trainable

        # Reasonable bounds for this architecture
        # Small config: should have at least a few thousand params
        assert total_params >= 1000, f"Too few parameters: {total_params}"
        # But not excessive for a small hidden_dim=64 model
        assert total_params <= 500000, f"Too many parameters: {total_params}"

    def test_device_transfer(self):
        """Test that model can be moved to device."""
        config = create_default_config()
        model = SimplifiedGNN(config)

        # Move to CPU (always available)
        model_cpu = model.to('cpu')
        assert next(model_cpu.parameters()).device.type == 'cpu'

        # Check forward still works after device transfer
        batch = create_sample_batch()
        batch_cpu = batch.to('cpu')

        model_cpu.eval()
        with torch.no_grad():
            output = model_cpu(batch_cpu)

        assert output.device.type == 'cpu'
        assert output.shape == (batch.num_molecules, config.output_dim)

    def test_state_dict_save_load(self):
        """Test that model can be saved/loaded via state_dict."""
        config = create_default_config()
        model1 = SimplifiedGNN(config)

        # Set some weights to specific values to verify they load correctly
        with torch.no_grad():
            for param in model1.parameters():
                param.fill_(0.5)

        # Save state dict
        state_dict = model1.state_dict()

        # Create new model and load state dict
        model2 = SimplifiedGNN(config)
        model2.load_state_dict(state_dict)

        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2), f"Mismatch in {name1}"

        # Verify forward produces same output
        batch = create_sample_batch()
        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(batch)
            output2 = model2(batch)

        assert torch.allclose(output1, output2)

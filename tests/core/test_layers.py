"""Tests for simplified model layers."""

import torch
import pytest

from src.core.layers import ShellConvBlock, AttentionPooling, FeedForwardNetwork


class TestShellConvBlock:
    """Test ShellConvBlock for multi-hop message passing."""

    def test_creation(self):
        """Test that ShellConvBlock can be created with expected parameters."""
        layer = ShellConvBlock(
            input_dim=64,
            hidden_dim=128,
            num_shells=3,
            dropout=0.05,
        )
        assert layer is not None
        assert isinstance(layer, torch.nn.Module)

    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        input_dim = 64
        hidden_dim = 128
        num_shells = 3
        num_atoms = 10

        layer = ShellConvBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_shells=num_shells,
            dropout=0.05,
        )

        # Create input tensor
        x = torch.randn(num_atoms, input_dim)

        # Create edge indices for each shell (3 shells)
        edge_indices = [
            torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),  # shell 1
            torch.tensor([[0, 2], [2, 0]], dtype=torch.long),  # shell 2
            torch.tensor([[0, 3], [3, 0]], dtype=torch.long),  # shell 3
        ]

        output = layer(x, edge_indices)

        assert output.shape == (num_atoms, hidden_dim)
        assert output.dtype == x.dtype

    def test_forward_no_edges(self):
        """Test that forward works with empty edge lists."""
        input_dim = 64
        hidden_dim = 128
        num_shells = 3
        num_atoms = 5

        layer = ShellConvBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_shells=num_shells,
            dropout=0.05,
        )

        x = torch.randn(num_atoms, input_dim)

        # Empty edge indices for all shells
        edge_indices = [
            torch.tensor([[], []], dtype=torch.long),
            torch.tensor([[], []], dtype=torch.long),
            torch.tensor([[], []], dtype=torch.long),
        ]

        output = layer(x, edge_indices)

        assert output.shape == (num_atoms, hidden_dim)
        # Should not contain NaN values
        assert not torch.isnan(output).any()


class TestAttentionPooling:
    """Test AttentionPooling for molecule-level aggregation."""

    def test_creation(self):
        """Test that AttentionPooling can be created with expected parameters."""
        pooling = AttentionPooling(
            input_dim=128,
            num_heads=4,
            temperature=1.0,
        )
        assert pooling is not None
        assert isinstance(pooling, torch.nn.Module)

    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        input_dim = 128
        num_heads = 4
        num_atoms = 15
        num_molecules = 3

        pooling = AttentionPooling(
            input_dim=input_dim,
            num_heads=num_heads,
            temperature=1.0,
        )

        # Create atom features
        x = torch.randn(num_atoms, input_dim)

        # Create batch index: 5 atoms in mol 0, 6 atoms in mol 1, 4 atoms in mol 2
        batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

        output = pooling(x, batch_idx)

        assert output.shape == (num_molecules, input_dim)
        assert output.dtype == x.dtype

    def test_single_molecule(self):
        """Test that forward works with a single molecule."""
        input_dim = 64
        num_heads = 2
        num_atoms = 8

        pooling = AttentionPooling(
            input_dim=input_dim,
            num_heads=num_heads,
            temperature=1.0,
        )

        x = torch.randn(num_atoms, input_dim)
        batch_idx = torch.zeros(num_atoms, dtype=torch.long)

        output = pooling(x, batch_idx)

        assert output.shape == (1, input_dim)
        assert not torch.isnan(output).any()


class TestFeedForwardNetwork:
    """Test FeedForwardNetwork."""

    def test_creation(self):
        """Test that FeedForwardNetwork can be created with expected parameters."""
        ffn = FeedForwardNetwork(
            input_dim=128,
            hidden_dim=256,
            output_dim=64,
            num_layers=3,
            dropout=0.05,
        )
        assert ffn is not None
        assert isinstance(ffn, torch.nn.Module)

    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        input_dim = 128
        hidden_dim = 256
        output_dim = 64
        batch_size = 16

        ffn = FeedForwardNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3,
            dropout=0.05,
        )

        x = torch.randn(batch_size, input_dim)
        output = ffn(x)

        assert output.shape == (batch_size, output_dim)
        assert output.dtype == x.dtype

    def test_single_layer(self):
        """Test that forward works with num_layers=1."""
        input_dim = 64
        hidden_dim = 128
        output_dim = 32
        batch_size = 8

        ffn = FeedForwardNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=1,
            dropout=0.05,
        )

        x = torch.randn(batch_size, input_dim)
        output = ffn(x)

        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()
